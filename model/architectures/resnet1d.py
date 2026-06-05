"""
model.architectures.resnet1d
============================
ResNet-1D encoder (ported from legacy model.py) + global-average-pool baseline.
Registered as ``resnet1d``. The encoder returns the feature sequence (B, C, L)
so pooling is a separate, swappable component (see attention_pool.py).

GroupNorm (not BatchNorm) keeps normalization independent of batch composition —
important for small/bucketed batches.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.architectures.base import SpinArchitecture
from model.architectures.registry import ARCHITECTURES
from model.heads import TypedMatrixHead
from model.schemas import ModelOutput
from model.schemas.constants import DEFAULT_DEG_VOCAB, N_GROUPS


# ── size presets: (stem_channels, stage_channels, blocks_per_stage) ─────────────
SIZE_PRESETS = {
    "tiny":   (16, (24, 48, 96, 128), (1, 1, 1, 1)),
    "small":  (24, (32, 64, 128, 192), (1, 1, 1, 1)),
    "medium": (32, (64, 128, 256, 512), (2, 2, 2, 2)),
    "large":  (48, (96, 192, 384, 768), (2, 2, 3, 2)),
    # "deep": deeper/wider conv stem for the 500k/3M PubChem regimes (the `med`
    # and `xl` model tiers below). Both tiers share this stem and differ only in
    # the transformer width/depth.
    "deep":   (64, (128, 256, 512, 768), (2, 3, 4, 3)),
}

# ---------------------------------------------------------------------------
# Model tiers — the SINGLE SOURCE OF TRUTH for the data-scaling fleet.
#
# A *tier* fully defines a production model size: the conv-stem preset (above)
# PLUS the transformer width/depth. Configs select a tier via `model.size` and
# normally specify nothing else size-related; the values here are authoritative.
#
#   tier   data tier   params   stem      dim / enc / dec
#   ----   ---------   ------   -------    ---------------
#   light  64k         ~10M     medium     256 / 2 / 4
#   med    500k        ~57M     deep       512 / 4 / 6
#   xl     3M          ~137M    deep       768 / 6 / 8
#
# (`tiny`/`small`/`medium`/`large`/`deep` above remain available as raw conv-stem
#  presets for back-compat / experiments; tier names take precedence.)
# ---------------------------------------------------------------------------
TIER_PRESETS = {
    "light": {"stem": "medium", "dim": 256, "enc_layers": 2, "dec_layers": 4, "n_heads": 8},
    "med":   {"stem": "deep",   "dim": 512, "enc_layers": 4, "dec_layers": 6, "n_heads": 8},
    "xl":    {"stem": "deep",   "dim": 768, "enc_layers": 6, "dec_layers": 8, "n_heads": 8},
}


def _conv(in_c, out_c, k=3, s=1):
    return nn.Conv1d(in_c, out_c, k, stride=s, padding=k // 2, bias=False)


def _norm(c):
    g = min(32, c)
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


class BasicBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv(in_c, out_c, 3, stride)
        self.bn1 = _norm(out_c)
        self.conv2 = _conv(out_c, out_c, 3, 1)
        self.bn2 = _norm(out_c)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False), _norm(out_c))

    def forward(self, x):
        idn = x if self.down is None else self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + idn)


class ResNet1DEncoder(nn.Module):
    """(B, P) or (B, in_channels, P) -> feature sequence (B, C, L)."""

    def __init__(self, stem_channels=32, stage_channels=(64, 128, 256, 512),
                 blocks_per_stage=(2, 2, 2, 2), stem_kernel=15, stem_stride=4,
                 in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, stem_kernel, stride=stem_stride,
                      padding=stem_kernel // 2, bias=False),
            _norm(stem_channels), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1))
        stages = []
        in_c = stem_channels
        for out_c, n in zip(stage_channels, blocks_per_stage):
            stages.append(BasicBlock1D(in_c, out_c, stride=2))
            for _ in range(n - 1):
                stages.append(BasicBlock1D(out_c, out_c, stride=1))
            in_c = out_c
        self.stages = nn.Sequential(*stages)
        self.out_dim = in_c

    def forward(self, x):              # x: (B, P) or (B, in_channels, P)
        if x.dim() == 2:               # (B, P) -> (B, 1, P)
            x = x.unsqueeze(1)
        x = self.stem(x)
        return self.stages(x)          # (B, C, L)


@ARCHITECTURES.register("resnet1d")
class ResNet1DModel(SpinArchitecture):
    """ResNet-1D encoder + global average pool + typed matrix head."""

    def __init__(self, size: str = "medium", n_groups: int = N_GROUPS,
                 n_deg_classes: int = len(DEFAULT_DEG_VOCAB),
                 head_hidden: int = 512, dropout: float = 0.1, **encoder_overrides):
        super().__init__()
        stem, stages, blocks = SIZE_PRESETS[size]
        self.encoder = ResNet1DEncoder(stem_channels=stem, stage_channels=stages,
                                       blocks_per_stage=blocks, **encoder_overrides)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = TypedMatrixHead(self.encoder.out_dim, n_groups, n_deg_classes,
                                    hidden=head_hidden, dropout=dropout)

    def forward(self, x) -> ModelOutput:
        feat = self.encoder(self.spectrum_of(x))     # (B, C, L)
        z = self.pool(feat).squeeze(-1)              # (B, C)
        return self.head(z)
