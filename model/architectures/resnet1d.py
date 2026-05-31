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
    """(B, P) -> feature sequence (B, C, L)."""

    def __init__(self, stem_channels=32, stage_channels=(64, 128, 256, 512),
                 blocks_per_stage=(2, 2, 2, 2), stem_kernel=15, stem_stride=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_channels, stem_kernel, stride=stem_stride,
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

    def forward(self, x):              # x: (B, P)
        x = x.unsqueeze(1)             # (B, 1, P)
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
