"""
aws_trainer.model
==================
Model size registry + attention-neck encoder.

Imports BasicBlock1D from model.model to avoid duplicating the ResNet block.
Larger configs (medium/large) are the main addition for the 100k dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.model import BasicBlock1D, SpinHanceModel


def _norm(c: int) -> nn.GroupNorm:
    g = min(32, c)
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


# ── Encoder (same structure as model.ResNet1DEncoder; adds optional attn neck) ─

class ResNet1DEncoderV2(nn.Module):
    """ResNet-1D encoder with optional Transformer attention neck.

    The attention neck sits between the conv stages and global pool, over the
    spatially-reduced feature map (P/128 ≈ 128 tokens for P=16384).  1-2 layers
    of pre-norm Transformer here capture long-range coupling structure at low cost.
    """

    def __init__(self, stem_channels: int = 64,
                 stage_channels: tuple = (128, 256, 512, 512),
                 blocks_per_stage: tuple = (2, 2, 3, 3),
                 stem_kernel: int = 15, stem_stride: int = 4,
                 attention_layers: int = 0, attention_heads: int = 8,
                 dropout: float = 0.1):
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

        self.attn: nn.TransformerEncoder | None = None
        if attention_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=in_c, nhead=attention_heads,
                dim_feedforward=in_c * 4, dropout=dropout,
                batch_first=True, norm_first=True)
            self.attn = nn.TransformerEncoder(layer, num_layers=attention_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = in_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)             # (B, 1, P)
        x = self.stem(x)
        x = self.stages(x)             # (B, C, L)
        if self.attn is not None:
            x = self.attn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.pool(x).squeeze(-1)  # (B, C)


# ── Registry ──────────────────────────────────────────────────────────────────

_CONFIGS: dict[str, dict] = {
    "tiny": dict(
        stem_channels=24, stage_channels=(32, 64, 128, 192),
        blocks_per_stage=(1, 1, 1, 1), attention_layers=0, head_hidden=256),
    "small": dict(
        stem_channels=32, stage_channels=(64, 128, 256, 512),
        blocks_per_stage=(2, 2, 2, 2), attention_layers=0, head_hidden=512),
    "medium": dict(
        stem_channels=64, stage_channels=(128, 256, 512, 512),
        blocks_per_stage=(2, 2, 3, 3), attention_layers=0, head_hidden=1024),
    "large": dict(
        stem_channels=64, stage_channels=(128, 256, 512, 1024),
        blocks_per_stage=(3, 4, 6, 3), attention_layers=0, head_hidden=1024),
    "medium-attn": dict(
        stem_channels=64, stage_channels=(128, 256, 512, 512),
        blocks_per_stage=(2, 2, 3, 3), attention_layers=2, head_hidden=1024),
    "large-attn": dict(
        stem_channels=64, stage_channels=(128, 256, 512, 1024),
        blocks_per_stage=(3, 4, 6, 3), attention_layers=2, head_hidden=1024),
}


def build_model(cfg) -> SpinHanceModel:
    """Build a SpinHanceModel from a VAWSConfig."""
    from model.targets import DegeneracyVocab
    mc = _CONFIGS[cfg.model_size]
    enc = ResNet1DEncoderV2(
        stem_channels=mc["stem_channels"],
        stage_channels=mc["stage_channels"],
        blocks_per_stage=mc["blocks_per_stage"],
        attention_layers=mc["attention_layers"],
        dropout=cfg.dropout,
    )
    return SpinHanceModel(
        n_groups=cfg.n_groups,
        n_deg_classes=len(DegeneracyVocab()),
        encoder=enc,
        head_hidden=mc["head_hidden"],
        dropout=cfg.dropout,
    )


def param_count(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n / 1e6:.2f}M"
