"""
model.architectures.attention_pool
==================================
ResNet-1D encoder + multi-head attention pooling over spectral positions
(Architecture family B). Registered as ``resnet1d_attention_pool``.

Most spectral bins carry little signal; global average pooling dilutes the
informative peaks. Learned pooling queries let the model weight positions before
pooling, while still seeing the whole spectrum.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.architectures.base import SpinArchitecture
from model.architectures.registry import ARCHITECTURES
from model.architectures.resnet1d import SIZE_PRESETS, ResNet1DEncoder
from model.heads import TypedMatrixHead
from model.schemas import ModelOutput
from model.schemas.constants import DEFAULT_DEG_VOCAB, N_GROUPS


class AttentionPool1d(nn.Module):
    """Multi-head attention pooling: K learned queries attend over L positions."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_heads, dim) * dim ** -0.5)
        self.scale = dim ** -0.5

    def forward(self, feat: torch.Tensor) -> torch.Tensor:   # feat (B, C, L)
        x = feat.transpose(1, 2)                              # (B, L, C)
        scores = torch.einsum("blc,hc->bhl", x, self.query) * self.scale
        attn = scores.softmax(dim=-1)                         # (B, H, L)
        pooled = torch.einsum("bhl,blc->bhc", attn, x)        # (B, H, C)
        return pooled.mean(dim=1)                             # (B, C)


@ARCHITECTURES.register("resnet1d_attention_pool")
class ResNet1DAttentionPoolModel(SpinArchitecture):
    def __init__(self, size: str = "medium", n_groups: int = N_GROUPS,
                 n_deg_classes: int = len(DEFAULT_DEG_VOCAB),
                 head_hidden: int = 512, dropout: float = 0.1,
                 pool_heads: int = 4, **encoder_overrides):
        super().__init__()
        stem, stages, blocks = SIZE_PRESETS[size]
        self.encoder = ResNet1DEncoder(stem_channels=stem, stage_channels=stages,
                                       blocks_per_stage=blocks, **encoder_overrides)
        self.pool = AttentionPool1d(self.encoder.out_dim, n_heads=pool_heads)
        self.head = TypedMatrixHead(self.encoder.out_dim, n_groups, n_deg_classes,
                                    hidden=head_hidden, dropout=dropout)

    def forward(self, x) -> ModelOutput:
        feat = self.encoder(self.spectrum_of(x))     # (B, C, L)
        z = self.pool(feat)                          # (B, C)
        return self.head(z)
