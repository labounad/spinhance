"""
model.heads.node_head
=====================
Per-node heads for the spin-group query decoder (Architecture Family G):
  shifts      (B, G)        regression (standardized ppm) — NO squashing
  degeneracy  (B, G, C)     class logits

Applied pointwise over the G node embeddings (Linear broadcasts over the node
axis). Shift stays linear/unbounded because the surrogate spectral loss inverts
the z-score standardization downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NodeHead(nn.Module):
    def __init__(self, dim: int, n_deg_classes: int = 8,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden, out_dim))

        self.shift_mlp = mlp(1)
        self.deg_mlp = mlp(n_deg_classes)

    def forward(self, h: torch.Tensor):           # h: (B, G, dim)
        shifts = self.shift_mlp(h).squeeze(-1)     # (B, G)
        deg_logits = self.deg_mlp(h)               # (B, G, C)
        return shifts, deg_logits
