"""
model.heads.typed_matrix_head
=============================
Four typed heads on a pooled embedding -> ModelOutput (Decision 2):
  shifts      (B, G)           regression (standardized ppm)
  couplings   (B, E)           regression (standardized Hz, edge-list upper-tri)
  presence    (B, E)           binary logits
  degeneracy  (B, G, C)        class logits

E = G*(G-1)/2. ModelOutput normalizes the edge-list couplings/presence to the
(B, G, G) matrix form losses/metrics consume.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.schemas import ModelOutput


class TypedMatrixHead(nn.Module):
    def __init__(self, emb_dim: int, n_groups: int = 8, n_deg_classes: int = 8,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.G = n_groups
        self.n_pairs = n_groups * (n_groups - 1) // 2
        self.n_deg = n_deg_classes

        def head(out_dim):
            return nn.Sequential(
                nn.Linear(emb_dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden, out_dim))

        self.shift_head = head(self.G)
        self.jmag_head = head(self.n_pairs)
        self.jpres_head = head(self.n_pairs)
        self.deg_head = head(self.G * self.n_deg)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        B = z.shape[0]
        return ModelOutput(
            shifts=self.shift_head(z),                                   # (B, G)
            coupling_values=self.jmag_head(z),                           # (B, E)
            coupling_presence_logits=self.jpres_head(z),                 # (B, E)
            degeneracy_logits=self.deg_head(z).view(B, self.G, self.n_deg),
        )
