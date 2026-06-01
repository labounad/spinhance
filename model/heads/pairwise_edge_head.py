"""
model.heads.pairwise_edge_head
==============================
Symmetric pairwise coupling head for the spin-group query decoder (Family G).
From G node embeddings it builds the E = G*(G-1)/2 upper-triangle edges and
predicts coupling magnitude + presence per edge:

  edge_ij = MLP([h_i + h_j, |h_i - h_j|])    # symmetric by construction (i<->j invariant)

Edge order is exactly ``torch.triu_indices(G, G, offset=1)`` so the returned
(B, E) tensors map straight onto ``ModelOutput``'s edge-list -> (B, G, G) matrix
conversion (which uses the same triu order). Magnitude is linear/unbounded
(standardized Hz); presence is a binary logit.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseEdgeHead(nn.Module):
    def __init__(self, dim: int, n_groups: int = 8,
                 hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        iu = torch.triu_indices(n_groups, n_groups, offset=1)   # (2, E)
        self.register_buffer("ei", iu[0])                       # (E,)
        self.register_buffer("ej", iu[1])                       # (E,)
        self.trunk = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.jmag_out = nn.Linear(hidden, 1)
        self.jpres_out = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor):                # h: (B, G, dim)
        hi = h[:, self.ei, :]                          # (B, E, dim)
        hj = h[:, self.ej, :]                          # (B, E, dim)
        feat = torch.cat([hi + hj, (hi - hj).abs()], dim=-1)   # (B, E, 2*dim) symmetric
        e = self.trunk(feat)                           # (B, E, hidden)
        jmag = self.jmag_out(e).squeeze(-1)            # (B, E)
        jpres = self.jpres_out(e).squeeze(-1)          # (B, E)
        return jmag, jpres, e
