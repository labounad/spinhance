"""
model.schemas.outputs
=====================
Typed model output. Every architecture returns a ``ModelOutput``. Architectures
may compute couplings in edge-list form ``(B, E)`` internally, but losses and
metrics consume the normalized matrix form ``(B, G, G)`` via ``coupling_matrix``
/ ``presence_matrix``.

Shapes:
    shifts                    (B, G)
    coupling_values           (B, G, G) or (B, E)   E = G*(G-1)/2 upper-triangle
    coupling_presence_logits  (B, G, G) or (B, E)
    degeneracy_logits         (B, G, C)             C = len(degeneracy vocab)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _edges_to_matrix(edges: torch.Tensor, n_groups: int) -> torch.Tensor:
    """(B, E) upper-triangle -> (B, G, G) symmetric, zero diagonal."""
    B = edges.shape[0]
    iu = torch.triu_indices(n_groups, n_groups, offset=1, device=edges.device)
    M = torch.zeros(B, n_groups, n_groups, dtype=edges.dtype, device=edges.device)
    M[:, iu[0], iu[1]] = edges
    M[:, iu[1], iu[0]] = edges
    return M


@dataclass
class ModelOutput:
    shifts: torch.Tensor
    coupling_values: torch.Tensor
    coupling_presence_logits: torch.Tensor
    degeneracy_logits: torch.Tensor

    node_embeddings: torch.Tensor | None = None
    edge_embeddings: torch.Tensor | None = None
    attention_maps: dict[str, torch.Tensor] = field(default_factory=dict)
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return int(self.shifts.shape[1])

    def _as_matrix(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3:                       # already (B, G, G)
            return t
        if t.dim() == 2:                       # (B, E) edge list
            return _edges_to_matrix(t, self.n_groups)
        raise ValueError(f"expected (B,G,G) or (B,E), got shape {tuple(t.shape)}")

    def coupling_matrix(self) -> torch.Tensor:
        """Symmetric (B, G, G) coupling magnitudes regardless of internal format."""
        return self._as_matrix(self.coupling_values)

    def presence_matrix(self) -> torch.Tensor:
        """Symmetric (B, G, G) presence logits regardless of internal format."""
        return self._as_matrix(self.coupling_presence_logits)

    def validate(self, n_groups: int | None = None) -> "ModelOutput":
        G = n_groups or self.n_groups
        B = self.shifts.shape[0]
        assert self.shifts.shape == (B, G), self.shifts.shape
        assert self.degeneracy_logits.shape[:2] == (B, G), self.degeneracy_logits.shape
        cm, pm = self.coupling_matrix(), self.presence_matrix()
        assert cm.shape == (B, G, G), cm.shape
        assert pm.shape == (B, G, G), pm.shape
        for name, t in (("shifts", self.shifts), ("coupling_values", cm),
                        ("presence", pm), ("degeneracy_logits", self.degeneracy_logits)):
            assert torch.isfinite(t).all(), f"non-finite values in {name}"
        return self
