"""
model.schemas.batch
===================
Typed training/evaluation batch. Every architecture, loss, renderer, and metric
consumes a ``SpinBatch`` — never a raw dict — so the contract is explicit and the
shapes are checked in one place.

Shapes (B = batch, G = groups, P = spectral points):
    spectrum            (B, P)
    spectrum_ref        (B, P)        clean target spectrum (renderer-noise-free)
    shifts              (B, G)        ppm
    couplings           (B, G, G)     Hz, symmetric, zero diagonal
    coupling_mask       (B, G, G)     1 where a real (ground-truth) coupling exists
    degeneracy_classes  (B, G)        long, index into the degeneracy vocab
    degeneracy_values   (B, G)        int, protons per group (e.g. 3 for CH3)
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch


@dataclass
class RegionTokenBatch:
    """Variable-length spectral support-region tokens (region-model direction).

    Placeholder contract for Architecture families D/E/R; populated by the region
    extractor. ``features`` is (B, R, F); ``mask`` is (B, R) with 1 for real tokens.
    """
    features: torch.Tensor
    mask: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpinBatch:
    spectrum: torch.Tensor
    spectrum_ref: torch.Tensor | None

    shifts: torch.Tensor
    couplings: torch.Tensor
    coupling_mask: torch.Tensor
    degeneracy_classes: torch.Tensor
    degeneracy_values: torch.Tensor

    molecule_ids: list[str]
    smiles: list[str] | None = None

    region_tokens: RegionTokenBatch | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Convenience ────────────────────────────────────────────────────────────

    @property
    def batch_size(self) -> int:
        return int(self.spectrum.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.shifts.shape[1])

    @property
    def device(self) -> torch.device:
        return self.spectrum.device

    def __len__(self) -> int:
        return self.batch_size

    def to(self, device, non_blocking: bool = False) -> "SpinBatch":
        """Return a copy with all tensor fields moved to ``device`` (lists untouched)."""
        moved: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if torch.is_tensor(v):
                moved[f.name] = v.to(device, non_blocking=non_blocking)
            elif isinstance(v, RegionTokenBatch):
                moved[f.name] = RegionTokenBatch(
                    features=v.features.to(device, non_blocking=non_blocking),
                    mask=v.mask.to(device, non_blocking=non_blocking),
                    metadata=v.metadata,
                )
            else:
                moved[f.name] = v
        return replace(self, **moved)

    def validate(self) -> "SpinBatch":
        """Assert the documented shape contract; returns self for chaining."""
        B, P = self.spectrum.shape
        G = self.shifts.shape[1]
        assert self.shifts.shape == (B, G), self.shifts.shape
        assert self.couplings.shape == (B, G, G), self.couplings.shape
        assert self.coupling_mask.shape == (B, G, G), self.coupling_mask.shape
        assert self.degeneracy_classes.shape == (B, G), self.degeneracy_classes.shape
        assert self.degeneracy_values.shape == (B, G), self.degeneracy_values.shape
        if self.spectrum_ref is not None:
            assert self.spectrum_ref.shape == (B, P), self.spectrum_ref.shape
        assert len(self.molecule_ids) == B, (len(self.molecule_ids), B)
        if self.smiles is not None:
            assert len(self.smiles) == B, (len(self.smiles), B)
        return self
