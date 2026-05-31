"""
model.data.collate
==================
Stack per-sample dicts from SpectrumMatrixDataset into a typed ``SpinBatch``.
"""
from __future__ import annotations

import torch

from model.schemas import SpinBatch

__all__ = ["collate_spin_batch"]


def collate_spin_batch(samples) -> SpinBatch:
    def stack(key):
        return torch.stack([s[key] for s in samples])

    return SpinBatch(
        spectrum=stack("spectrum"),
        spectrum_ref=stack("spectrum_ref"),
        shifts=stack("shifts"),
        couplings=stack("couplings"),
        coupling_mask=stack("coupling_mask"),
        degeneracy_classes=stack("degeneracy_classes"),
        degeneracy_values=stack("degeneracy_values"),
        molecule_ids=[s["mol_id"] for s in samples],
        smiles=[s.get("smiles") for s in samples],
        metadata={"bucket_keys": [s["bucket_key"] for s in samples]},
    )
