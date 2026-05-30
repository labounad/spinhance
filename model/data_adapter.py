"""
model.data_adapter
==================
Adapter from the Task 2/3 on-disk format into the ``records`` dicts the Task 4
pipeline (splits / dataset / train) consumes. Reuses ``simulation.graph_io`` so
the parsing stays consistent with the rest of the project.

Inputs
------
  spin_systems_json : mol_to_spin_system/data/spin_systems.json  (Task 2 graphs)
  spectra_root      : simulation/data/spectra/  containing <field>MHz/mol_*.npy

Each record:
  mol_id      "mol_000000"  (index-aligned with the spectra filenames)
  shifts      (G,) float ppm
  couplings   (G, G) float Hz, symmetric
  degeneracy  (G,) int
  smiles, chembl_id, inchikey
  n_spins     int  = sum(degeneracy)   (cost proxy: bounds the renderer Hilbert space)
  spec90_path / spec600_path           (consumed by SpectrumMatrixDataset)

Scaffold for splits is computed lazily by ``splits.make_splits`` from ``smiles``
(needs RDKit). In a torch/RDKit-free environment pass ``compute_scaffold=False``
to fall back to molecule-level + matrix-dedup splitting.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from simulation.graph_io import read_spin_systems, record_to_arrays, molecule_id

__all__ = ["load_records", "renderable_mask"]


def load_records(spin_systems_json, spectra_root, fields=(90, 600),
                 require_spectra=True):
    spectra_root = Path(spectra_root)
    # Check tar existence once per field — avoids 2×N stat() calls in the loop
    _tar_exists = {f: (spectra_root / f"{int(f)}MHz" / "mol_all.tar.gz").exists()
                   for f in fields}
    records = []
    missing = []
    for idx, rec in read_spin_systems(spin_systems_json):
        labels, shifts, couplings, degeneracy = record_to_arrays(rec)
        stem = f"mol_{idx:06d}"
        d = {
            "mol_id": stem,
            "shifts": np.asarray(shifts, dtype=float),
            "couplings": np.asarray(couplings, dtype=float),
            "degeneracy": np.asarray(degeneracy, dtype=int),
            "smiles": rec.get("smiles"),
            "chembl_id": rec.get("chembl_id"),
            "inchikey": rec.get("inchikey"),
            "n_spins": int(sum(degeneracy)),
        }
        ok = True
        for f in fields:
            p = spectra_root / f"{int(f)}MHz" / f"{stem}.npy"
            d[f"spec{int(f)}_path"] = str(p)
            if require_spectra and not _tar_exists[f] and not p.exists():
                ok = False
        if ok:
            records.append(d)
        else:
            missing.append(stem)
    if missing:
        print(f"[data_adapter] WARNING: {len(missing)} molecules missing spectra "
              f"(e.g. {missing[:3]}) — skipped.")
    return records


def renderable_mask(records, max_block=2048):
    """Boolean list: molecules whose largest Mz-block (the composite renderer's
    actual eigh size) is <= ``max_block``. With manifold reduction this covers
    ~100% of the preliminary set (vs ~89% for the old explicit 2^N renderer);
    the threshold just caps per-step Stage-2 cost. Uses composite_diff.max_block_dim."""
    from model.composite_diff import max_block_dim
    return [max_block_dim(r["degeneracy"]) <= max_block for r in records]
