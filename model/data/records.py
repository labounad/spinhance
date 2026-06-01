"""
model.data.records
==================
Adapter from the Task 2/3 on-disk format into the ``records`` dicts the data
layer consumes (ported from legacy data_adapter.py). Reuses
``simulation.graph_io`` so parsing stays consistent with the rest of the project.

Each record:
  mol_id      "mol_000000"  (index-aligned with the spectra filenames)
  shifts      (G,) float ppm
  couplings   (G, G) float Hz, symmetric
  degeneracy  (G,) int
  smiles, chembl_id, inchikey
  n_spins     int = sum(degeneracy)
  spec90_path / spec600_path   (consumed by the dataset)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from simulation.graph_io import read_spin_systems, record_to_arrays

__all__ = ["load_records", "load_pubchem_records"]


def load_records(spin_systems_json, spectra_root, fields=(90,), require_spectra=True):
    spectra_root = Path(spectra_root)
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
        (records if ok else missing).append(d if ok else stem)
    if missing:
        print(f"[records] WARNING: {len(missing)} molecules missing spectra "
              f"(e.g. {missing[:3]}) — skipped.")
    return records


def load_pubchem_records(spin_systems_json, max_mol=0, allowed_degeneracy=None):
    """Records for the PubChem 3M+ regime: spectra come from stacked ``part_<k>.npy``
    shards (``model.data.stacked_spectra.StackedSpectra``), not per-molecule files.
    Each record carries ``row`` = its global index in record order, which is also
    its row in the concatenated shards; the dataset's ``spectra_source[row]`` fetches
    the spectrum. ``max_mol`` truncates for quick prelim runs (streaming, so it stops
    early without parsing the whole file).

    Molecules whose degeneracy contains a value outside the model's vocab (default
    ``DEFAULT_DEG_VOCAB``) are skipped — ``row`` stays the global index so the
    spectrum mapping is unaffected. In PubChem this drops only a handful (deg 5/8,
    ~8 groups in 25.6M); the model has no class for them and can't learn them from
    a few examples, so filtering beats expanding the vocab (keeps n_deg_classes ==
    the 64k production model)."""
    from model.schemas.constants import DEFAULT_DEG_VOCAB
    allowed = set(allowed_degeneracy or DEFAULT_DEG_VOCAB)
    records = []
    skipped = 0
    for idx, rec in read_spin_systems(spin_systems_json):
        _labels, shifts, couplings, degeneracy = record_to_arrays(rec)
        if any(int(d) not in allowed for d in degeneracy):
            skipped += 1
            continue
        records.append({
            "mol_id": f"mol_{idx:06d}",
            "row": idx,
            "shifts": np.asarray(shifts, dtype=float),
            "couplings": np.asarray(couplings, dtype=float),
            "degeneracy": np.asarray(degeneracy, dtype=int),
            "smiles": rec.get("smiles"),
            "chembl_id": rec.get("chembl_id"),
            "inchikey": rec.get("inchikey"),
            "n_spins": int(sum(degeneracy)),
        })
        if max_mol and len(records) >= max_mol:
            break
    if skipped:
        print(f"[pubchem] filtered {skipped} molecules with out-of-vocab degeneracy "
              f"(vocab {sorted(allowed)})")
    return records
