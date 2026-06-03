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


def load_pubchem_records(spin_systems_json, max_mol=0, allowed_degeneracy=None,
                         sample_n=0, sample_seed=0):
    """Records for the PubChem 3M+ regime: spectra come from stacked ``part_<k>.npy``
    shards (``model.data.stacked_spectra.StackedSpectra``), not per-molecule files.
    Each record carries ``row`` = its global index in record order, which is also
    its row in the concatenated shards; the dataset's ``spectra_source[row]`` fetches
    the spectrum (so a sampled/filtered subset still maps to the full stacked set).

    Three subset modes:
      * default        — all valid records.
      * ``max_mol``    — first N valid records (streams, stops early).
      * ``sample_n``   — a uniform RANDOM sample of N valid records via reservoir
                         sampling (single pass, O(N) memory, seeded). Use this for
                         a representative "light" subset rather than the first-N
                         block (which can be biased by PubChem CID ordering).

    Molecules whose degeneracy contains a value outside the model's vocab (default
    ``DEFAULT_DEG_VOCAB``) are skipped — ``row`` stays the global index so the
    spectrum mapping is unaffected. In PubChem this drops only a handful (deg 5/8,
    ~8 groups in 25.6M); the model has no class for them and can't learn them from
    a few examples, so filtering beats expanding the vocab (keeps n_deg_classes ==
    the 64k production model)."""
    import random
    from model.schemas.constants import DEFAULT_DEG_VOCAB
    allowed = set(allowed_degeneracy or DEFAULT_DEG_VOCAB)
    rng = random.Random(sample_seed) if sample_n > 0 else None
    records, reservoir = [], []
    skipped, kept = 0, 0
    for idx, rec in read_spin_systems(spin_systems_json):
        _labels, shifts, couplings, degeneracy = record_to_arrays(rec)
        if any(int(d) not in allowed for d in degeneracy):
            skipped += 1
            continue
        # ``row`` indexes the stacked spectra. Default to the file position, but
        # honor an explicit ``row`` field when present — this lets a filtered
        # subset file (e.g. a train-only split, with the held-out test removed)
        # still index into the FULL stacked parts without re-materializing them.
        row = int(rec.get("row", idx))
        r = {
            "mol_id": f"mol_{row:06d}",
            "row": row,
            "shifts": np.asarray(shifts, dtype=float),
            "couplings": np.asarray(couplings, dtype=float),
            "degeneracy": np.asarray(degeneracy, dtype=int),
            "smiles": rec.get("smiles"),
            "chembl_id": rec.get("chembl_id"),
            "inchikey": rec.get("inchikey"),
            "n_spins": int(sum(degeneracy)),
        }
        if sample_n > 0:                       # reservoir sampling (uniform, seeded)
            kept += 1
            if len(reservoir) < sample_n:
                reservoir.append(r)
            else:
                j = rng.randint(0, kept - 1)
                if j < sample_n:
                    reservoir[j] = r
        else:
            records.append(r)
            if max_mol and len(records) >= max_mol:
                break
    out = reservoir if sample_n > 0 else records
    if skipped:
        print(f"[pubchem] filtered {skipped} molecules with out-of-vocab degeneracy "
              f"(vocab {sorted(allowed)})")
    if sample_n > 0:
        print(f"[pubchem] reservoir-sampled {len(out)} of {kept} valid molecules "
              f"(seed {sample_seed})")
    return out
