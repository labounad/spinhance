"""Build docs/data/field_sweep.json from PubChem 8-group molecules (the bundle's
spin systems), reusing ALL the sim logic in build_field_sweep.py (fields, sticks,
windows, 3D coords, N_FIELDS). The website ships PubChem molecules throughout, and
the consolidated_v2 records live on the HPC, so run this there:

    PYTHONPATH=. python docs/build_field_sweep_pubchem.py \
        /gpfs/group/shenvi/Users/labounader/spinhance/consolidated_v2/records_train_shuf.json.gz /tmp/field_sweep.json

Per-frame STICKS only (amps normalized per frame); the hero broadens + morphs them
client-side at the chosen field (docs/assets/sweep.js), so resolution and lineshape
are independent of stored data size.
"""
from __future__ import annotations

import base64
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root -> `simulation`
sys.path.insert(0, str(Path(__file__).resolve().parent))       # docs/ -> build_field_sweep
import build_field_sweep as B                                  # noqa: E402
from simulation.graph_io import record_to_arrays, molecule_id  # noqa: E402
from simulation.pyspin.composite import largest_component_spins  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/group/shenvi/Users/labounader/spinhance/consolidated_v2/records_train_shuf.json.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/data/field_sweep.json"
MAX_ELIGIBLE = 4000        # scan enough PubChem records to sample N_MOLECULES from


def main():
    random.seed(B.SAMPLE_SEED)
    elig = []
    for line in gzip.open(SRC, "rt"):
        line = line.strip().rstrip(",")
        if not line or line in "[]":
            continue
        try:
            _labels, shifts, couplings, deg = record_to_arrays(json.loads(line))
        except Exception:
            continue
        if largest_component_spins(couplings, deg) > B.MAX_FRAGMENT_SPINS:
            continue
        lo, hi = min(shifts), max(shifts)
        if lo < 0.3 or hi > 11.5:                      # keep peaks inside the 0-12 window
            continue
        elig.append((json.loads(line), shifts, couplings, deg))
        if len(elig) >= MAX_ELIGIBLE:
            break
    print(f"{len(elig)} eligible PubChem molecules (seed {B.SAMPLE_SEED})", flush=True)
    chosen = random.sample(elig, min(B.N_MOLECULES, len(elig)))

    fields = B.geometric_fields(B.LOW_MHZ, B.HIGH_MHZ, B.N_FIELDS)
    mols, n_xyz = [], 0
    for rank, (rec, shifts, couplings, deg) in enumerate(chosen):
        win_lo, win_hi = B.signal_window(shifts, couplings, deg)
        smiles = rec.get("smiles", "")
        xyz = B.smiles_to_xyz(smiles, title=molecule_id(rec) or "") if smiles else None
        n_xyz += bool(xyz)
        # Sticks for every field, then normalise amplitudes by ONE per-MOLECULE
        # constant (the max line intensity over all frames) — NOT per frame. Per-frame
        # normalisation made every other line halve whenever the tallest line merged
        # (a 50% intensity drop mid-sweep); a single scale preserves the true relative
        # intensities across fields so the area-conserving render grows peaks smoothly.
        raw = [B.molecule_sticks(shifts, couplings, deg, f, win_lo, win_hi) for f in fields]
        mol_amax = max((float(ma.max()) for _mc, ma in raw if len(ma)), default=1.0)
        frames = []
        for mc, ma in raw:
            cen = np.asarray(mc, dtype="<f4")
            amp = np.clip(np.round(ma / mol_amax * 65535.0), 0, 65535).astype("<u2")
            frames.append({"c": base64.b64encode(cen.tobytes()).decode(),
                           "a": base64.b64encode(amp.tobytes()).decode()})
        mols.append({"id": molecule_id(rec), "chembl_id": rec.get("chembl_id"), "smiles": smiles,
                     "n_groups": len(shifts), "degeneracy": [int(d) for d in deg],
                     "shifts": [round(float(s), 3) for s in shifts],
                     "couplings": [[round(float(couplings[i][j]), 1) for j in range(len(shifts))]
                                   for i in range(len(shifts))],
                     "win": [win_lo, win_hi], "xyz": xyz, "frames": frames})
        if (rank + 1) % 10 == 0:
            print(f"  {rank + 1}/{len(chosen)}", flush=True)

    payload = {"meta": {"low_mhz": B.LOW_MHZ, "high_mhz": B.HIGH_MHZ, "fields_mhz": fields,
                        "n_fields": B.N_FIELDS, "ppm_from": B.PPM_FROM, "ppm_to": B.PPM_TO,
                        "linewidth_hz": B.LINEWIDTH_HZ, "format": "sticks",
                        "encoding": "per frame {c: base64 f32 ppm, a: base64 u16/65535}; broaden+morph client-side",
                        "source": f"pyspin; {B.N_MOLECULES} random PubChem 8-group molecules; 3D via RDKit ETKDGv3+MMFF94"},
               "molecules": mols}
    Path(OUT).write_text(json.dumps(payload, separators=(",", ":")))
    import os
    print(f"wrote {OUT}  ({os.path.getsize(OUT)/1e6:.2f} MB, {len(mols)} molecules x "
          f"{B.N_FIELDS} fields, {n_xyz} with 3D)", flush=True)


if __name__ == "__main__":
    main()
