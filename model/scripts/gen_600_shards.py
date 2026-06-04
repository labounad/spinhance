"""
Generate 600 MHz exact spectra for the bundle's spin systems, stacked into
``part_<k>.npy`` shards ALIGNED row-for-row with the bundle's records.json.gz
(so the surrogate's ``record["row"]`` indexes straight into them).

Why re-simulate (vs re-broadening the 90 MHz line list): second-order coupling
(J/Δν) is field-dependent, so the transition list itself changes with field —
the 90 MHz peaks can't simply be re-broadened at 600.

One SLURM array task per shard: task k processes records [k*chunk, (k+1)*chunk)
of the JSONL bundle, in order, and writes part_{k:05d}.npy. On the rare
per-molecule failure a zero row is written so row alignment is preserved.

    python -m model.scripts.gen_600_shards \
        --records /gpfs/group/shenvi/Users/labounader/spinhance/legacy/bundle_500k_s0/records.json.gz \
        --out /gpfs/group/shenvi/Users/labounader/spinhance/legacy/bundle_600_2p16 \
        --shard-index $SLURM_ARRAY_TASK_ID --chunk 50000 \
        --field 600 --points 65536 --linewidth 0.35 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from simulation.graph_io import read_spin_systems, record_to_arrays
from simulation.pyspin.composite import simulate_spectrum_composite

# module-level config for the worker (set in main before the Pool is created)
_CFG = {"field": 600.0, "points": 65536, "linewidth": 0.7, "eta": 0.8}


def _simulate_one(rec):
    """rec -> (P,) float32 spectrum, or zeros on failure (alignment-preserving)."""
    P = _CFG["points"]
    try:
        _l, shifts, couplings, deg = record_to_arrays(rec)
        _, y = simulate_spectrum_composite(
            np.asarray(shifts, float), np.asarray(couplings, float), list(deg),
            _CFG["field"], points=P, linewidth_hz=_CFG["linewidth"], eta=_CFG["eta"])
        return np.asarray(y, np.float32), True
    except Exception:
        return np.zeros(P, np.float32), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="bundle records.json.gz (JSONL, ordered)")
    ap.add_argument("--out", required=True, help="output bundle dir for the 600 shards")
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=50000, help="records per shard (match the bundle)")
    ap.add_argument("--field", type=float, default=600.0)
    ap.add_argument("--points", type=int, default=65536)
    ap.add_argument("--linewidth", type=float, default=0.7)
    ap.add_argument("--eta", type=float, default=0.8, help="Lorentzian fraction (pseudo-Voigt)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    _CFG["field"], _CFG["points"] = a.field, a.points
    _CFG["linewidth"], _CFG["eta"] = a.linewidth, a.eta
    os.makedirs(a.out, exist_ok=True)
    out_npy = os.path.join(a.out, f"part_{a.shard_index:05d}.npy")
    if os.path.exists(out_npy):
        print(f"part {a.shard_index:05d} exists -> skip", flush=True)
        return

    lo, hi = a.shard_index * a.chunk, (a.shard_index + 1) * a.chunk
    recs = [rec for i, rec in read_spin_systems(a.records) if lo <= i < hi]
    print(f"shard {a.shard_index:05d}: {len(recs)} records [{lo},{hi}) | "
          f"field={a.field} points={a.points} lw={a.linewidth} eta={a.eta} "
          f"workers={a.workers}", flush=True)
    if not recs:
        print("no records in range -> nothing to do", flush=True)
        return

    with Pool(a.workers) as pool:
        results = pool.map(_simulate_one, recs, chunksize=64)
    specs = np.stack([r[0] for r in results])
    n_fail = sum(1 for r in results if not r[1])

    tmp = os.path.join(a.out, f"part_{a.shard_index:05d}.tmp")
    np.save(tmp, specs)
    os.replace(f"{tmp}.npy", out_npy)
    with open(os.path.join(a.out, f"part_{a.shard_index:05d}.meta.json"), "w") as g:
        json.dump({"shard": a.shard_index, "lo": lo, "hi": hi, "n": len(recs),
                   "n_fail": n_fail, "field": a.field, "points": a.points,
                   "linewidth_hz": a.linewidth, "eta": a.eta}, g)
    print(f"part {a.shard_index:05d}: {len(recs)} specs, {n_fail} failed (zero-filled) "
          f"-> {out_npy}", flush=True)


if __name__ == "__main__":
    main()
