"""
model.scripts.bundle_subset
===========================
Materialize a RANDOM molecule subset of the 3M+ PubChem set into a compact,
contiguous bundle — ONCE, on CPU — so the GPU job never scans the 3200 shards.

Reservoir-samples N valid molecules (degeneracy in vocab) from the records stream,
reads only their spectra (sequential shard reads), and writes:
  <out>/part_00000.npy ... part_NNNNN.npy   (sampled spectra, contiguous, chunked)
  <out>/records.json.gz                      (the sampled records, JSONL, same order)
The bundle is then a normal stacked-shard dataset (data.parts=<out>,
data.records=<out>/records.json.gz, sample_n=0) — a few GB that fits RAM, so the
GPU job can mmap one small file or preload it safely (no 196GB read, no 210GB
array, no GPFS I/O storm from concurrent jobs).

Usage:
  python -m model.scripts.bundle_subset --records <gz> --parts <dir> \
    --out <dir> --sample_n 500000 --seed 0 [--chunk 50000]
"""
from __future__ import annotations

import argparse, gzip, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np

from simulation.graph_io import read_spin_systems, record_to_arrays
from model.data.stacked_spectra import StackedSpectra
from model.schemas.constants import DEFAULT_DEG_VOCAB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample_n", type=int, default=500000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=50000, help="rows per output part")
    args = ap.parse_args()

    allowed = set(DEFAULT_DEG_VOCAB)
    rng = random.Random(args.seed)
    N = args.sample_n

    # 1) reservoir-sample N valid raw records, keeping their global index
    res = []   # list[(global_idx, raw_rec)]
    kept = 0
    for idx, rec in read_spin_systems(args.records):
        _l, _s, _c, dg = record_to_arrays(rec)
        if any(int(d) not in allowed for d in dg):
            continue
        kept += 1
        if len(res) < N:
            res.append((idx, rec))
        else:
            j = rng.randint(0, kept - 1)
            if j < N:
                res[j] = (idx, rec)
    print(f"[bundle] sampled {len(res)} of {kept} valid molecules (seed {args.seed})", flush=True)

    # 2) read the sampled spectra via sequential whole-shard reads (one pass)
    ss = StackedSpectra(args.parts)
    P = int(np.load(ss.files[0], mmap_mode="r").shape[1])
    out = np.empty((len(res), P), dtype=np.float32)
    by_shard = defaultdict(list)   # shard k -> [(out_pos, local_row)]
    for pos, (idx, _rec) in enumerate(res):
        k = int(np.searchsorted(ss.offsets, idx, side="right") - 1)
        by_shard[k].append((pos, idx - int(ss.offsets[k])))
    for n, (k, items) in enumerate(sorted(by_shard.items())):
        part = np.load(ss.files[k])           # sequential read
        for pos, lr in items:
            out[pos] = part[lr]
        del part
        if (n + 1) % 500 == 0:
            print(f"[bundle]   read {n + 1}/{len(by_shard)} shards", flush=True)

    # 3) write the bundle: chunked spectra parts + records JSONL (same order)
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    nchunks = 0
    for ci, start in enumerate(range(0, len(out), args.chunk)):
        np.save(od / f"part_{ci:05d}.npy", out[start:start + args.chunk])
        nchunks += 1
    with gzip.open(od / "records.json.gz", "wt") as f:
        for _idx, rec in res:
            f.write(json.dumps(rec) + "\n")
    gb = out.nbytes / 1e9
    print(f"[bundle] wrote {len(res)} spectra in {nchunks} parts (~{gb:.1f} GB) + records to {od}", flush=True)


if __name__ == "__main__":
    main()
