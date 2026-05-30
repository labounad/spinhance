"""
benchmark_pyspin.py
==================
Throughput + parallel-scaling benchmark for the pure-Python (pyspin) engine.

Replicates one source XML into a mini-dataset of ``--n`` molecules, then times
``run_pyspin_batch`` over each worker count in ``--workers-sweep``, reporting
sims/s and speedup vs 1 worker. License-free; no MNova.

    python -m simulation.benchmarks.benchmark_pyspin \
        --source_xml simulation/examples/R_5_methylcyclohexenone.xml \
        --n 200 --fields 90 600 --workers-sweep 1 2 4 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulation.pyspin.batch import run_pyspin_batch
from simulation.xml_io import matrix_to_xml, save_xml, xml_to_matrix


def run(source_xml: Path, n: int, fields, workers_sweep, out_dir: Path) -> list[dict]:
    m = xml_to_matrix(source_xml)
    xml_dir = out_dir / "xmls"
    xml_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        save_xml(matrix_to_xml(m["shifts"], m["couplings"], m["degeneracy"],
                               frequency_mhz=fields[0]),
                 xml_dir / f"mol_{i:05d}.xml")
    total_sims = n * len(fields)
    print(f"Dataset: {n} molecules × {len(fields)} fields = {total_sims} sims "
          f"(source: {source_xml.name})\n")

    rows = []
    base = None
    for w in workers_sweep:
        spectra = out_dir / f"run_w{w}"
        shutil.rmtree(spectra, ignore_errors=True)
        t = time.perf_counter()
        res = run_pyspin_batch(xml_dir, spectra, fields_mhz=fields, workers=w)
        dt = time.perf_counter() - t
        sps = total_sims / dt
        if base is None:
            base = dt
        rows.append({"workers": w, "seconds": round(dt, 2),
                     "sims_per_s": round(sps, 1), "speedup": round(base / dt, 2),
                     "ok": res["succeeded"], "total": res["tasks"]})

    print("\n============ pyspin scaling ============")
    print(f"  {'workers':>7} {'sec':>8} {'sims/s':>9} {'speedup':>8}  ok/total")
    for r in rows:
        print(f"  {r['workers']:>7} {r['seconds']:>8} {r['sims_per_s']:>9} "
              f"{r['speedup']:>7}x  {r['ok']}/{r['total']}")
    best = max(rows, key=lambda r: r["sims_per_s"])
    print(f"  best: {best['sims_per_s']} sims/s at {best['workers']} workers")
    print(f"  -> 100k molecules x {len(fields)} fields ~ "
          f"{100000*len(fields)/best['sims_per_s']/3600:.1f} h")
    print("========================================")
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark the pyspin parallel engine")
    p.add_argument("--source_xml", type=Path, required=True)
    p.add_argument("--n", type=int, default=200, help="Molecules in the mini-dataset")
    p.add_argument("--fields", type=float, nargs="+", default=[90.0, 600.0])
    p.add_argument("--workers-sweep", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--out_dir", type=Path, default=Path("/tmp/spinhance_pyspin_bench"))
    args = p.parse_args(argv)

    src = args.source_xml
    if not src.exists():
        cand = _REPO_ROOT / src
        if cand.exists():
            src = cand
        else:
            print(f"ERROR: source XML not found: {args.source_xml}", file=sys.stderr)
            return 2

    tmp = Path(tempfile.mkdtemp(prefix="pyspin_bench_")) if str(args.out_dir) == "" else args.out_dir
    try:
        run(src, args.n, args.fields, args.workers_sweep, tmp)
    finally:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
