"""
benchmark_scaling.py
===================
Stress test: grow a FULLY-COUPLED spin system one group at a time and time each
engine, with a hard per-simulation timeout. A fully-coupled chain is the worst
case (no connected-component decomposition to help), so this finds each engine's
practical ceiling.

    # pyspin only (no MNova needed):
    python -m simulation.benchmarks.benchmark_scaling --engines pyspin --start 8 --max 20

    # both engines:
    python -m simulation.benchmarks.benchmark_scaling --engines pyspin mnova \
        --mnova "/Applications/MestReNova.app/Contents/MacOS/MestReNova" --start 8 --max 18

Each size: a chain of N spin-½ groups (degeneracy 1), shifts spread 0.8-8.5 ppm,
vicinal J=7 Hz between neighbours → one connected component of dimension 2^N.
Stops when an engine exceeds --timeout (default 120 s).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulation.mnova_runner import MNOVA_DEFAULT, run_mnova_batch
from simulation.xml_io import matrix_to_xml, save_xml


def build_chain(n: int):
    """A fully-connected coupling chain of n spin-½ groups (degeneracy 1)."""
    import numpy as np
    shifts = list(np.linspace(0.8, 8.5, n))
    couplings = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        couplings[i][i + 1] = couplings[i + 1][i] = 7.0
    degeneracy = [1] * n
    return shifts, couplings, degeneracy


def _pyspin_worker(sh, J, dg, field, q):
    import time as _t
    from simulation.pyspin.composite import simulate_spectrum_composite
    t = _t.perf_counter()
    simulate_spectrum_composite(sh, J, dg, field)
    q.put(_t.perf_counter() - t)


def time_pyspin(sh, J, dg, field, timeout):
    """Run pyspin in a child process; return seconds, or None if it exceeds timeout."""
    # 'spawn' everywhere except Linux: fork + numpy/Accelerate can deadlock on macOS.
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_pyspin_worker, args=(sh, J, dg, field, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return None
    return q.get() if not q.empty() else (time.perf_counter() - t0)


def time_mnova(sh, J, dg, field, mnova_exe, timeout):
    """Run MNova on the system; return seconds, or None if it exceeds timeout."""
    import subprocess
    tmp = Path(tempfile.mkdtemp(prefix="scal_"))
    try:
        xdir = tmp / "xml"; odir = tmp / "txt"
        xdir.mkdir(); odir.mkdir()
        save_xml(matrix_to_xml(sh, J, dg, frequency_mhz=field), xdir / "mol.xml")
        t = time.perf_counter()
        try:
            run_mnova_batch(mnova_exe, xdir, odir, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if not list(odir.glob("*.txt")):
            return None
        return time.perf_counter() - t
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Scaling stress test (fully-coupled chain)")
    p.add_argument("--engines", nargs="+", choices=["pyspin", "mnova"],
                   default=["pyspin"])
    p.add_argument("--mnova", type=Path, default=MNOVA_DEFAULT)
    p.add_argument("--field", type=float, default=90.0)
    p.add_argument("--start", type=int, default=8)
    p.add_argument("--max", type=int, default=20)
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args(argv)

    print(f"Scaling stress: fully-coupled chain, {args.field} MHz, "
          f"hard cutoff {args.timeout:.0f}s\n")
    print(f"  {'groups':>6} {'2^N':>10} {'pyspin':>12} {'mnova':>12}")
    done = set()
    for n in range(args.start, args.max + 1):
        sh, J, dg = build_chain(n)
        cells = {}
        for eng in args.engines:
            if eng in done:
                cells[eng] = "  (skipped)"
                continue
            if eng == "pyspin":
                t = time_pyspin(sh, J, dg, args.field, args.timeout)
            else:
                t = time_mnova(sh, J, dg, args.field, args.mnova, args.timeout)
            if t is None:
                cells[eng] = f">{args.timeout:.0f}s CUTOFF"
                done.add(eng)
            else:
                cells[eng] = f"{t:8.2f}s"
        print(f"  {n:>6} {2**n:>10} "
              f"{cells.get('pyspin',''):>12} {cells.get('mnova',''):>12}")
        if len(done) == len(args.engines):
            print("\nAll requested engines hit the cutoff.")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
