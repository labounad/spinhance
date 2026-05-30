"""
benchmark_fields.py
===================
Throughput benchmark for the MNova spin simulator. Takes ONE source spin-system
XML, patches it to ``n`` geometrically-spaced spectrometer frequencies between
``fmin`` and ``fmax`` MHz, and simulates them all in a single MestReNova launch.

Geometric spacing means the frequency points are closer together at low field
and farther apart at high field (constant *ratio*, growing *gap*) — which mirrors
where strong-coupling effects vary most.

Timing model
------------
A single MNova launch costs ``startup + n * per_sim``. We measure two launches —
a small calibration run of ``calib_n`` files and the full run of ``n`` files —
and solve the two-point line for ``per_sim`` and ``startup``:

    per_sim = (t_full - t_calib) / (n - calib_n)
    startup =  t_calib - calib_n * per_sim

Run it
------
    python -m simulation.benchmarks.benchmark_fields \
        --source_xml "predicted_mnova_1h (10).xml" \
        --mnova "/Applications/MestReNova.app/Contents/MacOS/MestReNova" \
        --n 100 --fmin 40 --fmax 1200

Requires MestReNova (and its scripts folder registered). The pure
``geometric_frequencies`` helper has no such dependency and is unit-tested.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow `python simulation/benchmarks/benchmark_fields.py` as well as `-m`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulation.mnova_runner import MNOVA_DEFAULT, run_mnova_batch, run_mnova_parallel
from simulation.xml_io import patch_frequency, save_xml

__all__ = ["geometric_frequencies", "run_benchmark"]


def geometric_frequencies(fmin: float, fmax: float, n: int) -> list[float]:
    """Return ``n`` geometrically-spaced frequencies in ``[fmin, fmax]``.

    Spacing is denser near ``fmin`` and sparser near ``fmax`` (the gap between
    consecutive points grows monotonically). Endpoints are included exactly.

    Raises
    ------
    ValueError
        If ``n < 2`` or ``fmin``/``fmax`` are not positive with ``fmin < fmax``.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if not (0 < fmin < fmax):
        raise ValueError("require 0 < fmin < fmax")
    return [float(x) for x in np.geomspace(fmin, fmax, n)]


def _mnova_already_running() -> bool:
    """True if a MestReNova process appears to be running (macOS/Linux)."""
    import subprocess
    try:
        out = subprocess.run(["pgrep", "-fil", "mestrenova"],
                             capture_output=True, text=True)
        return out.returncode == 0 and bool(out.stdout.strip())
    except FileNotFoundError:
        return False  # pgrep not available; can't tell


def _time_batch(mnova_exe: Path, xml_paths: list[Path], work_dir: Path,
                label: str, workers: int = 1, launcher: str = "open") -> dict:
    """Copy ``xml_paths`` into a fresh dir, run one MNova batch, and time it.

    Returns a dict with the measured subprocess wall-clock (``dt``), the number
    of output ``.txt`` files actually produced (``n_out``) vs expected (``n``),
    and the spread of their modification times (``mtime_span``). If the
    subprocess returns before all outputs exist, that signals MNova handed the
    work off to an already-running instance and the timing is NOT reliable.
    """
    import shutil

    xml_dir = work_dir / f"xmls_{label}"
    txt_dir = work_dir / f"txt_{label}"
    # Clear BOTH dirs so leftovers from a previous run can't inflate the count.
    for d in (xml_dir, txt_dir):
        if d.exists():
            shutil.rmtree(d)
    xml_dir.mkdir(parents=True, exist_ok=True)
    for p in xml_paths:
        shutil.copy2(p, xml_dir / p.name)

    n = len(xml_paths)
    desc = "one launch" if workers <= 1 else f"{workers} workers ({launcher})"
    print(f"\n--- timing '{label}': {n} sims, {desc} ---")
    t0 = time.perf_counter()
    if workers <= 1:
        run_mnova_batch(mnova_exe, xml_dir, txt_dir)
    else:
        run_mnova_parallel(mnova_exe, xml_dir, txt_dir,
                           workers=workers, launcher=launcher)
    dt = time.perf_counter() - t0

    txts = sorted(txt_dir.glob("*.txt"))
    n_out = len(txts)
    mtimes = [p.stat().st_mtime for p in txts]
    mtime_span = (max(mtimes) - min(mtimes)) if mtimes else 0.0

    status = "OK" if n_out == n else f"INCOMPLETE ({n_out}/{n})"
    print(f"--- '{label}': subprocess {dt:.2f} s | outputs {status} | "
          f"output mtime span {mtime_span:.2f} s ---")
    if n_out < n:
        print(f"    *** WARNING: only {n_out}/{n} outputs existed when MNova "
              "returned. Timing is unreliable — MNova likely handed off to a "
              "running instance. Quit ALL MestReNova windows and re-run. ***")
    return {"dt": dt, "n": n, "n_out": n_out, "mtime_span": mtime_span}


def run_benchmark(
    source_xml: Path,
    out_dir: Path,
    mnova_exe: Path = MNOVA_DEFAULT,
    n: int = 100,
    fmin: float = 40.0,
    fmax: float = 1200.0,
    calib_n: int = 1,
    workers: int = 1,
    launcher: str = "open",
    baseline: bool = False,
) -> dict:
    """Patch one XML to ``n`` geometric frequencies and time the simulation.

    With ``workers > 1`` the full run is sharded across that many MNova
    instances; the calibration run is always single-worker (it estimates the
    one-launch startup cost). Returns a dict with the timing breakdown.
    """
    source_xml = Path(source_xml)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _mnova_already_running():
        print("*** WARNING: a MestReNova process is already running. MNova is "
              "single-instance, so the launched process may hand off to it and "
              "return early, making timings meaningless. Quit all MNova windows "
              "first for an accurate benchmark. ***")

    freqs = geometric_frequencies(fmin, fmax, n)
    print(f"Frequencies: {n} points, geometric in [{fmin}, {fmax}] MHz")
    if n >= 3:
        print(f"  first gaps: {freqs[1]-freqs[0]:.2f}, {freqs[2]-freqs[1]:.2f} MHz; "
              f"last gap: {freqs[-1]-freqs[-2]:.2f} MHz")
    else:
        print(f"  values: {', '.join(f'{f:.1f}' for f in freqs)} MHz")

    # Patch the source XML to every frequency.
    xml_dir = out_dir / "freq_xmls"
    xml_dir.mkdir(parents=True, exist_ok=True)
    all_xmls: list[Path] = []
    for i, f in enumerate(freqs):
        p = xml_dir / f"{source_xml.stem}_{i:03d}_{f:.1f}MHz.xml"
        save_xml(patch_frequency(source_xml, f), p)
        all_xmls.append(p)

    # Calibration is always single-worker (estimates one-launch startup cost);
    # the full run uses the requested worker count.
    calib = _time_batch(mnova_exe, all_xmls[:calib_n], out_dir, "calib")
    full = _time_batch(mnova_exe, all_xmls, out_dir, "full",
                       workers=workers, launcher=launcher)
    t_calib, t_full = calib["dt"], full["dt"]

    complete = (calib["n_out"] == calib["n"]) and (full["n_out"] == full["n"])
    throughput = (n / t_full) if t_full > 0 else None

    report = {
        "n": n,
        "fmin": fmin,
        "fmax": fmax,
        "calib_n": calib_n,
        "workers": workers,
        "launcher": launcher if workers > 1 else None,
        "outputs_complete": complete,
        "full_outputs": f"{full['n_out']}/{full['n']}",
        "full_output_mtime_span_s": round(full["mtime_span"], 3),
        "t_calib_s": round(t_calib, 3),
        "t_full_s": round(t_full, 3),
        "wallclock_per_sim_s": round(t_full / n, 4),
        "throughput_sims_per_s": round(throughput, 2) if throughput else None,
    }

    # The startup/marginal linear model is valid only for a single sequential
    # launch (calib and full share one per-launch model). Speedup must be
    # measured against a SAME-n single-worker baseline, not the calibration run
    # (the calib run is startup-dominated, which would inflate the ratio).
    if workers <= 1:
        per_sim = (t_full - t_calib) / (n - calib_n) if n > calib_n else float("nan")
        startup = max(0.0, t_calib - calib_n * per_sim)
        report["startup_overhead_s"] = round(startup, 3)
        report["marginal_per_sim_s"] = round(per_sim, 4)
        # With few sims (or sub-second sims), startup jitter swamps the marginal
        # estimate — fall back to the mtime span as the honest compute measure.
        if n - calib_n < 10 or per_sim <= 0:
            report["marginal_unreliable"] = True
            span_per_sim = (full["mtime_span"] / max(1, n - 1)) if full["mtime_span"] else None
            report["mtime_per_sim_s"] = round(span_per_sim, 4) if span_per_sim else None
    elif baseline:
        base = _time_batch(mnova_exe, all_xmls, out_dir, "baseline", workers=1)
        t_base = base["dt"]
        report["t_baseline_1worker_s"] = round(t_base, 3)
        report["speedup_vs_1worker_x"] = round(t_base / t_full, 2) if t_full > 0 else None

    print("\n================ BENCHMARK REPORT ================")
    print(f"  simulations (n)            : {report['n']}")
    print(f"  workers                    : {report['workers']}"
          + (f" ({launcher})" if workers > 1 else " (sequential)"))
    print(f"  outputs complete           : {report['outputs_complete']} "
          f"({report['full_outputs']})")
    print(f"  full run wall-clock        : {report['t_full_s']} s")
    print(f"  output mtime span (xcheck) : {report['full_output_mtime_span_s']} s")
    print(f"  wall-clock per-sim (total/n): {report['wallclock_per_sim_s']} s")
    print(f"  throughput                 : {report['throughput_sims_per_s']} sims/s")
    if workers <= 1:
        print(f"  estimated startup overhead : {report['startup_overhead_s']} s")
        print(f"  estimated per-sim (marginal): {report['marginal_per_sim_s']} s")
        if report.get("marginal_unreliable"):
            print("  >>> marginal unreliable at this n (startup jitter dominates).")
            print(f"  >>> use mtime-based per-sim: {report.get('mtime_per_sim_s')} s")
    elif baseline:
        print(f"  1-worker baseline wall-clock: {report['t_baseline_1worker_s']} s")
        print(f"  REAL speedup vs 1 worker   : {report['speedup_vs_1worker_x']}x")
    else:
        print("  (run with --baseline to measure true speedup vs 1 worker;")
        print("   comparing to the 1-sim calib run would be misleading)")
    if not report["outputs_complete"]:
        print("  >>> TIMING UNRELIABLE: outputs incomplete when MNova returned.")
        print("  >>> If launcher='open' gave 0 outputs, try --launcher direct.")
    print("==================================================")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark MNova over geometric field grid")
    p.add_argument("--source_xml", type=Path, required=True,
                   help="Single source mnova-spinsim XML to patch")
    p.add_argument("--out_dir", type=Path, default=Path("/tmp/spinhance_bench"),
                   help="Working/output directory")
    p.add_argument("--mnova", type=Path, default=MNOVA_DEFAULT,
                   help="Path to the MestReNova executable")
    p.add_argument("--n", type=int, default=100, help="Number of frequencies")
    p.add_argument("--fmin", type=float, default=40.0, help="Min frequency (MHz)")
    p.add_argument("--fmax", type=float, default=1200.0, help="Max frequency (MHz)")
    p.add_argument("--calib_n", type=int, default=1,
                   help="Calibration run size for startup-overhead estimate")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent MNova instances for the full run (default 1)")
    p.add_argument("--launcher", choices=["open", "direct"], default="open",
                   help="Parallel launch method on macOS (default: open)")
    p.add_argument("--baseline", action="store_true",
                   help="Also run a 1-worker full pass to measure TRUE speedup")
    args = p.parse_args(argv)

    if not args.mnova.exists():
        print(f"ERROR: MNova not found at {args.mnova}", file=sys.stderr)
        return 2

    # Resolve the source XML relative to cwd, then fall back to the repo root
    # (where the example XML lives) so the benchmark runs from any directory.
    source = args.source_xml
    if not source.exists():
        candidate = _REPO_ROOT / source
        if candidate.exists():
            source = candidate
        else:
            print(f"ERROR: source XML not found: {args.source_xml}\n"
                  f"  (also tried {candidate})", file=sys.stderr)
            return 2

    run_benchmark(
        source_xml=source, out_dir=args.out_dir, mnova_exe=args.mnova,
        n=args.n, fmin=args.fmin, fmax=args.fmax, calib_n=args.calib_n,
        workers=args.workers, launcher=args.launcher, baseline=args.baseline,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
