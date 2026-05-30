"""
validate_vs_mnova.py
====================
Accuracy check: simulate the SAME spin system with MestReNova and with pyspin,
then compare. Both engines read identical parameters (parsed from one XML via
xml_io.xml_to_matrix), so any difference is the simulators, not the inputs.

Run locally (needs MNova):

    python -m simulation.pyspin.validate_vs_mnova \
        --xml simulation/examples/R_5_methylcyclohexenone.xml \
        --mnova "/Applications/MestReNova.app/Contents/MacOS/MestReNova" \
        --field 90 --show

Reports Pearson correlation, RMSE, and peak-position agreement, and saves an
overlay PNG. Handles a possible ppm-axis orientation flip between engines.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulation.mnova_runner import MNOVA_DEFAULT, run_mnova_batch
from simulation.pyspin.simulator import simulate_spectrum
from simulation.xml_io import patch_frequency, save_xml, xml_to_matrix


def _normalize(y: np.ndarray, ppm_from: float, ppm_to: float) -> np.ndarray:
    dppm = (ppm_to - ppm_from) / len(y)
    total = y.sum() * dppm
    return y / total if total > 0 else y


def _peaks(ppm: np.ndarray, y: np.ndarray, frac: float = 0.05) -> np.ndarray:
    thr = y.max() * frac
    m = (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > thr)
    return ppm[1:-1][m]


def _peak_agreement(p1: np.ndarray, p2: np.ndarray) -> float:
    """Mean nearest-neighbour distance (ppm) between two peak lists."""
    if len(p1) == 0 or len(p2) == 0:
        return float("nan")
    return float(np.mean([min(abs(p - q) for q in p2) for p in p1]))


def run_mnova_spectrum(xml_path: Path, mnova_exe: Path, field: float,
                       points: int, ppm_from: float, ppm_to: float) -> np.ndarray:
    """Patch XML to `field`, run MNova, return a normalized intensity array."""
    tmp = Path(tempfile.mkdtemp(prefix="pyspin_val_"))
    try:
        xdir = tmp / "xml"; odir = tmp / "txt"
        xdir.mkdir(); odir.mkdir()
        save_xml(patch_frequency(xml_path, field), xdir / "mol.xml")
        run_mnova_batch(mnova_exe, xdir, odir)
        txts = list(odir.glob("*.txt"))
        if not txts:
            raise RuntimeError("MNova produced no output (is the scripts folder registered?)")
        y = np.loadtxt(txts[0], dtype=float)
        return _normalize(y, ppm_from, ppm_to)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compare(xml_path: Path, mnova_exe: Path, field: float | None = None,
            show: bool = False, out: Path | None = None) -> dict:
    meta = xml_to_matrix(xml_path)
    field = field or meta["frequency_mhz"] or 90.0
    points = meta["points"] or 16384
    ppm_from = meta["ppm_from"] if meta["ppm_from"] is not None else 0.0
    ppm_to = meta["ppm_to"] if meta["ppm_to"] is not None else 12.0

    print(f"Comparing at {field} MHz ({points} pts, {ppm_from}-{ppm_to} ppm)")

    # pyspin
    ppm, py = simulate_spectrum(meta["shifts"], meta["couplings"], meta["degeneracy"],
                                field, points=points, ppm_from=ppm_from, ppm_to=ppm_to)
    py = _normalize(py, ppm_from, ppm_to)

    # MNova
    mn = run_mnova_spectrum(xml_path, mnova_exe, field, points, ppm_from, ppm_to)
    if len(mn) != len(py):
        # resample MNova onto pyspin grid if point counts differ
        mn = np.interp(np.linspace(0, 1, len(py)), np.linspace(0, 1, len(mn)), mn)
        mn = _normalize(mn, ppm_from, ppm_to)

    # Resolve possible axis orientation flip: pick the one with higher correlation.
    r_fwd = float(np.corrcoef(py, mn)[0, 1])
    r_rev = float(np.corrcoef(py, mn[::-1])[0, 1])
    flipped = r_rev > r_fwd
    if flipped:
        mn = mn[::-1]
    r = max(r_fwd, r_rev)

    rmse = float(np.sqrt(np.mean((py - mn) ** 2)))
    pk_py, pk_mn = _peaks(ppm, py), _peaks(ppm, mn)
    pk_err = _peak_agreement(pk_py, pk_mn)

    print("\n============ pyspin vs MNova ============")
    print(f"  Pearson correlation : {r:.4f}" + ("  (MNova axis flipped)" if flipped else ""))
    print(f"  RMSE (norm. intens) : {rmse:.4g}")
    print(f"  peaks  py / mnova   : {len(pk_py)} / {len(pk_mn)}")
    print(f"  mean peak offset    : {pk_err:.4f} ppm")
    print("=========================================")

    out = Path(out) if out else xml_path.with_suffix(f".cmp_{field:.0f}MHz.png")
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax[0].plot(ppm, py, lw=0.6); ax[0].set_title(f"pyspin @ {field:.0f} MHz"); ax[0].invert_xaxis()
        ax[1].plot(ppm, mn, lw=0.6, color="C1"); ax[1].set_title("MNova"); ax[1].invert_xaxis()
        ax[1].set_xlabel("¹H shift (ppm)")
        fig.tight_layout(); fig.savefig(out, dpi=150)
        print(f"  overlay saved: {out}")
        if show:
            plt.show()
    except Exception as e:  # noqa: BLE001
        print(f"  (plot skipped: {e})")

    return {"field_mhz": field, "correlation": r, "rmse": rmse,
            "n_peaks_pyspin": len(pk_py), "n_peaks_mnova": len(pk_mn),
            "mean_peak_offset_ppm": pk_err, "mnova_axis_flipped": flipped}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate pyspin against MestReNova")
    p.add_argument("--xml", type=Path, required=True, help="Source spin-system XML")
    p.add_argument("--mnova", type=Path, default=MNOVA_DEFAULT)
    p.add_argument("--field", type=float, default=None, help="MHz (default: XML's)")
    p.add_argument("--show", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if not args.mnova.exists():
        print(f"ERROR: MNova not found at {args.mnova}", file=sys.stderr)
        return 2
    if not args.xml.exists():
        cand = _REPO_ROOT / args.xml
        if cand.exists():
            args.xml = cand
        else:
            print(f"ERROR: XML not found: {args.xml}", file=sys.stderr)
            return 2
    compare(args.xml, args.mnova, field=args.field, show=args.show, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
