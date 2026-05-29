"""
plot_spectra.py
---------------
Quick QC plot for SpinHance simulated spectra. Overlays the low-field (90 MHz)
and high-field (600 MHz) spectra for one molecule so you can eyeball that the
low-field spectrum is more strongly coupled (more/overlapping lines).

Usage
-----
    micromamba activate spinhance
    python simulation/plot_spectra.py \
        --spectra_dir /tmp/sh_test/spectra/spectra \
        --stem mol_test                      # omit to plot the first molecule found

Reads <spectra_dir>/<field>MHz/<stem>.npy and the shared ppm_axis.npy.
Saves a PNG next to the spectra (or shows it with --show).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PPM_FROM, PPM_TO = 0.0, 12.0


def load_axis(spectra_dir: Path, n_points: int) -> np.ndarray:
    """Use a saved ppm_axis.npy if present, else reconstruct it."""
    for axis_path in spectra_dir.rglob("ppm_axis.npy"):
        ax = np.load(axis_path)
        if len(ax) == n_points:
            return ax
    return np.linspace(PPM_FROM, PPM_TO, n_points)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay 90 vs 600 MHz simulated spectra")
    ap.add_argument("--spectra_dir", type=Path, required=True,
                    help="Dir containing <field>MHz/<stem>.npy subfolders")
    ap.add_argument("--stem", type=str, default=None,
                    help="Molecule stem (default: first .npy found)")
    ap.add_argument("--fields", type=float, nargs="+", default=[90.0, 600.0],
                    help="Fields to overlay (default: 90 600)")
    ap.add_argument("--show", action="store_true", help="Show interactively")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG path")
    args = ap.parse_args()

    # Resolve stem if not given
    stem = args.stem
    if stem is None:
        cands = sorted(p for p in args.spectra_dir.rglob("*.npy")
                       if p.stem != "ppm_axis")
        if not cands:
            raise FileNotFoundError(f"No spectra .npy under {args.spectra_dir}")
        stem = cands[0].stem
        print(f"No --stem given; using '{stem}'")

    fig, axes = plt.subplots(len(args.fields), 1, figsize=(11, 6), sharex=True)
    if len(args.fields) == 1:
        axes = [axes]

    for ax, field in zip(axes, args.fields):
        npy = args.spectra_dir / f"{field:.0f}MHz" / f"{stem}.npy"
        if not npy.exists():
            ax.set_title(f"{field:.0f} MHz — MISSING ({npy})")
            continue
        y = np.load(npy)
        ppm = load_axis(args.spectra_dir, len(y))
        ax.plot(ppm, y, lw=0.6)
        ax.set_title(f"{stem} @ {field:.0f} MHz")
        ax.set_ylabel("intensity")
        ax.invert_xaxis()  # NMR convention: high ppm on the left

    axes[-1].set_xlabel("¹H chemical shift (ppm)")
    fig.tight_layout()

    out = args.out or (args.spectra_dir / f"{stem}_compare.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
