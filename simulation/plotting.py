"""
plotting.py
===========
Quick QC plotting for simulated spectra. Overlays the low-field (90 MHz) and
high-field (600 MHz) ``.npy`` spectra for one molecule so you can confirm the
low-field spectrum is more strongly coupled (more / overlapping lines).

Use :func:`plot_field_comparison` programmatically, or the ``plot`` subcommand
in :mod:`simulation.cli`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pipeline import N_POINTS, PPM_FROM, PPM_TO

__all__ = ["load_ppm_axis", "resolve_stem", "plot_field_comparison"]


def load_ppm_axis(spectra_dir: Path, n_points: int) -> np.ndarray:
    """Return a saved ``ppm_axis.npy`` of matching length, else reconstruct it."""
    for axis_path in spectra_dir.rglob("ppm_axis.npy"):
        ax = np.load(axis_path)
        if len(ax) == n_points:
            return ax
    return np.linspace(PPM_FROM, PPM_TO, n_points)


def resolve_stem(spectra_dir: Path, stem: str | None) -> str:
    """Return ``stem`` if given, else the first molecule ``.npy`` found."""
    if stem is not None:
        return stem
    cands = sorted(p for p in spectra_dir.rglob("*.npy") if p.stem != "ppm_axis")
    if not cands:
        raise FileNotFoundError(f"No spectra .npy under {spectra_dir}")
    return cands[0].stem


def plot_field_comparison(
    spectra_dir: Path,
    stem: str | None = None,
    fields_mhz: list[float] = (90.0, 600.0),
    out: Path | None = None,
    show: bool = False,
) -> Path:
    """Stack one molecule's spectra across fields and save a PNG.

    Parameters
    ----------
    spectra_dir
        Directory containing ``<field>MHz/<stem>.npy`` subfolders.
    stem
        Molecule stem; defaults to the first molecule found.
    fields_mhz
        Fields to plot, one panel each (x-axis reversed, NMR convention).
    out
        Output PNG path; defaults to ``<spectra_dir>/<stem>_compare.png``.
    show
        If True, also display the figure interactively.

    Returns
    -------
    Path
        The path the PNG was written to.
    """
    # Imported lazily so importing this module never requires a display backend.
    import matplotlib.pyplot as plt

    spectra_dir = Path(spectra_dir)
    stem = resolve_stem(spectra_dir, stem)

    fig, axes = plt.subplots(len(fields_mhz), 1, figsize=(11, 6), sharex=True)
    if len(fields_mhz) == 1:
        axes = [axes]

    for ax, field in zip(axes, fields_mhz):
        npy = spectra_dir / f"{field:.0f}MHz" / f"{stem}.npy"
        if not npy.exists():
            ax.set_title(f"{field:.0f} MHz — MISSING ({npy})")
            continue
        y = np.load(npy)
        ppm = load_ppm_axis(spectra_dir, len(y))
        ax.plot(ppm, y, lw=0.6)
        ax.set_title(f"{stem} @ {field:.0f} MHz")
        ax.set_ylabel("intensity")
        ax.invert_xaxis()  # high ppm on the left

    axes[-1].set_xlabel("¹H chemical shift (ppm)")
    fig.tight_layout()

    out = Path(out) if out else (spectra_dir / f"{stem}_compare.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    if show:
        plt.show()
    return out
