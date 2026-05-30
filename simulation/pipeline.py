"""
pipeline.py
===========
End-to-end orchestration for SpinHance Task 3 (simulation). Ties together
:mod:`simulation.xml_io` (XML generation), :mod:`simulation.mnova_runner`
(MNova invocation), and NumPy post-processing.

Stages
------
1. **Patch**   — for each source XML, emit one frequency-patched copy per field.
2. **Simulate**— run MNova on each field's XML directory → one ``.txt`` per file.
3. **Convert** — load each ``.txt``, normalise to unit integral, save ``.npy``.

Output layout (under ``out_dir``)
---------------------------------
::

    xmls/<field>MHz/       patched XML files
    txt/<field>MHz/        raw MNova intensity exports
    spectra/<field>MHz/    normalised .npy arrays + ppm_axis.npy

Use :func:`run_pipeline` programmatically, or the ``run`` subcommand in
:mod:`simulation.cli`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .mnova_runner import MNOVA_DEFAULT, run_mnova_parallel
from .xml_io import (
    HIGH_FIELD_MHZ,
    LOW_FIELD_MHZ,
    generate_field_pair,  # noqa: F401  (re-exported for convenience)
    patch_frequency,
    save_xml,
)

__all__ = [
    "PPM_FROM",
    "PPM_TO",
    "N_POINTS",
    "DEFAULT_FIELDS_MHZ",
    "prepare_xmls",
    "txt_to_npy",
    "run_pipeline",
]

# Spectral grid constants — must match the XML parameters in xml_io.
PPM_FROM = 0.0
PPM_TO = 12.0
N_POINTS = 16384  # 2**14

DEFAULT_FIELDS_MHZ: tuple[float, float] = (LOW_FIELD_MHZ, HIGH_FIELD_MHZ)


# ── Stage 1: patch ────────────────────────────────────────────────────────────

def prepare_xmls(
    source_xml_dir: Path,
    patched_xml_dir: Path,
    fields_mhz: list[float],
) -> dict[float, Path]:
    """Emit one frequency-patched copy of each source XML per field.

    Parameters
    ----------
    source_xml_dir
        Directory of source ``mnova-spinsim`` XML files (any field).
    patched_xml_dir
        Root under which ``<field>MHz/`` subdirectories are created.
    fields_mhz
        Spectrometer frequencies to generate.

    Returns
    -------
    dict[float, Path]
        Mapping ``{field_mhz: output_subdir}``.

    Raises
    ------
    FileNotFoundError
        If ``source_xml_dir`` contains no ``*.xml`` files.
    """
    source_xmls = sorted(source_xml_dir.glob("*.xml"))
    if not source_xmls:
        raise FileNotFoundError(f"No XML files found in {source_xml_dir}")

    field_dirs: dict[float, Path] = {}
    for field in fields_mhz:
        fdir = patched_xml_dir / f"{field:.0f}MHz"
        fdir.mkdir(parents=True, exist_ok=True)
        field_dirs[field] = fdir

    for xml_path in source_xmls:
        stem = xml_path.stem
        for field in fields_mhz:
            save_xml(patch_frequency(xml_path, field),
                     field_dirs[field] / f"{stem}.xml")

    print(f"Prepared {len(source_xmls)} XMLs × {len(fields_mhz)} fields "
          f"→ {patched_xml_dir}")
    return field_dirs


# ── Stage 3: convert ──────────────────────────────────────────────────────────

def txt_to_npy(
    txt_dir: Path,
    npy_dir: Path,
    n_points: int = N_POINTS,
    ppm_from: float = PPM_FROM,
    ppm_to: float = PPM_TO,
) -> None:
    """Convert MNova ``.txt`` exports to normalised ``.npy`` arrays.

    Each spectrum is normalised so its integral (``sum × Δppm``) equals 1.
    A shared ``ppm_axis.npy`` is written once. Files whose length differs from
    ``n_points`` are skipped with a warning.
    """
    npy_dir.mkdir(parents=True, exist_ok=True)

    ppm_axis = np.linspace(ppm_from, ppm_to, n_points)
    np.save(npy_dir / "ppm_axis.npy", ppm_axis)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print(f"  WARNING: no .txt files found in {txt_dir}")
        return

    dppm = (ppm_to - ppm_from) / n_points
    for txt_path in txt_files:
        try:
            intensities = np.loadtxt(txt_path, dtype=np.float32)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"  WARN: could not load {txt_path.name}: {e}")
            continue

        if len(intensities) != n_points:
            print(f"  WARN: {txt_path.name} has {len(intensities)} points, "
                  f"expected {n_points}. Skipping.")
            continue

        total = intensities.sum() * dppm
        if total > 0:
            intensities /= total

        np.save(npy_dir / (txt_path.stem + ".npy"), intensities)

    n = len(list(npy_dir.glob("*.npy"))) - 1  # exclude ppm_axis
    print(f"  Saved {n} spectra to {npy_dir}")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    source_xml_dir: Path,
    out_dir: Path,
    mnova_exe: Path = MNOVA_DEFAULT,
    fields_mhz: list[float] = DEFAULT_FIELDS_MHZ,
    workers: int = 1,
    launcher: str = "open",
) -> None:
    """Run patch → simulate → convert for every field.

    Parameters
    ----------
    workers
        Number of concurrent MNova instances (``1`` = sequential single launch).
    launcher
        Parallel launch method, ``"open"`` or ``"direct"`` (see mnova_runner).

    See module docstring for the output directory layout.
    """
    patched_xml_dir = out_dir / "xmls"
    txt_base = out_dir / "txt"
    npy_base = out_dir / "spectra"

    print("=== Step 1: Patch XMLs for each field ===")
    field_dirs = prepare_xmls(source_xml_dir, patched_xml_dir, list(fields_mhz))

    for field, xml_dir in field_dirs.items():
        label = f"{field:.0f}MHz"
        print(f"\n=== Step 2: MNova simulation @ {label} ===")
        txt_dir = txt_base / label
        run_mnova_parallel(mnova_exe, xml_dir, txt_dir,
                           workers=workers, launcher=launcher)

        print(f"\n=== Step 3: Convert txt → npy @ {label} ===")
        txt_to_npy(txt_dir, npy_base / label)

    print("\n=== Pipeline complete ===")
    print(f"Spectra saved to: {npy_base}")
