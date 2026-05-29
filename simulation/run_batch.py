"""
run_batch.py
------------
Python orchestrator for the SpinHance simulation pipeline.

Pipeline
--------
1. Take a directory of source XMLs (one per molecule, any field).
2. For each molecule generate two field-patched XMLs: 100 MHz and 600 MHz.
3. Invoke MNova headlessly on each XML directory via batch_simulate.qs.
4. Load the exported .txt intensity arrays, normalise, and save as .npy.

Usage
-----
    micromamba activate spinhance
    python simulation/run_batch.py \\
        --xml_dir   data/processed/xmls_source \\
        --out_dir   data/processed/spectra \\
        --mnova     /Applications/MestReNova.app/Contents/MacOS/MestReNova \\
        --fields    100 600

Or import and call `run_pipeline()` programmatically.
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Repo root (two levels up from this file)
REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPT_PATH = Path(__file__).resolve().parent / "batch_simulate.qs"

# Default MNova executable location on macOS
MNOVA_DEFAULT = Path("/Applications/MestReNova.app/Contents/MacOS/MestReNova")

# Spectral grid constants (must match XML parameters)
PPM_FROM  = 0.0
PPM_TO    = 12.0
N_POINTS  = 16384  # 2^14


# ── XML generation ────────────────────────────────────────────────────────────

def prepare_xmls(
    source_xml_dir: Path,
    patched_xml_dir: Path,
    fields_mhz: list[float],
) -> dict[float, Path]:
    """
    For each source XML in source_xml_dir, produce one copy per field in
    patched_xml_dir/<field>MHz/.

    Returns {field_mhz: output_subdir}.
    """
    from simulation.xml_utils import generate_field_pair, patch_frequency, save_xml
    import xml.etree.ElementTree as ET

    field_dirs: dict[float, Path] = {}
    source_xmls = sorted(source_xml_dir.glob("*.xml"))
    if not source_xmls:
        raise FileNotFoundError(f"No XML files found in {source_xml_dir}")

    for field in fields_mhz:
        fdir = patched_xml_dir / f"{field:.0f}MHz"
        fdir.mkdir(parents=True, exist_ok=True)
        field_dirs[field] = fdir

    for xml_path in source_xmls:
        stem = xml_path.stem
        for field in fields_mhz:
            out = field_dirs[field] / f"{stem}.xml"
            patched = patch_frequency(xml_path, field)
            save_xml(patched, out)

    n = len(source_xmls)
    print(f"Prepared {n} XMLs × {len(fields_mhz)} fields → {patched_xml_dir}")
    return field_dirs


# ── MNova script installation ─────────────────────────────────────────────────

MNOVA_SCRIPTS_DIR = Path.home() / "Library" / "Application Support" / \
                    "Mestrelab Research S.L." / "MestReNova" / "scripts"
QS_SCRIPT = Path(__file__).parent / "spinhanceBatch.qs"


def install_qs_script(force: bool = False) -> Path:
    """
    Copy batch_simulate.qs into MNova's user scripts directory so MNova
    auto-loads it on startup and the function spinhanceBatch() is available.
    """
    MNOVA_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MNOVA_SCRIPTS_DIR / QS_SCRIPT.name
    if not dest.exists() or force:
        import shutil
        shutil.copy2(QS_SCRIPT, dest)
        print(f"  Installed {QS_SCRIPT.name} → {dest}")
    else:
        print(f"  Script already installed at {dest}")
    return dest


# ── MNova invocation ──────────────────────────────────────────────────────────

CONFIG_PATH = Path.home() / ".spinhance_batch_config.json"


def run_mnova_batch(
    mnova_exe: Path,
    xml_dir: Path,
    txt_out_dir: Path,
    timeout_per_file: float = 30.0,
) -> None:
    """
    Call MNova 16 headlessly on all XMLs in xml_dir.
    Outputs one .txt per XML into txt_out_dir.

    Strategy (MNova 16):
        --nogui          headless mode
        --sf "spinhanceBatch()"   call our auto-loaded JS function

    Paths are passed via a JSON config file at /tmp/spinhance_batch_config.json
    which the .qs script reads on startup.
    """
    import json

    txt_out_dir.mkdir(parents=True, exist_ok=True)

    n_xml = len(list(xml_dir.glob("*.xml")))
    total_timeout = max(120, n_xml * timeout_per_file)

    # Write config for the JS script
    config = {
        "xml_dir": str(xml_dir.resolve()),
        "out_dir": str(txt_out_dir.resolve()),
    }
    CONFIG_PATH.write_text(json.dumps(config))
    print(f"  Config written to {CONFIG_PATH}")

    # Ensure script is installed in MNova's scripts dir
    install_qs_script()

    cmd = [
        str(mnova_exe),
        "--nogui",
        "--sf", str(QS_SCRIPT.resolve()),
    ]
    print(f"  Running MNova on {n_xml} files in {xml_dir.name} ...")
    print(f"  CMD: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=total_timeout,
    )

    if result.returncode != 0:
        print("  MNova stderr:", result.stderr[:2000])
        raise RuntimeError(f"MNova exited with code {result.returncode}")

    print(result.stdout.strip())


# ── Post-processing: txt → npy ────────────────────────────────────────────────

def txt_to_npy(
    txt_dir: Path,
    npy_dir: Path,
    n_points: int = N_POINTS,
    ppm_from: float = PPM_FROM,
    ppm_to: float   = PPM_TO,
) -> None:
    """
    Load exported .txt intensity files, normalise (integral = 1), save as .npy.
    Also saves the shared ppm axis once as ppm_axis.npy.
    """
    npy_dir.mkdir(parents=True, exist_ok=True)

    ppm_axis = np.linspace(ppm_from, ppm_to, n_points)
    np.save(npy_dir / "ppm_axis.npy", ppm_axis)

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print(f"  WARNING: no .txt files found in {txt_dir}")
        return

    for txt_path in txt_files:
        try:
            intensities = np.loadtxt(txt_path, dtype=np.float32)
        except Exception as e:
            print(f"  WARN: could not load {txt_path.name}: {e}")
            continue

        if len(intensities) != n_points:
            print(
                f"  WARN: {txt_path.name} has {len(intensities)} points, "
                f"expected {n_points}. Skipping."
            )
            continue

        # Normalise so the integral (sum × Δppm per point) = 1
        dppm = (ppm_to - ppm_from) / n_points
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
    mnova_exe: Path,
    fields_mhz: list[float] = (100.0, 600.15),
) -> None:
    """
    End-to-end: source XMLs → patched XMLs → MNova simulation → .npy arrays.

    Directory layout created under out_dir:
        xmls/<field>MHz/       patched XML files
        txt/<field>MHz/        raw MNova exports
        spectra/<field>MHz/    normalised .npy arrays + ppm_axis.npy
    """
    patched_xml_dir = out_dir / "xmls"
    txt_base        = out_dir / "txt"
    npy_base        = out_dir / "spectra"

    print("=== Step 1: Patch XMLs for each field ===")
    field_dirs = prepare_xmls(source_xml_dir, patched_xml_dir, list(fields_mhz))

    for field, xml_dir in field_dirs.items():
        label = f"{field:.0f}MHz"
        print(f"\n=== Step 2: MNova simulation @ {label} ===")
        txt_dir = txt_base / label
        run_mnova_batch(mnova_exe, xml_dir, txt_dir)

        print(f"\n=== Step 3: Convert txt → npy @ {label} ===")
        npy_dir = npy_base / label
        txt_to_npy(txt_dir, npy_dir)

    print("\n=== Pipeline complete ===")
    print(f"Spectra saved to: {npy_base}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpinHance simulation pipeline: XML → MNova → .npy spectra"
    )
    p.add_argument(
        "--xml_dir", type=Path,
        default=REPO_ROOT / "data" / "processed" / "xmls_source",
        help="Directory of source mnova-spinsim XML files",
    )
    p.add_argument(
        "--out_dir", type=Path,
        default=REPO_ROOT / "data" / "processed",
        help="Root output directory",
    )
    p.add_argument(
        "--mnova", type=Path,
        default=MNOVA_DEFAULT,
        help="Path to MestReNova executable",
    )
    p.add_argument(
        "--fields", type=float, nargs="+",
        default=[100.0, 600.15],
        help="Spectrometer frequencies in MHz (default: 100.0 600.15)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.mnova.exists():
        sys.exit(
            f"ERROR: MNova executable not found at {args.mnova}\n"
            "Pass --mnova /path/to/MestReNova"
        )

    run_pipeline(
        source_xml_dir=args.xml_dir,
        out_dir=args.out_dir,
        mnova_exe=args.mnova,
        fields_mhz=args.fields,
    )
