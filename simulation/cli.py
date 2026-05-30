"""
cli.py
======
Command-line entry point for the SpinHance simulation package.

Run as a module from the repository root::

    python -m simulation.cli run  --xml_dir SRC --out_dir OUT [--mnova PATH] [--fields 90 600]
    python -m simulation.cli plot --spectra_dir DIR [--stem NAME] [--show]

Subcommands
-----------
``run``   End-to-end pipeline: patch XMLs → MNova simulation → normalised ``.npy``.
``plot``  QC overlay of one molecule's spectra across fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python simulation/cli.py ...` (direct path) as well as `-m simulation.cli`
# by ensuring the repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, "") and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from simulation.mnova_runner import MNOVA_DEFAULT
from simulation.pipeline import DEFAULT_FIELDS_MHZ, run_pipeline
from simulation.plotting import plot_field_comparison

_REPO_DATA = _REPO_ROOT / "data" / "processed"


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="Run the full simulation pipeline")
    p.add_argument("--xml_dir", type=Path, default=_REPO_DATA / "xmls_source",
                   help="Directory of source mnova-spinsim XML files")
    p.add_argument("--out_dir", type=Path, default=_REPO_DATA,
                   help="Root output directory")
    p.add_argument("--mnova", type=Path, default=MNOVA_DEFAULT,
                   help="Path to the MestReNova executable")
    p.add_argument("--fields", type=float, nargs="+", default=list(DEFAULT_FIELDS_MHZ),
                   help="Spectrometer frequencies in MHz (default: 90.0 600.15)")


def _add_plot(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("plot", help="Overlay 90 vs 600 MHz spectra for one molecule")
    p.add_argument("--spectra_dir", type=Path, required=True,
                   help="Dir containing <field>MHz/<stem>.npy subfolders")
    p.add_argument("--stem", type=str, default=None,
                   help="Molecule stem (default: first .npy found)")
    p.add_argument("--fields", type=float, nargs="+", default=[90.0, 600.0],
                   help="Fields to overlay (default: 90 600)")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path")
    p.add_argument("--show", action="store_true", help="Show interactively")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="simulation",
        description="SpinHance Task 3 — spin simulation pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run(sub)
    _add_plot(sub)
    args = parser.parse_args(argv)

    if args.command == "run":
        if not args.mnova.exists():
            print(f"ERROR: MNova executable not found at {args.mnova}\n"
                  "Pass --mnova /path/to/MestReNova", file=sys.stderr)
            return 2
        run_pipeline(
            source_xml_dir=args.xml_dir,
            out_dir=args.out_dir,
            mnova_exe=args.mnova,
            fields_mhz=args.fields,
        )
        return 0

    if args.command == "plot":
        plot_field_comparison(
            spectra_dir=args.spectra_dir,
            stem=args.stem,
            fields_mhz=args.fields,
            out=args.out,
            show=args.show,
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
