"""generate/cli.py — unified command-line interface for the generate pipeline.

Subcommands
-----------

``run``
    Screen the full ChEMBL database end-to-end in a single streaming pass.
    Produces ``candidates_final.csv``.

``view``
    Launch the interactive Tkinter gallery viewer.  Defaults to
    ``candidates_final.csv`` so no arguments are required after screening.

Usage
-----
::

    # All three invocation styles are equivalent:
    python generate/cli.py run
    python -m generate.cli run
    spinhance-gen run               # after pip install -e .

    python generate/cli.py view
    python generate/cli.py view --file generate/data/candidates_final.csv

    # Override paths / parameters:
    python generate/cli.py run --chembl /data/chembl_37_chemreps.txt
    python generate/cli.py run --n-groups 6

Notes
-----
All heavy imports (RDKit, pipeline, viewer) are deferred to inside the
command functions so that ``python generate/cli.py --help`` and startup
are instant, regardless of WSL2 filesystem overhead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Only config.py (pure Python, no RDKit) is imported at module level.
from generate.config import N_SPIN_GROUPS  # noqa: E402

# Path defaults — computed here so the argument parser can show them,
# but without importing RDKit or the pipeline.
_REPO_ROOT        = Path(__file__).resolve().parent.parent
_DEFAULT_CHEMBL   = _REPO_ROOT / "generate" / "chembl" / "chembl_37_chemreps.txt"
_DEFAULT_OUTPUT   = _REPO_ROOT / "generate" / "data" / "candidates_final.csv"
_DEFAULT_WORKERS  = max(1, (os.cpu_count() or 2) - 1)
_DEFAULT_CHUNK    = 32


# ── Command handlers (all heavy imports deferred to here) ────────────────────

def _cmd_run(args: argparse.Namespace) -> int:
    from generate.pipeline import run_pipeline  # noqa: PLC0415 — intentional lazy import
    _, kept = run_pipeline(
        chembl_path   = Path(args.chembl),
        output_path   = Path(args.output),
        target_groups = args.n_groups,
        workers       = args.workers,
        chunk_size    = args.chunk_size,
    )
    return 0 if kept >= 0 else 1


def _cmd_view(args: argparse.Namespace) -> int:
    import os, signal  # noqa: E401

    # RDKit can take 30-120 s to import on WSL2 (Windows Defender scans .so files).
    # Install an os._exit handler so Ctrl+C works immediately during that phase.
    _orig = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda sig, frame: os._exit(130))

    print("Loading viewer (first run may take a moment)...", end=" ", flush=True)
    from generate.viewer import launch  # noqa: PLC0415
    print("ready.", flush=True)

    signal.signal(signal.SIGINT, _orig)   # restore normal handler for the GUI
    launch(Path(args.file), n=args.n, seed=args.seed)
    return 0


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spinhance-gen",
        description=(
            f"SpinHance Task 1 — screen ChEMBL for molecules with exactly "
            f"{N_SPIN_GROUPS} magnetically distinct ¹H spin groups."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Screen ChEMBL end-to-end and write candidates_final.csv.",
        description=(
            "Single-pass pipeline: streams ChEMBL, applies the fast proton-count "
            "heuristic, then the 3-D deuterium substitution test.  Molecules with "
            f"exactly N spin groups are written to {_DEFAULT_OUTPUT.name}."
        ),
    )
    p_run.add_argument(
        "--chembl", default=str(_DEFAULT_CHEMBL), metavar="PATH",
        help=f"ChEMBL chemreps .txt file  (default: {_DEFAULT_CHEMBL.name})",
    )
    p_run.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT), metavar="PATH",
        help=f"Output CSV  (default: {_DEFAULT_OUTPUT.name})",
    )
    p_run.add_argument(
        "--n-groups", type=int, default=N_SPIN_GROUPS, metavar="N",
        help=f"Target spin-group count  (default: {N_SPIN_GROUPS})",
    )
    p_run.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS, metavar="N",
        help=f"Worker processes for the 3-D deuterium test  (default: {_DEFAULT_WORKERS})",
    )
    p_run.add_argument(
        "--chunk-size", type=int, default=_DEFAULT_CHUNK, metavar="N",
        dest="chunk_size",
        help=f"Molecules per worker batch  (default: {_DEFAULT_CHUNK})",
    )
    p_run.set_defaults(func=_cmd_run)

    # ── view ──────────────────────────────────────────────────────────────────
    p_view = sub.add_parser(
        "view",
        help="Launch the interactive gallery viewer.",
        description=(
            "Opens a Tkinter GUI with a 4×4 molecule grid.  Click any thumbnail "
            "for a labelled 2-D structure and spin-group table."
        ),
    )
    p_view.add_argument(
        "--file", default=str(_DEFAULT_OUTPUT), metavar="PATH",
        help=(
            f"CSV to browse  (default: {_DEFAULT_OUTPUT.name}). "
            "Any pipeline CSV is accepted."
        ),
    )
    p_view.add_argument(
        "--n", type=int, default=80, metavar="N",
        help="Molecules to sample for the gallery  (default: 80)",
    )
    p_view.add_argument(
        "--seed", type=int, default=42, metavar="SEED",
        help="Random seed  (default: 42)",
    )
    p_view.set_defaults(func=_cmd_view)

    return parser


def main() -> None:
    """Entry point for the ``spinhance-gen`` CLI command."""
    parser = build_parser()
    args   = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
