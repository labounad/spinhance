"""generate/cli.py — unified command-line interface for the generate pipeline.

Subcommands
-----------

``run``
    Screen the full ChEMBL database end-to-end in a single streaming pass:
    the heuristic pre-filter and the exact 3-D deuterium substitution test
    run back-to-back with no intermediate file.  Produces
    ``candidates_final.csv``.

``view``
    Launch the interactive Tkinter gallery viewer.  Defaults to
    ``candidates_final.csv`` so no arguments are required after screening.

Usage
-----
::

    # From the repo root (all three forms are equivalent):
    python generate/cli.py run
    python -m generate.cli run
    spinhance-gen run               # after pip install -e .

    python generate/cli.py view
    python generate/cli.py view --file generate/data/candidates_final.csv

    # Override ChEMBL source, output path, or spin-group target:
    python generate/cli.py run \\
        --chembl /data/chembl_37_chemreps.txt \\
        --output /data/candidates_final.csv \\
        --n-groups 6

Notes
-----
``sys.path.insert`` at module level ensures ``python generate/cli.py``
(direct invocation) behaves identically to ``python -m generate.cli``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate.config import N_SPIN_GROUPS  # noqa: E402
from generate.pipeline import (  # noqa: E402
    DEFAULT_CHEMBL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OUTPUT,
    DEFAULT_WORKERS,
    run_pipeline,
)


def _cmd_run(args: argparse.Namespace) -> int:
    _, kept = run_pipeline(
        chembl_path   = Path(args.chembl),
        output_path   = Path(args.output),
        target_groups = args.n_groups,
        workers       = args.workers,
        chunk_size    = args.chunk_size,
    )
    return 0 if kept >= 0 else 1


def _cmd_view(args: argparse.Namespace) -> int:
    # Deferred import: tkinter raises on headless hosts, so we only import
    # the viewer when the view subcommand is actually invoked.
    from generate.viewer import launch
    launch(Path(args.file), n=args.n, seed=args.seed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="spinhance-gen",
        description=(
            "SpinHance Task 1 — screen ChEMBL for molecules with exactly "
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
            f"exactly N spin groups are written to {DEFAULT_OUTPUT.name}."
        ),
    )
    p_run.add_argument(
        "--chembl",
        default=str(DEFAULT_CHEMBL),
        metavar="PATH",
        help=f"ChEMBL chemreps .txt file  (default: {DEFAULT_CHEMBL.name})",
    )
    p_run.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="PATH",
        help=f"Output CSV  (default: {DEFAULT_OUTPUT.name})",
    )
    p_run.add_argument(
        "--n-groups",
        type=int,
        default=N_SPIN_GROUPS,
        metavar="N",
        help=f"Target spin-group count  (default: {N_SPIN_GROUPS})",
    )
    p_run.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Worker processes for the 3-D deuterium test  (default: {DEFAULT_WORKERS})",
    )
    p_run.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="N",
        dest="chunk_size",
        help=(
            f"Molecules per work unit dispatched to a worker  "
            f"(default: {DEFAULT_CHUNK_SIZE}). "
            "Larger = less IPC overhead; smaller = better load balancing."
        ),
    )
    p_run.set_defaults(func=_cmd_run)

    # ── view ──────────────────────────────────────────────────────────────────
    p_view = sub.add_parser(
        "view",
        help="Launch the interactive gallery viewer.",
        description=(
            "Opens a Tkinter GUI showing a 4×4 paginated gallery of molecules. "
            "Click any thumbnail for a detail pane with structure, stats, SMILES, "
            "and a collapsible deuterium-substitution sidebar."
        ),
    )
    p_view.add_argument(
        "--file",
        default=str(DEFAULT_OUTPUT),
        metavar="PATH",
        help=(
            f"CSV to browse  (default: {DEFAULT_OUTPUT.name} — the final screened set). "
            "Any pipeline CSV is accepted."
        ),
    )
    p_view.add_argument(
        "--n",
        type=int,
        default=80,
        metavar="N",
        help="Molecules to sample for the gallery  (default: 80)",
    )
    p_view.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for the sample  (default: 42)",
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
