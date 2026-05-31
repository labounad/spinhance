"""generate/cli.py — unified command-line interface for the generate pipeline.

Subcommands
-----------

``run``
    Screen the full ChEMBL database end-to-end in a single streaming pass.
    Produces ``chembl_8spin.csv``.

``view``
    Launch the interactive Tkinter gallery viewer.  Defaults to
    ``chembl_8spin.csv`` so no arguments are required after screening.

Usage
-----
::

    # All three invocation styles are equivalent:
    python generate/cli.py run
    python -m generate.cli run
    spinhance-gen run               # after pip install -e .

    python generate/cli.py view
    python generate/cli.py view --file generate/data/chembl_8spin.csv

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
_DEFAULT_OUTPUT   = _REPO_ROOT / "generate" / "data" / "chembl_8spin.csv"
_DEFAULT_XYZ      = _REPO_ROOT / "generate" / "data" / "chembl_8spin.xyz.gz"
_DEFAULT_WORKERS  = max(1, (os.cpu_count() or 2) - 1)
_DEFAULT_CHUNK    = 32


# ── Command handlers (all heavy imports deferred to here) ────────────────────

def _cmd_run(args: argparse.Namespace) -> int:
    from generate.pipeline import run_pipeline  # noqa: PLC0415 — intentional lazy import
    # --groups N is shorthand for --min-groups N --max-groups N; an explicit
    # --min/--max-groups overrides it.
    min_g = args.min_groups if args.min_groups is not None else args.n_groups
    max_g = args.max_groups if args.max_groups is not None else args.n_groups
    _, kept = run_pipeline(
        source_path     = Path(args.chembl),
        output_path     = Path(args.output),
        source          = args.source,
        xyz_path        = None if args.no_xyz else Path(args.xyz_output),
        min_groups      = min_g,
        max_groups      = max_g,
        max_heavy_atoms = args.max_heavy_atoms,
        num_shards      = args.num_shards,
        shard_index     = args.shard_index,
        workers         = args.workers,
        chunk_size      = args.chunk_size,
    )
    return 0 if kept >= 0 else 1


def _cmd_merge(args: argparse.Namespace) -> int:
    from generate.merge_shards import merge_shards  # noqa: PLC0415
    out_csv = Path(args.output)
    out_xyz = None if args.no_xyz else Path(args.xyz_output)
    rows, n_xyz = merge_shards(Path(args.shard_dir), out_csv, out_xyz)
    print(f"merged {rows:,} rows -> {out_csv}")
    if not args.no_xyz:
        print(f"merged {n_xyz} shard XYZ files -> {out_xyz}")
    if args.dedup:
        from generate.dedup import dedup_dataset  # noqa: PLC0415
        tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
        tmp_xyz = out_xyz.with_suffix(out_xyz.suffix + ".tmp") if out_xyz else None
        kept, dropped, nx = dedup_dataset(
            out_csv, tmp_csv, in_xyz=out_xyz, out_xyz=tmp_xyz,
        )
        tmp_csv.replace(out_csv)
        if tmp_xyz:
            tmp_xyz.replace(out_xyz)
        print(f"deduped: kept {kept:,} unique (dropped {dropped:,} duplicate InChIKeys)")
    return 0


def _cmd_dedup(args: argparse.Namespace) -> int:
    from generate.dedup import dedup_dataset  # noqa: PLC0415
    kept, dropped, n_xyz = dedup_dataset(
        Path(args.in_csv), Path(args.out_csv),
        in_xyz=Path(args.in_xyz) if args.in_xyz else None,
        out_xyz=Path(args.out_xyz) if args.out_xyz else None,
    )
    print(f"kept {kept:,} unique  (dropped {dropped:,} duplicate InChIKeys) -> {args.out_csv}")
    if args.out_xyz:
        print(f"wrote {n_xyz:,} XYZ blocks -> {args.out_xyz}")
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    from generate.buckets import split_dataset  # noqa: PLC0415
    csv_counts, xyz_counts = split_dataset(
        Path(args.in_csv), Path(args.out_dir), args.prefix,
        in_xyz=Path(args.xyz) if args.xyz else None,
    )
    total = sum(csv_counts.values())
    print(f"split {total:,} rows into {len(csv_counts)} buckets "
          f"-> {args.out_dir}/{args.prefix}_<n>spin.csv.gz")
    for n in sorted(csv_counts):
        x = f", {xyz_counts.get(n, 0):,} xyz" if xyz_counts else ""
        print(f"  {n:>2}spin: {csv_counts[n]:>10,} rows{x}")
    return 0


def _cmd_xyz(args: argparse.Namespace) -> int:
    from generate.xyz_writer import write_xyz_gz  # noqa: PLC0415
    _, written = write_xyz_gz(
        input_path  = Path(args.input),
        output_path = Path(args.output),
        workers     = args.workers,
    )
    return 0 if written >= 0 else 1


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
        help="Screen ChEMBL end-to-end and write chembl_8spin.csv + chembl_8spin.xyz.gz.",
        description=(
            "Single-pass pipeline: streams ChEMBL, applies the fast proton-count "
            "heuristic, then the 3-D deuterium substitution test.  Molecules with "
            f"exactly N spin groups are written to {_DEFAULT_OUTPUT.name}, and "
            f"their annotated 3-D structures to {_DEFAULT_XYZ.name} in the same "
            "pass (use --no-xyz to skip).  This fuses the old 'run' and 'xyz' "
            "steps, avoiding a second 3-D embedding of every kept molecule."
        ),
    )
    p_run.add_argument(
        "--source", default="chembl", metavar="SRC",
        choices=("chembl", "pubchem", "zinc", "smiles"),
        help="Input database format: chembl | pubchem | zinc | smiles  "
             "(default: chembl).  Selects how --chembl is parsed.",
    )
    p_run.add_argument(
        "--chembl", default=str(_DEFAULT_CHEMBL), metavar="PATH",
        help=f"Input compound file; .gz handled transparently.  For "
             f"--source pubchem this is CID-SMILES.gz  "
             f"(default: {_DEFAULT_CHEMBL.name})",
    )
    p_run.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT), metavar="PATH",
        help=f"Output CSV  (default: {_DEFAULT_OUTPUT.name})",
    )
    p_run.add_argument(
        "--xyz-output", default=str(_DEFAULT_XYZ), metavar="PATH",
        dest="xyz_output",
        help=f"Fused XYZ output  (default: {_DEFAULT_XYZ.name})",
    )
    p_run.add_argument(
        "--no-xyz", action="store_true",
        help="Skip XYZ generation; write the CSV only (old 'run' behaviour).",
    )
    p_run.add_argument(
        "--n-groups", type=int, default=N_SPIN_GROUPS, metavar="N",
        help=f"Exact spin-group count to select  (default: {N_SPIN_GROUPS}). "
             "Shorthand for --min-groups N --max-groups N.",
    )
    p_run.add_argument(
        "--min-groups", type=int, default=None, metavar="N", dest="min_groups",
        help="Minimum spin-group count to keep (overrides --n-groups). "
             "Use with --max-groups for a categorising scan, e.g. 1..26.",
    )
    p_run.add_argument(
        "--max-groups", type=int, default=None, metavar="N", dest="max_groups",
        help="Maximum spin-group count to keep (overrides --n-groups).",
    )
    p_run.add_argument(
        "--max-heavy-atoms", type=int, default=50, metavar="N",
        dest="max_heavy_atoms",
        help="Skip molecules with more than N heavy atoms before embedding; "
             "0 disables  (default: 50, ~600-700 Da).",
    )
    p_run.add_argument(
        "--num-shards", type=int, default=None, metavar="N",
        dest="num_shards",
        help="Split the input into N shards; process only --shard-index.  "
             "For Slurm arrays (default: no sharding).",
    )
    p_run.add_argument(
        "--shard-index", type=int, default=0, metavar="K",
        dest="shard_index",
        help="Which shard (0..N-1) this task processes  (default: 0).",
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

    # ── xyz ───────────────────────────────────────────────────────────────────
    _xyz_in  = _REPO_ROOT / "generate" / "data" / "chembl_8spin.csv"
    _xyz_out = _REPO_ROOT / "generate" / "data" / "chembl_8spin.xyz.gz"

    p_xyz = sub.add_parser(
        "xyz",
        help="Convert chembl_8spin.csv to a gzip-compressed multi-XYZ file.",
        description=(
            "Reads chembl_8spin.csv, embeds each molecule in 3-D, classifies spin groups, "
            "and writes a single gzip-compressed multi-XYZ file.  "
            "Each H atom is annotated with its group letter, tier (H/S/N), "
            "and chemical-shift class number.  "
            f"~100k molecules compress to ~10-12 MB."
        ),
    )
    p_xyz.add_argument(
        "--input",  default=str(_xyz_in),  metavar="PATH",
        help=f"Input CSV  (default: {_xyz_in.name})",
    )
    p_xyz.add_argument(
        "--output", default=str(_xyz_out), metavar="PATH",
        help=f"Output .xyz.gz  (default: {_xyz_out.name})",
    )
    p_xyz.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS, metavar="N",
        help=f"Worker processes  (default: {_DEFAULT_WORKERS})",
    )
    p_xyz.set_defaults(func=_cmd_xyz)

    # ── merge ─────────────────────────────────────────────────────────────────
    p_merge = sub.add_parser(
        "merge",
        help="Combine Slurm-array shard outputs into one CSV + XYZ.",
        description=(
            "Merges part_*.csv and part_*.xyz.gz produced by a sharded run "
            "(--num-shards) into single-file outputs."
        ),
    )
    p_merge.add_argument(
        "shard_dir", metavar="SHARD_DIR",
        help="Directory containing part_*.csv / part_*.xyz.gz.",
    )
    p_merge.add_argument(
        "--output", default=str(_DEFAULT_OUTPUT), metavar="PATH",
        help=f"Merged CSV  (default: {_DEFAULT_OUTPUT.name})",
    )
    p_merge.add_argument(
        "--xyz-output", default=str(_DEFAULT_XYZ), metavar="PATH",
        dest="xyz_output",
        help=f"Merged .xyz.gz  (default: {_DEFAULT_XYZ.name})",
    )
    p_merge.add_argument(
        "--no-xyz", action="store_true",
        help="Merge only the CSV shards.",
    )
    p_merge.add_argument(
        "--dedup", action="store_true",
        help="After merging, collapse rows sharing an InChIKey to one "
             "(first occurrence) and filter the XYZ to match.",
    )
    p_merge.set_defaults(func=_cmd_merge)

    # ── dedup ─────────────────────────────────────────────────────────────────
    p_dedup = sub.add_parser(
        "dedup",
        help="Drop duplicate molecules (same InChIKey) from a merged dataset.",
        description=(
            "Collapse CSV rows sharing an InChIKey to the first occurrence and "
            "filter the companion multi-XYZ to the surviving IDs."
        ),
    )
    p_dedup.add_argument("in_csv",  metavar="IN.csv")
    p_dedup.add_argument("out_csv", metavar="OUT.csv")
    p_dedup.add_argument("--in-xyz",  dest="in_xyz",  default=None, metavar="IN.xyz.gz")
    p_dedup.add_argument("--out-xyz", dest="out_xyz", default=None, metavar="OUT.xyz.gz")
    p_dedup.set_defaults(func=_cmd_dedup)

    # ── split ─────────────────────────────────────────────────────────────────
    p_split = sub.add_parser(
        "split",
        help="Split a categorising scan into per-spin-count datasets.",
        description=(
            "Partition a combined range-scan CSV (+ XYZ) into "
            "<prefix>_<n>spin.{csv,xyz.gz}, one file per spin-group count."
        ),
    )
    p_split.add_argument("in_csv",  metavar="COMBINED.csv")
    p_split.add_argument("out_dir", metavar="OUT_DIR")
    p_split.add_argument("--prefix", default="pubchem", metavar="NAME",
                         help="Output filename prefix (default: pubchem).")
    p_split.add_argument("--xyz", default=None, metavar="COMBINED.xyz.gz",
                         help="Also split this combined multi-XYZ to match.")
    p_split.set_defaults(func=_cmd_split)

    return parser


def main() -> None:
    """Entry point for the ``spinhance-gen`` CLI command."""
    parser = build_parser()
    args   = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
