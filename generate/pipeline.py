"""generate/pipeline.py — end-to-end molecule screening pipeline.

Streams the ChEMBL chemreps file and writes qualifying molecules directly to
``chembl_8spin.csv`` in a single pass with no intermediate file.

Two filters run in sequence for every molecule:

1. **Heuristic pre-filter** (:func:`~generate.spin_equivalence.passes_heuristic`)
   — O(1) atom-count check run in the main process.  Molecules that obviously
   cannot have the target number of spin groups are discarded immediately,
   before any 3-D work is done.

2. **Exact deuterium test** — 3-D embedding + deuterium substitution, run in
   parallel worker processes.

Parallelism model
-----------------
The main process streams ChEMBL line-by-line, applies the cheap heuristic,
and assembles heuristic-passing molecules into fixed-size chunks.  Each chunk
is dispatched to a :class:`~concurrent.futures.ProcessPoolExecutor` worker.
Workers import RDKit independently (safe under both fork and spawn), run the
expensive 3-D embedding and deuterium test, and return only molecules that
pass.  The main process collects results as futures complete and writes them
directly to the output CSV.

The number of in-flight futures is capped at ``workers × 4`` so the main
process never builds an unbounded backlog of pending work.

Running
-------
::

    python generate/cli.py run
    python generate/cli.py run --workers 8 --chunk-size 64
    python generate/cli.py run --n-groups 6
    spinhance-gen run                   # after pip install -e .
"""

from __future__ import annotations

import csv
import gzip
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as _fut_wait
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem, RDLogger  # noqa: E402

from generate.config import N_SPIN_GROUPS  # noqa: E402
from generate.sources import HAS_HEADER, iter_compounds  # noqa: E402
from generate.spin_equivalence import passes_heuristic  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHEMBL     = _REPO_ROOT / "generate" / "chembl" / "chembl_37_chemreps.txt"
DEFAULT_OUTPUT     = _REPO_ROOT / "generate" / "data" / "chembl_8spin.csv"
DEFAULT_XYZ_OUTPUT = _REPO_ROOT / "generate" / "data" / "chembl_8spin.xyz.gz"
DEFAULT_WORKERS    = max(1, (os.cpu_count() or 2) - 1)
DEFAULT_CHUNK_SIZE = 32

#: Reject molecules with more than this many heavy atoms *before* the
#: expensive 3-D embedding.  A clean ≤8-spin-group molecule is small; large
#: structures (peptides, polymers, macrocycles) embed slowly, often fail, and
#: never yield 8 groups anyway.  ~50 heavy atoms ≈ 600-700 Da.  Set to 0 to
#: disable the cap.
DEFAULT_MAX_HEAVY_ATOMS = 50


# ── Worker function ───────────────────────────────────────────────────────────

def _screen_chunk(
    chunk: list[tuple[str, str, str]],
    target_groups: int,
    want_xyz: bool = False,
) -> list[tuple[tuple[str, str, str, int, str], str | None]]:
    """Classify a batch of heuristic-passing molecules in a single 3-D pass.

    This function runs inside worker processes.  Imports are deferred to the
    function body so it is picklable under both ``fork`` and ``spawn``
    multiprocessing start methods.

    The molecule is embedded and classified **once** via
    :func:`~generate.spin_equivalence.classify_spin_groups`.  The spin-group
    count and per-group sizes (formerly recomputed by ``analyze_spin_systems``)
    are derived from the returned ``SpinGroup`` list, and — when *want_xyz* is
    set — the annotated XYZ block is rendered from the same classified molecule.
    This fuses the old ``run`` and ``xyz`` phases, eliminating the second 3-D
    embedding that ``xyz`` previously performed for every kept molecule.

    Parameters
    ----------
    chunk:
        List of ``(chembl_id, smiles, inchikey)`` tuples, all of which have
        already passed the heuristic pre-filter in the main process.
    target_groups:
        Number of magnetically distinct spin groups to select for.
    want_xyz:
        When ``True``, also render the annotated XYZ block for each kept
        molecule (the InChI computation it requires is skipped otherwise).

    Returns
    -------
    list of ``((chembl_id, smiles, inchikey, n_groups, group_sizes_str), xyz)``
        One entry per molecule that has exactly *target_groups* spin groups.
        *group_sizes_str* is a semicolon-separated string of per-group proton
        counts sorted descending (e.g. ``"3;3;1;1;1;1;1;1"``).  *xyz* is the
        XYZ block string when *want_xyz* is set and embedding succeeded, else
        ``None``.
    """
    # Deferred imports — safe across fork and spawn.
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    from generate.spin_equivalence import classify_spin_groups  # noqa: PLC0415
    if want_xyz:
        from generate.xyz_writer import build_xyz_block  # noqa: PLC0415

    results: list[tuple[tuple[str, str, str, int, str], str | None]] = []

    for source_id, smiles, inchikey in chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        try:
            mol_h, groups = classify_spin_groups(mol)
        except Exception:
            continue

        # n_groups and per-group sizes are exactly what analyze_spin_systems
        # returned: one entry per SpinGroup, sized by its proton count.
        if len(groups) != target_groups:
            continue

        # Sources without an InChIKey column (PubChem, ZINC) pass "" — compute
        # it here, only for the molecules that survive, never for the whole DB.
        if not inchikey:
            try:
                inchikey = Chem.MolToInchiKey(mol)
            except Exception:
                inchikey = ""

        group_sizes = sorted((len(g.h_indices) for g in groups), reverse=True)
        row = (
            source_id, smiles, inchikey,
            len(groups), ";".join(map(str, group_sizes)),
        )
        xyz = (
            build_xyz_block(
                mol_h, groups,
                smiles=smiles, chembl_id=source_id, inchikey=inchikey,
            )
            if want_xyz else None
        )
        results.append((row, xyz))

    return results


# ── Line counter ──────────────────────────────────────────────────────────────

def _count_molecules(path: Path, *, has_header: bool) -> int:
    """Count data rows in *path* (total lines, minus a header if present).

    Reads the file in 1 MB binary chunks and counts newline bytes — faster
    than iterating over decoded text lines for a 700+ MB file.  Gzipped
    inputs are decompressed on the fly; for very large compressed databases
    (PubChem) the caller may prefer to skip the count entirely.
    """
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rb") as f:
        n = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    return max(0, n - (1 if has_header else 0))


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    source_path: Path = DEFAULT_CHEMBL,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    source: str = "chembl",
    xyz_path: Path | None = DEFAULT_XYZ_OUTPUT,
    target_groups: int  = N_SPIN_GROUPS,
    max_heavy_atoms: int = DEFAULT_MAX_HEAVY_ATOMS,
    num_shards: int | None = None,
    shard_index: int    = 0,
    workers: int        = DEFAULT_WORKERS,
    chunk_size: int     = DEFAULT_CHUNK_SIZE,
    verbose: bool       = True,
) -> tuple[int, int]:
    """Stream a compound database for molecules with exactly *target_groups* groups.

    The input format is selected by *source* (see :mod:`generate.sources`):
    ``chembl`` (default), ``pubchem``, ``zinc``, or a generic ``smiles`` file.
    The screening itself is source-agnostic — only the ``(id, SMILES)`` parsing
    differs.  Sources without an InChIKey column have it computed for the
    molecules that pass (never for the whole database).

    When *xyz_path* is given (the default), the annotated 3-D XYZ block for each
    kept molecule is written in the **same pass**, fusing the old ``run`` and
    ``xyz`` phases.  Because both outputs come from a single
    :func:`~generate.spin_equivalence.classify_spin_groups` call per molecule —
    using the same fixed-seed embedding the standalone ``xyz`` command would
    use — the fused ``chembl_8spin.xyz.gz`` is identical (modulo block order) to
    running ``run`` then ``xyz``, but skips re-embedding every kept molecule.

    Parameters
    ----------
    source_path:
        Compound file to screen (``.gz`` handled transparently).  For
        ``source="pubchem"`` this is ``CID-SMILES.gz``.
    output_path:
        Destination CSV.  Parent directories are created if absent.
        Columns: ``chembl_id``, ``smiles``, ``inchikey``,
        ``n_groups``, ``group_sizes``.  (The ``chembl_id`` column header is
        kept for downstream compatibility; for PubChem it holds ``CID<n>``.)
    source:
        Input database format — one of :data:`generate.sources.SOURCES`.
    xyz_path:
        Destination gzip multi-XYZ file, or ``None`` to skip XYZ generation
        (CSV only — the old ``run`` behaviour).  Parent directories are
        created if absent.
    target_groups:
        Number of magnetically distinct ¹H spin groups to select for.
        Defaults to :data:`~generate.config.N_SPIN_GROUPS`.
    max_heavy_atoms:
        Skip molecules with more than this many heavy atoms before embedding
        (peptides/polymers/macrocycles are slow and never qualify).  ``0``
        disables the cap.  Defaults to :data:`DEFAULT_MAX_HEAVY_ATOMS`.
    num_shards, shard_index:
        Split one input across a Slurm array: each task processes only the
        records whose global index satisfies ``index % num_shards ==
        shard_index``.  ``num_shards=None`` (default) processes everything.
        Each shard writes its own CSV / XYZ; combine with
        :mod:`generate.merge_shards`.
    workers:
        Number of worker processes for the 3-D deuterium test.
        Defaults to ``cpu_count - 1``.
    chunk_size:
        Molecules per work unit dispatched to a worker.  Larger chunks
        reduce IPC overhead; smaller chunks improve load balancing.
        32 is a good default for drug-like molecules (~100-500 ms each).
    verbose:
        Print a progress bar and summary when ``True``.

    Returns
    -------
    total : int
        Molecules examined (parse failures excluded).
    kept : int
        Molecules written to *output_path*.
    """
    want_xyz = xyz_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if want_xyz:
        xyz_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-counting means a full pass over the input.  Cheap for an
    # uncompressed ChEMBL TSV; for a multi-GB gzip (PubChem) it means
    # decompressing the whole file twice, so skip it and let the bar run
    # without a known total.
    pbar = None
    if verbose:
        try:
            from tqdm import tqdm  # noqa: PLC0415
            if Path(source_path).suffix == ".gz":
                n_total = None
            else:
                print(f"Counting molecules in {Path(source_path).name}…",
                      end=" ", flush=True)
                n_total = _count_molecules(
                    source_path, has_header=source in HAS_HEADER
                )
                print(f"{n_total:,}")
            pbar = tqdm(
                total=n_total,
                desc=f"Screening  [{workers} workers, chunk={chunk_size}]",
                unit=" mol",
                mininterval=1.0,
                dynamic_ncols=True,
            )
        except ImportError:
            pass

    total        = 0
    heur_pass    = 0
    parse_failed = 0
    too_large    = 0
    kept         = 0
    xyz_written  = 0
    sharded      = num_shards is not None and num_shards > 1

    # Cap pending futures to avoid an unbounded backlog.
    max_pending = workers * 4

    with (
        open(output_path, "w", newline="") as f_out,
        (gzip.open(xyz_path, "wt", encoding="utf-8", compresslevel=6)
         if want_xyz else nullcontext()) as gz,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        writer = csv.writer(f_out)
        writer.writerow(
            ["chembl_id", "smiles", "inchikey", "n_groups", "group_sizes"]
        )

        chunk:   list[tuple[str, str, str]] = []
        pending: set = set()

        def _flush_chunk() -> None:
            """Submit the current chunk to the pool, draining if needed."""
            nonlocal pending
            if not chunk:
                return
            if len(pending) >= max_pending:
                # Block until at least one future completes before submitting.
                done, pending = _fut_wait(pending, return_when=FIRST_COMPLETED)
                _write_done(done)
            pending.add(
                pool.submit(_screen_chunk, list(chunk), target_groups, want_xyz)
            )
            chunk.clear()

        def _write_done(futures) -> None:
            """Write results from a set of completed futures; refresh postfix."""
            nonlocal kept, xyz_written
            for fut in futures:
                for row, xyz in fut.result():
                    writer.writerow(row)
                    kept += 1
                    if want_xyz and xyz is not None:
                        gz.write(xyz)
                        xyz_written += 1
            if pbar is not None:
                pbar.set_postfix(
                    heur=f"{heur_pass:,}",
                    kept=f"{kept:,}",
                    refresh=False,
                )

        for raw_idx, (source_id, smiles, inchikey) in enumerate(
            iter_compounds(source_path, source)
        ):
            # Shard selection runs on the raw record index, before any
            # filtering, so shard boundaries are deterministic and balanced
            # regardless of how many molecules each task ends up keeping.
            if sharded and raw_idx % num_shards != shard_index:
                continue

            if "." in smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                parse_failed += 1
                # Still advance the bar — these are counted in the total.
                if pbar is not None:
                    pbar.update(1)
                continue

            total += 1
            if pbar is not None:
                pbar.update(1)

            # Size cap before the expensive embedding: peptides/polymers are
            # slow to embed and never produce a clean 8-spin-group system.
            if max_heavy_atoms and mol.GetNumHeavyAtoms() > max_heavy_atoms:
                too_large += 1
                continue

            ok, _, _ = passes_heuristic(mol)
            if not ok:
                continue

            heur_pass += 1
            chunk.append((source_id, smiles, inchikey))

            if len(chunk) >= chunk_size:
                _flush_chunk()

        # Submit any remaining partial chunk.
        _flush_chunk()

        # Drain all in-flight futures.
        if pending:
            done, _ = _fut_wait(pending, return_when="ALL_COMPLETED")
            _write_done(done)

    if pbar is not None:
        pbar.close()

    if verbose:
        if sharded:
            print(f"\nShard           : {shard_index} of {num_shards}")
        print(f"Examined        : {total:>10,}")
        print(f"  parse failures: {parse_failed:>10,}")
        if max_heavy_atoms:
            print(f"  too large (>{max_heavy_atoms} heavy): {too_large:>10,}")
        print(f"  heuristic pass: {heur_pass:>10,}  ({100 * heur_pass / max(total, 1):.1f}%)")
        print(f"Kept (n={target_groups})      : {kept:>10,}  ({100 * kept / max(total, 1):.2f}%)")
        print(f"Output          : {output_path}")
        if want_xyz:
            size_mb = xyz_path.stat().st_size / 1e6
            print(f"XYZ written     : {xyz_written:>10,}  ({size_mb:.1f} MB compressed)")
            print(f"XYZ output      : {xyz_path}")
            n_embed_fail = kept - xyz_written
            if n_embed_fail:
                print(f"  (no 3-D conf  : {n_embed_fail:>8,} kept molecules omitted from XYZ)")

    return total, kept
