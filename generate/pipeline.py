"""generate/pipeline.py — end-to-end molecule screening pipeline.

Streams the ChEMBL chemreps file and writes qualifying molecules directly to
``8spin.csv`` in a single pass with no intermediate file.

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
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as _fut_wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit import Chem, RDLogger  # noqa: E402

from generate.config import N_SPIN_GROUPS  # noqa: E402
from generate.spin_equivalence import passes_heuristic  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CHEMBL     = _REPO_ROOT / "generate" / "chembl" / "chembl_37_chemreps.txt"
DEFAULT_OUTPUT     = _REPO_ROOT / "generate" / "data" / "8spin.csv"
DEFAULT_WORKERS    = max(1, (os.cpu_count() or 2) - 1)
DEFAULT_CHUNK_SIZE = 32


# ── Worker function ───────────────────────────────────────────────────────────

def _screen_chunk(
    chunk: list[tuple[str, str, str]],
    target_groups: int,
) -> list[tuple[str, str, str, int, str]]:
    """Apply the 3-D deuterium test to a batch of heuristic-passing molecules.

    This function runs inside worker processes.  Imports are deferred to the
    function body so it is picklable under both ``fork`` and ``spawn``
    multiprocessing start methods.

    Parameters
    ----------
    chunk:
        List of ``(chembl_id, smiles, inchikey)`` tuples, all of which have
        already passed the heuristic pre-filter in the main process.
    target_groups:
        Number of magnetically distinct spin groups to select for.

    Returns
    -------
    list of ``(chembl_id, smiles, inchikey, n_groups, group_sizes_str)``
        One entry per molecule that has exactly *target_groups* spin groups.
        *group_sizes_str* is a semicolon-separated string of per-group proton
        counts sorted descending (e.g. ``"3;3;1;1;1;1;1;1"``).
    """
    # Deferred imports — safe across fork and spawn.
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    from generate.spin_equivalence import analyze_spin_systems  # noqa: PLC0415

    results: list[tuple[str, str, str, int, str]] = []

    for chembl_id, smiles, inchikey in chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        n_groups, group_sizes = analyze_spin_systems(mol)
        if n_groups == target_groups:
            results.append((
                chembl_id, smiles, inchikey,
                n_groups, ";".join(map(str, group_sizes)),
            ))

    return results


# ── Line counter ──────────────────────────────────────────────────────────────

def _count_molecules(path: Path) -> int:
    """Count data rows in *path* (total lines minus the header) efficiently.

    Reads the file in 1 MB binary chunks and counts newline bytes — faster
    than iterating over decoded text lines for a 700+ MB file.
    """
    with open(path, "rb") as f:
        n = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    return max(0, n - 1)  # subtract the header line


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    chembl_path: Path = DEFAULT_CHEMBL,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    target_groups: int  = N_SPIN_GROUPS,
    workers: int        = DEFAULT_WORKERS,
    chunk_size: int     = DEFAULT_CHUNK_SIZE,
    verbose: bool       = True,
) -> tuple[int, int]:
    """Stream ChEMBL and write molecules with exactly *target_groups* spin groups.

    Parameters
    ----------
    chembl_path:
        Path to the ChEMBL ``chembl_XX_chemreps.txt`` tab-separated file.
    output_path:
        Destination CSV.  Parent directories are created if absent.
        Columns: ``chembl_id``, ``smiles``, ``inchikey``,
        ``n_groups``, ``group_sizes``.
    target_groups:
        Number of magnetically distinct ¹H spin groups to select for.
        Defaults to :data:`~generate.config.N_SPIN_GROUPS`.
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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = None
    if verbose:
        try:
            from tqdm import tqdm  # noqa: PLC0415
            print(f"Counting molecules in {chembl_path.name}…", end=" ", flush=True)
            n_total = _count_molecules(chembl_path)
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
    kept         = 0

    # Cap pending futures to avoid an unbounded backlog.
    max_pending = workers * 4

    with (
        open(chembl_path) as f_in,
        open(output_path, "w", newline="") as f_out,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        writer = csv.writer(f_out)
        writer.writerow(
            ["chembl_id", "smiles", "inchikey", "n_groups", "group_sizes"]
        )

        lines = iter(f_in)
        next(lines)  # skip header

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
            pending.add(pool.submit(_screen_chunk, list(chunk), target_groups))
            chunk.clear()

        def _write_done(futures) -> None:
            """Write results from a set of completed futures; refresh postfix."""
            nonlocal kept
            for fut in futures:
                for row in fut.result():
                    writer.writerow(row)
                    kept += 1
            if pbar is not None:
                pbar.set_postfix(
                    heur=f"{heur_pass:,}",
                    kept=f"{kept:,}",
                    refresh=False,
                )

        for line in lines:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            chembl_id, smiles, inchikey = parts[0], parts[1], parts[3]

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

            ok, _, _ = passes_heuristic(mol)
            if not ok:
                continue

            heur_pass += 1
            chunk.append((chembl_id, smiles, inchikey))

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
        print(f"\nExamined        : {total:>10,}")
        print(f"  parse failures: {parse_failed:>10,}")
        print(f"  heuristic pass: {heur_pass:>10,}  ({100 * heur_pass / max(total, 1):.1f}%)")
        print(f"Kept (n={target_groups})      : {kept:>10,}  ({100 * kept / max(total, 1):.2f}%)")
        print(f"Output          : {output_path}")

    return total, kept
