"""
pyspin.batch
============
Batch driver for the pure-Python simulator: turn a directory of spin-system
XMLs into normalised ``.npy`` spectra, in parallel across CPU cores (and HPC
nodes). No MestReNova, no license, no temp txt files.

Output layout matches the MNova pipeline so Task 4 consumes either identically:

    <out_dir>/spectra/<field>MHz/<stem>.npy   (+ ppm_axis.npy per field)
"""

from __future__ import annotations

# Pin BLAS to one thread per process BEFORE numpy is imported: parallelism is
# across molecules, so per-sim multi-threaded BLAS only oversubscribes cores.
# (Set first so 'spawn' children inherit single-threaded numpy.)
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import multiprocessing as mp
from pathlib import Path

import numpy as np

from simulation.pyspin.cluster import simulate_spectrum_pyspin
from simulation.xml_io import xml_to_matrix

try:                                  # optional pretty progress bar
    from tqdm import tqdm as _tqdm
except Exception:                     # tqdm not installed → no-op passthrough
    _tqdm = None

__all__ = ["simulate_xml_to_npy", "run_pyspin_batch", "run_pyspin_batch_graphs"]


def _progress(iterable, total, desc):
    """Wrap an iterable in a tqdm bar if tqdm is available, else pass through."""
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, total=total, desc=desc, unit="sim")


def simulate_xml_to_npy(
    xml_path: Path,
    field_mhz: float,
    out_npy: Path,
    points: int = 16384,
    ppm_from: float = 0.0,
    ppm_to: float = 12.0,
    linewidth_hz: float = 1.0,
) -> Path:
    """Parse one XML, simulate at ``field_mhz``, save a normalised ``.npy``.

    Uses the wall-free dispatcher: exact composite for small coupled fragments,
    clustered approximation for large ones — so no molecule can stall the batch.
    """
    m = xml_to_matrix(xml_path)
    _, spec = simulate_spectrum_pyspin(
        m["shifts"], m["couplings"], m["degeneracy"], field_mhz,
        points=points, ppm_from=ppm_from, ppm_to=ppm_to, linewidth_hz=linewidth_hz,
    )
    out_npy = Path(out_npy)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, spec.astype(np.float32))
    return out_npy


def _worker(task):
    """Top-level (picklable) worker: one (xml, field) -> npy task."""
    xml_path, field, out_npy, points, ppm_from, ppm_to, linewidth = task
    try:
        simulate_xml_to_npy(Path(xml_path), field, Path(out_npy),
                            points=points, ppm_from=ppm_from, ppm_to=ppm_to,
                            linewidth_hz=linewidth)
        return (out_npy, True, "")
    except Exception as e:  # noqa: BLE001 - report, keep batch going
        return (out_npy, False, repr(e))


def run_pyspin_batch(
    source_xml_dir: Path,
    out_dir: Path,
    fields_mhz=(90.0, 600.0),
    workers: int | None = None,
    points: int = 16384,
    ppm_from: float = 0.0,
    ppm_to: float = 12.0,
    linewidth_hz: float = 1.0,
) -> dict:
    """Simulate every XML in ``source_xml_dir`` at each field, in parallel.

    Parameters
    ----------
    workers
        Process count (default: all CPU cores). Each task is one (molecule,
        field) pair — embarrassingly parallel.

    Returns ``{"tasks", "succeeded", "failed", "failures"}``.
    """
    source_xml_dir = Path(source_xml_dir)
    out_dir = Path(out_dir)
    xmls = sorted(source_xml_dir.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"No XML files found in {source_xml_dir}")

    workers = workers or os.cpu_count() or 1
    spectra_root = out_dir / "spectra"

    # One shared ppm axis per field directory.
    ppm_axis = np.linspace(ppm_from, ppm_to, points)
    tasks = []
    for field in fields_mhz:
        fdir = spectra_root / f"{field:.0f}MHz"
        fdir.mkdir(parents=True, exist_ok=True)
        np.save(fdir / "ppm_axis.npy", ppm_axis)
        for xml in xmls:
            tasks.append((str(xml), float(field), str(fdir / f"{xml.stem}.npy"),
                          points, ppm_from, ppm_to, linewidth_hz))

    print(f"pyspin batch: {len(xmls)} molecules × {len(fields_mhz)} fields "
          f"= {len(tasks)} sims on {workers} workers")

    if workers == 1:
        results = [_worker(t) for t in _progress(tasks, len(tasks), "simulating")]
    else:
        # 'spawn' children re-import this module → inherit the 1-thread BLAS env.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = list(_progress(pool.imap_unordered(_worker, tasks, chunksize=4),
                                     len(tasks), "simulating"))

    failures = [(o, err) for o, ok, err in results if not ok]
    succeeded = len(results) - len(failures)
    print(f"  done: {succeeded}/{len(tasks)} succeeded")
    for o, err in failures[:10]:
        print(f"  FAILED {o}: {err}")
    return {"tasks": len(tasks), "succeeded": succeeded,
            "failed": len(failures), "failures": failures}


# ── Graph input (Task 2 JSONL → npy), pyspin engine ──────────────────────────

# Process-local array cache; populated by _init_graph_workers in each worker.
# The main process sets this directly for the workers==1 path (no IPC).
_worker_graphs: dict | None = None  # idx -> (shifts, couplings, degeneracy)


def _init_graph_workers(jsonl_path: str) -> None:
    """Initializer: parse the spin-system file ONCE per worker process.

    Stores pre-extracted numeric arrays so task tuples only carry an index,
    eliminating the per-task IPC cost of pickling full graph dicts.
    """
    global _worker_graphs
    from simulation.graph_io import read_spin_systems, record_to_arrays
    _worker_graphs = {
        idx: record_to_arrays(rec)[1:]  # drop labels → (shifts, couplings, degeneracy)
        for idx, rec in read_spin_systems(jsonl_path)
    }


def _graph_worker(task):
    """Top-level picklable worker: (idx, field) → spectrum file.

    Reads spin-system arrays from the process-local cache set by
    ``_init_graph_workers``; task tuple carries only an integer index so IPC
    cost is O(1) per task regardless of molecule size.
    """
    idx, field, out_path, points, ppm_from, ppm_to, linewidth, fmt = task
    try:
        from simulation.spectrum_io import save_dense, save_peaks
        shifts, couplings, degeneracy = _worker_graphs[idx]
        if fmt == "peaks":
            from simulation.pyspin.cluster import transitions_pyspin
            centers, amps = transitions_pyspin(shifts, couplings, degeneracy, field)
            save_peaks(out_path, centers, amps, linewidth_hz=linewidth,
                       field_mhz=field, points=points, ppm_from=ppm_from, ppm_to=ppm_to)
        else:
            _, spec = simulate_spectrum_pyspin(
                shifts, couplings, degeneracy, field,
                points=points, ppm_from=ppm_from, ppm_to=ppm_to, linewidth_hz=linewidth)
            save_dense(out_path, spec)
        return (out_path, True, "")
    except Exception as e:  # noqa: BLE001
        return (out_path, False, repr(e))


def run_pyspin_batch_graphs(
    jsonl_path: Path,
    out_dir: Path,
    fields_mhz=(90.0, 600.0),
    workers: int | None = None,
    points: int = 16384,
    ppm_from: float = 0.0,
    ppm_to: float = 12.0,
    linewidth_hz: float = 1.0,
    fmt: str = "dense",
) -> dict:
    """Simulate every spin-graph in a Task-2 JSON(L) file with the pyspin engine.

    ``fmt`` selects the stored representation: ``"dense"`` (``mol_<i>.npy``) or
    ``"peaks"`` (``mol_<i>.npz`` line list, lineshape applied on load). Output:
    ``<out_dir>/spectra/<field>MHz/mol_<i>.{npy,npz}`` (index = record line) +
    shared ``ppm_axis.npy`` per field + ``index.csv`` (index → molecule id).

    IPC design: graphs are loaded ONCE per worker process via
    ``_init_graph_workers`` (pool initializer). Task tuples carry only an
    integer index — not the graph dict — so the main process is never the
    serialization bottleneck regardless of worker count or dataset size.
    Tasks are generated lazily so the 2 × N_molecules task list is never fully
    materialised in RAM.
    """
    import csv

    from simulation.graph_io import molecule_id, read_spin_systems

    jsonl_path = Path(jsonl_path)
    out_dir = Path(out_dir)
    ext = "npz" if fmt == "peaks" else "npy"
    graphs = list(read_spin_systems(jsonl_path))   # [(idx, record), ...]
    if not graphs:
        raise ValueError(f"No spin systems found in {jsonl_path}")
    workers = workers or os.cpu_count() or 1

    spectra_root = out_dir / "spectra"
    ppm_axis = np.linspace(ppm_from, ppm_to, points)
    for field in fields_mhz:
        fdir = spectra_root / f"{field:.0f}MHz"
        fdir.mkdir(parents=True, exist_ok=True)
        np.save(fdir / "ppm_axis.npy", ppm_axis)

    # id manifest: index → molecule identifier
    spectra_root.mkdir(parents=True, exist_ok=True)
    with (spectra_root / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "id"])
        for idx, graph in graphs:
            w.writerow([f"mol_{idx:06d}", molecule_id(graph, "")])

    n = len(graphs)
    total = n * len(fields_mhz)

    def _task_gen():
        for field in fields_mhz:
            fdir = spectra_root / f"{field:.0f}MHz"
            for idx, _ in graphs:
                yield (idx, float(field), str(fdir / f"mol_{idx:06d}.{ext}"),
                       points, ppm_from, ppm_to, linewidth_hz, fmt)

    print(f"pyspin graph batch: {n} molecules × {len(fields_mhz)} fields "
          f"= {total} sims on {workers} workers")

    if workers == 1:
        _init_graph_workers(str(jsonl_path))
        results = [_graph_worker(t) for t in _progress(_task_gen(), total, "simulating")]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers,
                      initializer=_init_graph_workers,
                      initargs=(str(jsonl_path),)) as pool:
            results = list(_progress(
                pool.imap_unordered(_graph_worker, _task_gen(), chunksize=16),
                total, "simulating"))

    failures = [(o, err) for o, ok, err in results if not ok]
    succeeded = len(results) - len(failures)
    print(f"  done: {succeeded}/{total} succeeded")
    for o, err in failures[:10]:
        print(f"  FAILED {o}: {err}")
    return {"tasks": total, "succeeded": succeeded,
            "failed": len(failures), "failures": failures}
