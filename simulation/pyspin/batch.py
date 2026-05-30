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

__all__ = ["simulate_xml_to_npy", "run_pyspin_batch", "run_pyspin_batch_graphs"]


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
        results = [_worker(t) for t in tasks]
    else:
        # 'spawn' children re-import this module → inherit the 1-thread BLAS env.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks, chunksize=4)

    failures = [(o, err) for o, ok, err in results if not ok]
    succeeded = len(results) - len(failures)
    print(f"  done: {succeeded}/{len(tasks)} succeeded")
    for o, err in failures[:10]:
        print(f"  FAILED {o}: {err}")
    return {"tasks": len(tasks), "succeeded": succeeded,
            "failed": len(failures), "failures": failures}


# ── Graph input (Task 2 JSONL → npy), pyspin engine ──────────────────────────

def _graph_worker(task):
    """Top-level (picklable) worker: one (graph, field) -> npy task."""
    graph, field, out_npy, points, ppm_from, ppm_to, linewidth = task
    try:
        from simulation.graph_io import graph_to_arrays
        _labels, shifts, couplings, degeneracy = graph_to_arrays(graph)
        _, spec = simulate_spectrum_pyspin(
            shifts, couplings, degeneracy, field,
            points=points, ppm_from=ppm_from, ppm_to=ppm_to, linewidth_hz=linewidth)
        Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, spec.astype(np.float32))
        return (out_npy, True, "")
    except Exception as e:  # noqa: BLE001
        return (out_npy, False, repr(e))


def run_pyspin_batch_graphs(
    jsonl_path: Path,
    out_dir: Path,
    fields_mhz=(90.0, 600.0),
    workers: int | None = None,
    points: int = 16384,
    ppm_from: float = 0.0,
    ppm_to: float = 12.0,
    linewidth_hz: float = 1.0,
) -> dict:
    """Simulate every spin-graph in a Task-2 JSONL file with the pyspin engine.

    Output: ``<out_dir>/spectra/<field>MHz/mol_<i>.npy`` (index = JSONL line) plus
    a shared ``ppm_axis.npy`` per field and an ``index.csv`` mapping the spectrum
    index to the molecule id (SMILES). Parallel across (molecule, field) tasks.
    """
    import csv

    from simulation.graph_io import molecule_id, read_graphs_jsonl

    out_dir = Path(out_dir)
    graphs = list(read_graphs_jsonl(jsonl_path))   # [(line_idx, graph), ...]
    if not graphs:
        raise ValueError(f"No graphs found in {jsonl_path}")
    workers = workers or os.cpu_count() or 1

    spectra_root = out_dir / "spectra"
    ppm_axis = np.linspace(ppm_from, ppm_to, points)
    tasks = []
    for field in fields_mhz:
        fdir = spectra_root / f"{field:.0f}MHz"
        fdir.mkdir(parents=True, exist_ok=True)
        np.save(fdir / "ppm_axis.npy", ppm_axis)
        for idx, graph in graphs:
            tasks.append((graph, float(field), str(fdir / f"mol_{idx:06d}.npy"),
                          points, ppm_from, ppm_to, linewidth_hz))

    # id manifest: index → molecule identifier
    spectra_root.mkdir(parents=True, exist_ok=True)
    with (spectra_root / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "id"])
        for idx, graph in graphs:
            w.writerow([f"mol_{idx:06d}", molecule_id(graph, "")])

    print(f"pyspin graph batch: {len(graphs)} molecules × {len(fields_mhz)} fields "
          f"= {len(tasks)} sims on {workers} workers")

    if workers == 1:
        results = [_graph_worker(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_graph_worker, tasks, chunksize=4)

    failures = [(o, err) for o, ok, err in results if not ok]
    succeeded = len(results) - len(failures)
    print(f"  done: {succeeded}/{len(tasks)} succeeded")
    for o, err in failures[:10]:
        print(f"  FAILED {o}: {err}")
    return {"tasks": len(tasks), "succeeded": succeeded,
            "failed": len(failures), "failures": failures}
