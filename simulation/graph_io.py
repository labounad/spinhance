"""
graph_io.py
===========
The Task 2 → Task 3 data contract. Task 2 (mol_to_matrix) emits each molecule's
spin system as a **labelled graph**, not a dense matrix:

- **Nodes** are spin groups labelled ``"A"``, ``"B"``, … with two attributes:
  ``sigma`` (chemical shift, ppm) and ``degeneracy`` (number of protons).
- **Edges** are coupling constants in Hz between node pairs. Absent edges mean
  *no significant coupling* (J = 0).
- A molecule identifier (e.g. ``smiles``) rides along.

On-disk format: **JSONL** — one molecule-graph JSON object per line — so a whole
dataset is one file. Example line::

    {"smiles": "...", "nodes": {"A": {"sigma": 1.06, "degeneracy": 3},
     "B": {"sigma": 2.02, "degeneracy": 1}}, "edges": [["A", "B", 6.6]]}

This module converts graphs to the arrays the simulators use (pyspin consumes
the graph essentially directly; the MNova path goes graph → XML).

⚠ FIELD NAMES ARE PROVISIONAL. Task 2's naming isn't finalised. All keys live in
the constants below — when the contract settles, change them here only.
"""

from __future__ import annotations

import json
from pathlib import Path

import xml.etree.ElementTree as ET

from simulation.xml_io import matrix_to_xml

# ── Schema (single source of truth; adapt here if Task 2 renames fields) ──────
KEY_NODES = "nodes"
KEY_EDGES = "edges"
KEY_SHIFT = "sigma"        # node attribute: chemical shift in ppm
KEY_DEGEN = "degeneracy"   # node attribute: protons in the group
KEY_ID = "smiles"          # molecule identifier (optional)

__all__ = [
    "validate_graph",
    "graph_to_arrays",
    "arrays_to_graph",
    "graph_to_xml",
    "molecule_id",
    "read_graphs_jsonl",
    "write_graphs_jsonl",
    "graphs_jsonl_to_xml_dir",
]


def molecule_id(graph: dict, default: str | None = None) -> str | None:
    """Return the molecule identifier (SMILES) if present."""
    return graph.get(KEY_ID, default)


def validate_graph(graph: dict) -> None:
    """Raise ValueError if the graph is malformed.

    Checks node attributes exist, degeneracy ≥ 1, and every edge references
    existing nodes with a numeric coupling.
    """
    if KEY_NODES not in graph:
        raise ValueError(f"graph missing '{KEY_NODES}'")
    nodes = graph[KEY_NODES]
    if not nodes:
        raise ValueError("graph has no nodes")
    for label, attr in nodes.items():
        if KEY_SHIFT not in attr or KEY_DEGEN not in attr:
            raise ValueError(f"node {label!r} missing '{KEY_SHIFT}'/'{KEY_DEGEN}'")
        if int(attr[KEY_DEGEN]) < 1:
            raise ValueError(f"node {label!r} degeneracy must be ≥ 1")
    for edge in graph.get(KEY_EDGES, []):
        a, b, _j = edge
        if a not in nodes or b not in nodes:
            raise ValueError(f"edge {edge!r} references unknown node")
        if a == b:
            raise ValueError(f"self-edge not allowed: {edge!r}")


def graph_to_arrays(graph: dict):
    """Convert a spin-graph to ``(labels, shifts, couplings, degeneracy)``.

    ``labels`` are the node keys in sorted order; ``shifts``/``degeneracy`` are
    lists in that order; ``couplings`` is the symmetric n×n matrix (Hz) with
    absent edges = 0. This is exactly what pyspin and ``matrix_to_xml`` expect.
    """
    validate_graph(graph)
    nodes = graph[KEY_NODES]
    labels = sorted(nodes.keys())
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)

    shifts = [float(nodes[lab][KEY_SHIFT]) for lab in labels]
    degeneracy = [int(nodes[lab][KEY_DEGEN]) for lab in labels]
    couplings = [[0.0] * n for _ in range(n)]
    for a, b, j in graph.get(KEY_EDGES, []):
        i, k = index[a], index[b]
        couplings[i][k] = couplings[k][i] = float(j)

    return labels, shifts, couplings, degeneracy


def arrays_to_graph(labels, shifts, couplings, degeneracy, smiles=None,
                    j_threshold: float = 0.0) -> dict:
    """Inverse of :func:`graph_to_arrays` (for tests / generating examples).

    Only couplings with ``abs(J) > j_threshold`` become edges (absent = 0).
    """
    nodes = {labels[i]: {KEY_SHIFT: float(shifts[i]), KEY_DEGEN: int(degeneracy[i])}
             for i in range(len(labels))}
    edges = []
    n = len(labels)
    for i in range(n):
        for k in range(i + 1, n):
            if abs(couplings[i][k]) > j_threshold:
                edges.append([labels[i], labels[k], float(couplings[i][k])])
    graph = {KEY_NODES: nodes, KEY_EDGES: edges}
    if smiles is not None:
        graph[KEY_ID] = smiles
    return graph


def graph_to_xml(graph: dict, frequency_mhz: float = 90.0, **kwargs) -> ET.ElementTree:
    """Build a ``mnova-spinsim`` XML tree from a spin-graph (for the MNova path)."""
    _labels, shifts, couplings, degeneracy = graph_to_arrays(graph)
    return matrix_to_xml(shifts, couplings, degeneracy,
                         frequency_mhz=frequency_mhz, **kwargs)


# ── JSONL I/O ─────────────────────────────────────────────────────────────────

def read_graphs_jsonl(path: str | Path):
    """Yield ``(line_index, graph_dict)`` for each non-blank line of a JSONL file."""
    path = Path(path)
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                yield i, json.loads(line)


def write_graphs_jsonl(path: str | Path, graphs) -> int:
    """Write an iterable of graph dicts to JSONL. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for g in graphs:
            f.write(json.dumps(g) + "\n")
            n += 1
    return n


def graphs_jsonl_to_xml_dir(jsonl_path: str | Path, xml_dir: str | Path,
                            frequency_mhz: float = 90.0) -> int:
    """Materialise each graph in a JSONL as ``mol_<i>.xml`` for the MNova path.

    Files are named by JSONL line index (``mol_000000.xml``) so output spectra
    line up with the molecule id manifest. The pipeline patches the frequency
    per field, so the ``frequency_mhz`` written here is only a placeholder.
    Writes an ``index.csv`` (spectrum stem → molecule id) alongside. Returns the
    number of XMLs written.
    """
    import csv

    from simulation.xml_io import save_xml

    xml_dir = Path(xml_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for idx, graph in read_graphs_jsonl(jsonl_path):
        stem = f"mol_{idx:06d}"
        save_xml(graph_to_xml(graph, frequency_mhz=frequency_mhz), xml_dir / f"{stem}.xml")
        rows.append([stem, molecule_id(graph, "")])
        n += 1
    with (xml_dir / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "id"])
        w.writerows(rows)
    return n
