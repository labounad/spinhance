"""
graph_io.py
===========
The Task 2 → Task 3 data contract. Task 2 (mol_to_matrix) emits each molecule's
¹H spin system as a labelled graph. The on-disk format is a **single JSON array**
(``mol_to_matrix/data/spin_systems.json``); each element is one molecule:

    {
      "chembl_id": "CHEMBL6622",
      "smiles": "...", "inchikey": "...",
      "labels": ["A", "B", ..., "H"],            # spin-group labels, sorted
      "spin_groups": [[4.59, 1], [4.05, 1], ...], # [shift ppm, #protons], aligned to labels
      "couplings": [["A", "C", 5.7], ...]         # [group_i, group_j, J Hz]; absent ⇒ 0
    }

- ``spin_groups[i]`` describes ``labels[i]``: ``[chemical shift (ppm), degeneracy]``.
- ``couplings`` lists only non-zero inter-group couplings (Hz, sign retained);
  any pair not listed is J = 0.

This module converts a record to the arrays the simulators use (pyspin consumes
it directly; the MNova path goes record → XML).

Schema field names live in the constants below — if Task 2 renames anything,
change it here only.
"""

from __future__ import annotations

import json
from pathlib import Path

import xml.etree.ElementTree as ET

from simulation.xml_io import matrix_to_xml

# ── Schema (single source of truth; matches mol_to_matrix/data/README.md) ─────
KEY_LABELS = "labels"
KEY_GROUPS = "spin_groups"   # list of [shift_ppm, degeneracy], aligned to labels
KEY_COUPLINGS = "couplings"  # list of [label_i, label_j, J_Hz]
ID_KEYS = ("chembl_id", "smiles", "inchikey")  # tried in order for the molecule id

__all__ = [
    "validate_record",
    "record_to_arrays",
    "arrays_to_record",
    "record_to_xml",
    "molecule_id",
    "read_spin_systems",
    "write_spin_systems",
    "spin_systems_to_xml_dir",
]


def molecule_id(record: dict, default: str | None = None) -> str | None:
    """Return the first available molecule identifier (chembl_id/smiles/inchikey)."""
    for k in ID_KEYS:
        if record.get(k):
            return record[k]
    return default


def validate_record(record: dict) -> None:
    """Raise ValueError if a spin-system record is malformed."""
    if KEY_LABELS not in record or KEY_GROUPS not in record:
        raise ValueError(f"record missing '{KEY_LABELS}'/'{KEY_GROUPS}'")
    labels = record[KEY_LABELS]
    groups = record[KEY_GROUPS]
    if not labels:
        raise ValueError("record has no labels")
    if len(labels) != len(groups):
        raise ValueError(f"labels ({len(labels)}) and spin_groups ({len(groups)}) "
                         "length mismatch")
    for lab, g in zip(labels, groups):
        if len(g) < 2:
            raise ValueError(f"spin_group for {lab!r} must be [shift, degeneracy]")
        if int(g[1]) < 1:
            raise ValueError(f"group {lab!r} degeneracy must be ≥ 1")
    label_set = set(labels)
    for edge in record.get(KEY_COUPLINGS, []):
        a, b, _j = edge
        if a not in label_set or b not in label_set:
            raise ValueError(f"coupling {edge!r} references unknown label")
        if a == b:
            raise ValueError(f"self-coupling not allowed: {edge!r}")


def record_to_arrays(record: dict):
    """Convert a spin-system record to ``(labels, shifts, couplings, degeneracy)``.

    Uses the record's own ``labels`` order (index-aligned with ``spin_groups``).
    ``couplings`` is the symmetric n×n matrix in Hz, absent pairs = 0 — exactly
    what pyspin and ``matrix_to_xml`` expect.
    """
    validate_record(record)
    labels = list(record[KEY_LABELS])
    groups = record[KEY_GROUPS]
    index = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)

    shifts = [float(groups[i][0]) for i in range(n)]
    degeneracy = [int(groups[i][1]) for i in range(n)]
    couplings = [[0.0] * n for _ in range(n)]
    for a, b, j in record.get(KEY_COUPLINGS, []):
        i, k = index[a], index[b]
        couplings[i][k] = couplings[k][i] = float(j)

    return labels, shifts, couplings, degeneracy


def arrays_to_record(labels, shifts, couplings, degeneracy, j_threshold: float = 0.0,
                     **ids) -> dict:
    """Inverse of :func:`record_to_arrays` (for tests / fixtures).

    Only couplings with ``abs(J) > j_threshold`` become entries (absent = 0).
    Extra keyword args (e.g. ``chembl_id=``, ``smiles=``) become id fields.
    """
    n = len(labels)
    record = {**{k: v for k, v in ids.items() if v is not None},
              KEY_LABELS: list(labels),
              KEY_GROUPS: [[float(shifts[i]), int(degeneracy[i])] for i in range(n)]}
    edges = []
    for i in range(n):
        for k in range(i + 1, n):
            if abs(couplings[i][k]) > j_threshold:
                edges.append([labels[i], labels[k], float(couplings[i][k])])
    record[KEY_COUPLINGS] = edges
    return record


def record_to_xml(record: dict, frequency_mhz: float = 90.0, **kwargs) -> ET.ElementTree:
    """Build a ``mnova-spinsim`` XML tree from a record (for the MNova path)."""
    _labels, shifts, couplings, degeneracy = record_to_arrays(record)
    return matrix_to_xml(shifts, couplings, degeneracy,
                         frequency_mhz=frequency_mhz, **kwargs)


# ── I/O: JSON array (Task 2's format), tolerant of JSONL ──────────────────────

def read_spin_systems(path: str | Path):
    """Yield ``(index, record)`` for each molecule.

    Accepts Task 2's single JSON array, and also tolerates JSONL (one object
    per line) so either layout works.
    """
    path = Path(path)
    text = path.read_text().lstrip()
    if text.startswith("["):
        for i, rec in enumerate(json.loads(text)):
            yield i, rec
    else:
        for i, line in enumerate(l for l in text.splitlines() if l.strip()):
            yield i, json.loads(line)


def write_spin_systems(path: str | Path, records) -> int:
    """Write records as a single JSON array (Task 2's format). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    path.write_text("[\n" + ",\n".join(json.dumps(r) for r in records) + "\n]\n")
    return len(records)


def spin_systems_to_xml_dir(json_path: str | Path, xml_dir: str | Path,
                            frequency_mhz: float = 90.0) -> int:
    """Materialise each record as ``mol_<i>.xml`` for the MNova path.

    Files are named by record index so output spectra line up with the molecule
    id manifest (``index.csv``). The pipeline patches the frequency per field,
    so ``frequency_mhz`` here is just a placeholder. Returns the count written.
    """
    import csv

    from simulation.xml_io import save_xml

    xml_dir = Path(xml_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for idx, rec in read_spin_systems(json_path):
        stem = f"mol_{idx:06d}"
        save_xml(record_to_xml(rec, frequency_mhz=frequency_mhz), xml_dir / f"{stem}.xml")
        rows.append([stem, molecule_id(rec, "")])
        n += 1
    with (xml_dir / "index.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "id"])
        w.writerows(rows)
    return n
