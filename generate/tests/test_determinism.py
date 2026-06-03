"""Tests for the audited determinism fixes (PIPELINE_AUDIT_2 §5-C).

Covered cases
-------------
- C4: a generated XYZ block carries a non-empty ``inchi`` (was always "").
- C5: ``run_pipeline`` emits records in a stable order — the same input
  produces an identical CSV/XYZ row order across two runs, regardless of
  worker-completion order.
- C5: ``dedup_dataset`` keeps a deterministic representative independent of
  input row order.

All tests use RDKit + stdlib only.  No network, no MNova, no AWS.
"""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from rdkit import Chem

from generate.spin_equivalence import classify_spin_groups
from generate.xyz_writer import build_xyz_block
from generate.dedup import dedup_dataset
from generate.pipeline import run_pipeline


# ── C4: InChI is populated ──────────────────────────────────────────────────────

def test_xyz_block_has_nonempty_inchi():
    """A rendered XYZ block's JSON comment must carry a non-empty InChI."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert mol is not None
    mol_h, groups = classify_spin_groups(mol)

    block = build_xyz_block(
        mol_h, groups, smiles="CC(=O)Oc1ccccc1C(=O)O", chembl_id="t1",
    )
    assert block is not None

    comment = block.splitlines()[1]
    meta = json.loads(comment)
    assert meta["inchi"].startswith("InChI="), meta["inchi"]


# ── C5: deterministic pipeline output ordering ──────────────────────────────────

# A small "smiles" source: a handful of drug-like molecules in a scrambled
# order.  The pipeline must emit them in a stable (sorted-by-key) order, not in
# worker-completion order.
# Format is "<smiles> <id>" (see generate.sources._iter_smiles).  Ids are in a
# scrambled order so a naive completion-order write would not be sorted.
_SMILES_INPUT = """\
CCO m3
c1ccccc1C m1
CC(C)O m4
CC(=O)Oc1ccccc1C(=O)O m2
CCN(CC)CC m5
"""


def _write_smiles_source(tmp_path: Path) -> Path:
    src = tmp_path / "input.smi"
    src.write_text(_SMILES_INPUT)
    return src


def _read_csv_rows(path: Path) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def _read_xyz_comments(path: Path) -> list[str]:
    ids: list[str] = []
    with gzip.open(path, "rt") as fin:
        while True:
            head = fin.readline()
            if not head:
                break
            na = int(head)
            comment = fin.readline()
            for _ in range(na):
                fin.readline()
            ids.append(json.loads(comment)["chembl_id"])
    return ids


def test_run_pipeline_is_order_stable(tmp_path):
    """Two runs over the same input yield byte-identical CSV and XYZ order."""
    src = _write_smiles_source(tmp_path)

    def _run(tag: str) -> tuple[list[list[str]], list[str]]:
        out_csv = tmp_path / f"out_{tag}.csv"
        out_xyz = tmp_path / f"out_{tag}.xyz.gz"
        # Wide group range so every parseable molecule is kept; multiple
        # workers + small chunks maximise the chance completion order varies.
        run_pipeline(
            src, out_csv,
            source="smiles",
            xyz_path=out_xyz,
            min_groups=1, max_groups=26,
            workers=4, chunk_size=1,
            verbose=False,
        )
        return _read_csv_rows(out_csv), _read_xyz_comments(out_xyz)

    rows_a, xyz_a = _run("a")
    rows_b, xyz_b = _run("b")

    assert rows_a == rows_b, "CSV row order differs between identical runs"
    assert xyz_a == xyz_b, "XYZ block order differs between identical runs"

    # Order must be the stable sort key (inchikey, smiles, id), not input order.
    data_rows = rows_a[1:]  # drop header
    inchikeys = [r[2] for r in data_rows]
    assert inchikeys == sorted(inchikeys)

    # XYZ ids follow the same ordering as the kept CSV rows.
    csv_ids = [r[0] for r in data_rows]
    # Every XYZ id appears in the CSV (XYZ may drop embed-failures) in order.
    assert xyz_a == [i for i in csv_ids if i in set(xyz_a)]


# ── C5: deterministic dedup representative ──────────────────────────────────────

def test_dedup_representative_is_order_independent(tmp_path):
    """The surviving row for a duplicated key must not depend on input order."""
    header = ["chembl_id", "smiles", "inchikey", "n_groups", "group_sizes"]
    # Three rows share one InChIKey ("KEY1"); one distinct key ("KEY2").
    rows_forward = [
        ["idC", "CCO", "KEY1", "2", "3;3"],
        ["idA", "CCO", "KEY1", "2", "3;3"],
        ["idB", "CCO", "KEY1", "2", "3;3"],
        ["idZ", "CCN", "KEY2", "2", "3;2"],
    ]
    rows_reversed = list(reversed(rows_forward))

    def _run(rows, tag):
        in_csv = tmp_path / f"in_{tag}.csv"
        out_csv = tmp_path / f"out_{tag}.csv"
        with open(in_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        dedup_dataset(in_csv, out_csv)
        return _read_csv_rows(out_csv)

    out_fwd = _run(rows_forward, "fwd")
    out_rev = _run(rows_reversed, "rev")

    assert out_fwd == out_rev, "dedup output depends on input order"
    # Survivor for KEY1 is the smallest row tuple → id 'idA'.
    key1_rows = [r for r in out_fwd[1:] if r[2] == "KEY1"]
    assert len(key1_rows) == 1
    assert key1_rows[0][0] == "idA"
