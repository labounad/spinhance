"""
Tests for graph_io — the Task 2 → Task 3 spin-system contract. No MNova required.
Validated against the real mol_to_matrix/data/spin_systems.json sample.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.graph_io import (  # noqa: E402
    validate_record,
    record_to_arrays,
    arrays_to_record,
    record_to_xml,
    molecule_id,
    read_spin_systems,
    write_spin_systems,
)
from simulation.xml_io import xml_to_matrix, save_xml  # noqa: E402
from simulation.pyspin.composite import simulate_spectrum_composite  # noqa: E402

SAMPLE = REPO_ROOT / "mol_to_matrix" / "data" / "spin_systems.json"


def _sample_ready() -> bool:
    """True only if the sample exists AND is real data (not a git-LFS pointer)."""
    if not SAMPLE.exists():
        return False
    try:
        return not SAMPLE.read_text(errors="ignore").lstrip().startswith(
            "version https://git-lfs")
    except Exception:
        return False


SAMPLE_READY = _sample_ready()


def _record():
    return {
        "chembl_id": "CHEMBL_X",
        "smiles": "CC(C)=O",
        "labels": ["A", "B", "C"],
        "spin_groups": [[1.06, 3], [2.02, 1], [7.20, 2]],
        "couplings": [["A", "B", 6.6], ["B", "C", 7.8]],
    }


def test_record_to_arrays_basic():
    labels, shifts, couplings, deg = record_to_arrays(_record())
    assert labels == ["A", "B", "C"]
    assert shifts == [1.06, 2.02, 7.20]
    assert deg == [3, 1, 2]
    assert couplings[0][1] == 6.6 and couplings[1][0] == 6.6
    assert couplings[1][2] == 7.8 and couplings[2][1] == 7.8
    assert couplings[0][2] == 0.0  # absent A-C => 0


def test_uses_record_label_order():
    # labels are index-aligned with spin_groups; order must be preserved (not sorted)
    rec = {"labels": ["B", "A"], "spin_groups": [[2.0, 1], [1.0, 3]],
           "couplings": [["A", "B", 5.0]]}
    labels, shifts, couplings, deg = record_to_arrays(rec)
    assert labels == ["B", "A"] and shifts == [2.0, 1.0] and deg == [1, 3]
    assert couplings[0][1] == 5.0


def test_molecule_id_prefers_chembl():
    assert molecule_id(_record()) == "CHEMBL_X"
    assert molecule_id({"smiles": "CCO"}) == "CCO"
    assert molecule_id({}) is None


def test_validate_rejects_bad_records():
    with pytest.raises(ValueError):
        validate_record({"labels": ["A"], "spin_groups": []})          # length mismatch
    with pytest.raises(ValueError):
        validate_record({"labels": ["A"], "spin_groups": [[1.0, 0]]})  # degeneracy < 1
    with pytest.raises(ValueError):
        validate_record({"labels": ["A"], "spin_groups": [[1.0, 1]],
                         "couplings": [["A", "Z", 5.0]]})               # unknown label


def test_roundtrip_arrays_record():
    labels = ["A", "B", "C"]
    shifts = [1.0, 2.5, 7.2]
    couplings = [[0, 7.1, 0], [7.1, 0, 0], [0, 0, 0]]
    deg = [3, 2, 1]
    rec = arrays_to_record(labels, shifts, couplings, deg, chembl_id="X")
    l2, s2, c2, d2 = record_to_arrays(rec)
    assert l2 == labels and s2 == shifts and d2 == deg
    assert c2[0][1] == 7.1 and len(rec["couplings"]) == 1
    assert molecule_id(rec) == "X"


def test_jsonarray_roundtrip(tmp_path):
    recs = [_record(),
            arrays_to_record(["A", "B"], [1.0, 4.0], [[0, 7], [7, 0]], [3, 2],
                             smiles="Y")]
    p = tmp_path / "data.json"
    assert write_spin_systems(p, recs) == 2
    loaded = list(read_spin_systems(p))
    assert [i for i, _ in loaded] == [0, 1]
    assert molecule_id(loaded[1][1]) == "Y"


# ── Against the real Task 2 sample ────────────────────────────────────────────

@pytest.mark.skipif(not SAMPLE_READY, reason="spin_systems.json absent or unresolved git-LFS pointer")
def test_real_sample_parses_and_simulates():
    recs = list(read_spin_systems(SAMPLE))
    assert len(recs) >= 1
    for idx, rec in recs:
        labels, shifts, couplings, deg = record_to_arrays(rec)
        assert len(labels) == len(shifts) == len(deg)
        # every record simulates to a unit-integral spectrum
        _, sp = simulate_spectrum_composite(shifts, couplings, deg, 90.0)
        assert abs(sp.sum() * (12 / len(sp)) - 1.0) < 1e-6


@pytest.mark.skipif(not SAMPLE_READY, reason="spin_systems.json absent or unresolved git-LFS pointer")
def test_real_sample_record_xml_equivalence(tmp_path):
    # record → XML → parse must reproduce the record's arrays (MNova path parity)
    _, rec = next(read_spin_systems(SAMPLE))
    labels, shifts, couplings, deg = record_to_arrays(rec)
    p = tmp_path / "m.xml"
    save_xml(record_to_xml(rec, frequency_mhz=90.0), p)
    m = xml_to_matrix(p)
    assert m["shifts"] == pytest.approx(shifts)
    assert m["degeneracy"] == deg
    for i in range(len(labels)):
        for j in range(len(labels)):
            assert m["couplings"][i][j] == pytest.approx(couplings[i][j])
