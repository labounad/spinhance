"""
Tests for simulation.xml_io — no MNova required.

Run from the repo root::

    micromamba activate spinhance
    python -m pytest simulation/tests -v
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.xml_io import (  # noqa: E402
    matrix_to_xml,
    save_xml,
    patch_frequency,
    generate_field_pair,
    xml_to_matrix,
    _labels,
    LOW_FIELD_MHZ,
    HIGH_FIELD_MHZ,
)

EXAMPLE_XML = REPO_ROOT / "simulation" / "examples" / "reference_15group.xml"


# ── Label generation ──────────────────────────────────────────────────────────

def test_labels_8():
    assert _labels(8) == list("ABCDEFGH")


def test_labels_26():
    labs = _labels(26)
    assert len(labs) == 26 and labs[-1] == "Z"


def test_labels_27():
    assert _labels(27)[26] == "AA"


# ── XML construction ──────────────────────────────────────────────────────────

def test_matrix_to_xml_basic():
    tree = matrix_to_xml([3.0, 7.5], [[0.0, 8.0], [8.0, 0.0]], [1, 1],
                         frequency_mhz=LOW_FIELD_MHZ)
    root = tree.getroot()
    assert root.tag == "mnova-spinsim"
    assert len(root.findall(".//group")) == 2
    assert float(root.find(".//spectrum/frequency").text) == LOW_FIELD_MHZ
    assert int(root.find(".//spectrum/points").text) == 16384


def test_group_attributes():
    tree = matrix_to_xml([1.0, 2.0], [[0.0, 5.0], [5.0, 0.0]], [2, 3])
    groups = tree.getroot().findall(".//group")
    assert groups[0].attrib["name"] == "A"
    assert groups[1].attrib["name"] == "B"
    assert groups[0].attrib["number"] == "2"
    assert groups[1].attrib["number"] == "3"


def test_jcoupling_symmetry():
    tree = matrix_to_xml([1.0, 2.0], [[0.0, 7.2], [7.2, 0.0]], [1, 1])
    root = tree.getroot()
    j_ab = root.find(".//group[@name='A']").find("jCoupling[@name='B']")
    j_ba = root.find(".//group[@name='B']").find("jCoupling[@name='A']")
    assert abs(float(j_ab.text) - 7.2) < 1e-5
    assert abs(float(j_ba.text) - 7.2) < 1e-5


def test_xml_to_matrix_roundtrip(tmp_path):
    shifts = [1.0, 2.5, 7.2]
    couplings = [[0.0, 7.1, 0.0], [7.1, 0.0, -12.0], [0.0, -12.0, 0.0]]
    degeneracy = [3, 2, 1]
    tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=90.0)
    out = tmp_path / "rt.xml"
    save_xml(tree, out)

    m = xml_to_matrix(out)
    assert m["shifts"] == pytest.approx(shifts)
    assert m["degeneracy"] == degeneracy
    assert m["frequency_mhz"] == pytest.approx(90.0)
    assert m["points"] == 16384
    for i in range(3):
        for j in range(3):
            assert m["couplings"][i][j] == pytest.approx(couplings[i][j])


def test_save_and_reload(tmp_path):
    tree = matrix_to_xml([3.0, 7.5], [[0.0, 8.0], [8.0, 0.0]], [1, 1])
    out = tmp_path / "test.xml"
    save_xml(tree, out)
    assert out.exists()
    assert ET.parse(str(out)).getroot().tag == "mnova-spinsim"


# ── Frequency patching ────────────────────────────────────────────────────────

@pytest.mark.skipif(not EXAMPLE_XML.exists(), reason="Example XML not present")
def test_patch_frequency():
    patched = patch_frequency(EXAMPLE_XML, LOW_FIELD_MHZ)
    freq = patched.getroot().find(".//spectrum/frequency")
    assert abs(float(freq.text) - LOW_FIELD_MHZ) < 1e-6


@pytest.mark.skipif(not EXAMPLE_XML.exists(), reason="Example XML not present")
def test_generate_field_pair(tmp_path):
    lo, hi = generate_field_pair(EXAMPLE_XML, tmp_path, stem="mol0")
    assert lo.exists() and hi.exists()
    assert lo.name == "mol0_90MHz.xml"
    assert hi.name == "mol0_600MHz.xml"
    lo_freq = ET.parse(str(lo)).getroot().find(".//spectrum/frequency")
    hi_freq = ET.parse(str(hi)).getroot().find(".//spectrum/frequency")
    assert abs(float(lo_freq.text) - LOW_FIELD_MHZ) < 1e-6
    assert abs(float(hi_freq.text) - HIGH_FIELD_MHZ) < 1e-5


# ── Round-trip: 8-group system ────────────────────────────────────────────────

def test_8group_roundtrip(tmp_path):
    import random
    random.seed(42)
    n = 8
    shifts = [random.uniform(0.5, 9.0) for _ in range(n)]
    couplings = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = random.uniform(0, 15.0)
            couplings[i][j] = couplings[j][i] = val
    degeneracy = [random.randint(1, 3) for _ in range(n)]

    tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=LOW_FIELD_MHZ)
    out = tmp_path / "mol_8group.xml"
    save_xml(tree, out)

    groups = ET.parse(str(out)).getroot().findall(".//group")
    assert len(groups) == n
    for i, grp in enumerate(groups):
        assert abs(float(grp.find("shift").text) - shifts[i]) < 1e-5
