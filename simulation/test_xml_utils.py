"""
test_xml_utils.py
-----------------
Smoke tests for xml_utils.py — no MNova required.

Run from repo root:
    micromamba activate spinhance
    python -m pytest simulation/test_xml_utils.py -v
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile

import pytest

# Allow running from repo root without install
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.xml_utils import (
    matrix_to_xml,
    save_xml,
    patch_frequency,
    generate_field_pair,
    _labels,
)

EXAMPLE_XML = Path(__file__).parent.parent / "predicted_mnova_1h (10).xml"


# ── Label generation ──────────────────────────────────────────────────────────

def test_labels_8():
    labs = _labels(8)
    assert labs == list("ABCDEFGH")

def test_labels_26():
    labs = _labels(26)
    assert len(labs) == 26
    assert labs[-1] == "Z"

def test_labels_27():
    labs = _labels(27)
    assert labs[26] == "AA"


# ── XML construction ──────────────────────────────────────────────────────────

def test_matrix_to_xml_basic():
    shifts     = [3.0, 7.5]
    couplings  = [[0.0, 8.0], [8.0, 0.0]]
    degeneracy = [1, 1]

    tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=90.0)
    root = tree.getroot()

    assert root.tag == "mnova-spinsim"
    groups = root.findall(".//group")
    assert len(groups) == 2

    freq = root.find(".//spectrum/frequency")
    assert float(freq.text) == 90.0

    pts = root.find(".//spectrum/points")
    assert int(pts.text) == 16384


def test_group_attributes():
    tree = matrix_to_xml([1.0, 2.0], [[0.0, 5.0], [5.0, 0.0]], [2, 3])
    root = tree.getroot()
    groups = root.findall(".//group")

    assert groups[0].attrib["name"] == "A"
    assert groups[1].attrib["name"] == "B"
    assert groups[0].attrib["number"] == "2"
    assert groups[1].attrib["number"] == "3"


def test_jcoupling_symmetry():
    """J(A,B) and J(B,A) should be equal."""
    couplings = [[0.0, 7.2], [7.2, 0.0]]
    tree = matrix_to_xml([1.0, 2.0], couplings, [1, 1])
    root = tree.getroot()

    grp_a = root.find(".//group[@name='A']")
    grp_b = root.find(".//group[@name='B']")

    j_ab = grp_a.find("jCoupling[@name='B']")
    j_ba = grp_b.find("jCoupling[@name='A']")

    assert abs(float(j_ab.text) - 7.2) < 1e-5
    assert abs(float(j_ba.text) - 7.2) < 1e-5


def test_save_and_reload(tmp_path):
    tree = matrix_to_xml([3.0, 7.5], [[0.0, 8.0], [8.0, 0.0]], [1, 1])
    out = tmp_path / "test.xml"
    save_xml(tree, out)
    assert out.exists()
    reloaded = ET.parse(str(out))
    assert reloaded.getroot().tag == "mnova-spinsim"


# ── Frequency patching ────────────────────────────────────────────────────────

@pytest.mark.skipif(not EXAMPLE_XML.exists(), reason="Example XML not present")
def test_patch_frequency():
    patched = patch_frequency(EXAMPLE_XML, 90.0)
    freq = patched.getroot().find(".//spectrum/frequency")
    assert abs(float(freq.text) - 90.0) < 1e-6


@pytest.mark.skipif(not EXAMPLE_XML.exists(), reason="Example XML not present")
def test_generate_field_pair(tmp_path):
    lo, hi = generate_field_pair(EXAMPLE_XML, tmp_path, stem="mol0")
    assert lo.exists()
    assert hi.exists()

    lo_freq = ET.parse(str(lo)).getroot().find(".//spectrum/frequency")
    hi_freq = ET.parse(str(hi)).getroot().find(".//spectrum/frequency")
    assert abs(float(lo_freq.text) - 90.0) < 1e-6
    assert abs(float(hi_freq.text) - 600.15) < 1e-5


# ── Round-trip: 8-group system ────────────────────────────────────────────────

def test_8group_roundtrip(tmp_path):
    import random
    random.seed(42)
    n = 8
    shifts = [random.uniform(0.5, 9.0) for _ in range(n)]
    couplings = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            j_val = random.uniform(0, 15.0)
            couplings[i][j] = j_val
            couplings[j][i] = j_val
    degeneracy = [random.randint(1, 3) for _ in range(n)]

    tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=90.0)
    out = tmp_path / "mol_8group.xml"
    save_xml(tree, out)

    root = ET.parse(str(out)).getroot()
    groups = root.findall(".//group")
    assert len(groups) == n

    # Check all shift values
    for i, grp in enumerate(groups):
        shift_el = grp.find("shift")
        assert abs(float(shift_el.text) - shifts[i]) < 1e-5
