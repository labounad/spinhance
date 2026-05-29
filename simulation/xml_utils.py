"""
xml_utils.py
------------
Utilities for creating and patching mnova-spinsim XML files.

A shift+J matrix is represented as a dict with keys:
    shifts      : list of float, length n_groups  (ppm, diagonal)
    couplings   : list of list of float, n x n    (Hz, off-diagonal; symmetric)
    degeneracy  : list of int, length n_groups    (number of protons per group)
    linewidths  : list of float (Hz), optional    (default 1.0 for all groups)

Group labels are auto-assigned: A, B, C, ...
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
import copy
import string
from pathlib import Path


# ── Label helpers ─────────────────────────────────────────────────────────────

def _labels(n: int) -> list[str]:
    """Return n single-letter labels: A, B, C, ..., Z, AA, AB, ..."""
    alpha = string.ascii_uppercase
    if n <= 26:
        return list(alpha[:n])
    labels = list(alpha)
    for a in alpha:
        for b in alpha:
            labels.append(a + b)
            if len(labels) == n:
                return labels
    raise ValueError(f"Cannot generate {n} labels")


# ── XML construction from matrix ──────────────────────────────────────────────

def matrix_to_xml(
    shifts: list[float],
    couplings: list[list[float]],
    degeneracy: list[int],
    frequency_mhz: float = 600.15,
    points: int = 16384,
    ppm_from: float = 0.0,
    ppm_to: float = 12.0,
    linewidths: list[float] | None = None,
) -> ET.ElementTree:
    """
    Build a mnova-spinsim XML ElementTree from spin-system parameters.

    Parameters
    ----------
    shifts       : chemical shifts in ppm (diagonal of J-matrix)
    couplings    : scalar couplings in Hz (off-diagonal); must be symmetric
    degeneracy   : number of protons per spin group
    frequency_mhz: spectrometer frequency (e.g. 90.0 or 600.15)
    points       : number of spectral points (recommend 16384 = 2^14)
    ppm_from/to  : spectral window in ppm
    linewidths   : peak linewidths in Hz per group (default 1.0 Hz each)
    """
    n = len(shifts)
    assert len(couplings) == n and len(couplings[0]) == n, "Coupling matrix must be n×n"
    assert len(degeneracy) == n, "Degeneracy list must have length n"

    labels = _labels(n)
    lw = linewidths if linewidths is not None else [1.0] * n

    root = ET.Element("mnova-spinsim")
    ss = ET.SubElement(root, "spin-system")

    # ── summary block (human-readable matrix) ────────────────────────────────
    header = "\t" + "\t".join(labels)
    rows = [header]
    for i, lab in enumerate(labels):
        row_vals = [f"{shifts[i]:.6f}"] + [
            f"{couplings[i][j]:.6f}" if j < i else ""
            for j in range(n)
        ]
        # fill diagonal and lower triangle only (mirrors MNova convention)
        cells = []
        for j in range(i + 1):
            if i == j:
                cells.append(f"{shifts[i]:.6f}")
            else:
                cells.append(f"{couplings[i][j]:.6f}")
        rows.append(lab + "\t" + "\t".join(cells))

    summary = ET.SubElement(ss, "summary")
    summary.text = "\n\t" + "\n\t".join(rows) + "\n\t"

    ET.SubElement(ss, "population").text = "1"

    # ── per-group blocks ──────────────────────────────────────────────────────
    for i, lab in enumerate(labels):
        grp = ET.SubElement(
            ss, "group",
            name=lab,
            spinByTwo="1",
            lineWidth=str(lw[i]),
            number=str(degeneracy[i]),
        )
        ET.SubElement(grp, "shift").text = f"{shifts[i]:.6f}"
        ET.SubElement(grp, "qConst").text = "0"

        for j, other in enumerate(labels):
            if other == lab:
                continue
            j_val = couplings[i][j] if i != j else 0.0
            ET.SubElement(grp, "jCoupling", name=other).text = f"{j_val:.6f}"
            ET.SubElement(grp, "dCoupling", name=other).text = "0"

    # ── spectrum parameters ───────────────────────────────────────────────────
    spec = ET.SubElement(root, "spectrum")
    ET.SubElement(spec, "frequency").text = str(frequency_mhz)
    ET.SubElement(spec, "points").text = str(points)
    ET.SubElement(spec, "from").text = str(ppm_from)
    ET.SubElement(spec, "to").text = str(ppm_to)

    return ET.ElementTree(root)


def save_xml(tree: ET.ElementTree, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True)


# ── Patch an existing XML to a different field ────────────────────────────────

def patch_frequency(xml_path: str | Path, new_freq_mhz: float) -> ET.ElementTree:
    """
    Load an existing mnova-spinsim XML and change its spectrometer frequency.
    Returns a modified ElementTree (does not overwrite the original).
    """
    tree = ET.parse(str(xml_path))
    root = copy.deepcopy(tree.getroot())
    freq_el = root.find(".//spectrum/frequency")
    if freq_el is None:
        raise ValueError(f"No <frequency> tag found in {xml_path}")
    freq_el.text = str(new_freq_mhz)
    return ET.ElementTree(root)


# ── Generate paired XMLs (90 MHz + 600 MHz) for one molecule ──────────────────

def generate_field_pair(
    source_xml: str | Path,
    output_dir: str | Path,
    stem: str | None = None,
    low_field_mhz: float = 90.0,
    high_field_mhz: float = 600.15,
) -> tuple[Path, Path]:
    """
    From a single source XML, produce two field-specific XMLs.

    Returns
    -------
    (low_field_path, high_field_path)
    """
    source_xml = Path(source_xml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or source_xml.stem

    low_path = output_dir / f"{stem}_90MHz.xml"
    high_path = output_dir / f"{stem}_600MHz.xml"

    save_xml(patch_frequency(source_xml, low_field_mhz), low_path)
    save_xml(patch_frequency(source_xml, high_field_mhz), high_path)

    return low_path, high_path


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    # Trivial 2-spin AX system
    shifts = [3.0, 7.5]
    couplings = [[0.0, 8.0], [8.0, 0.0]]
    degeneracy = [1, 1]

    tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=90.0)
    save_xml(tree, "/tmp/test_ax.xml")
    print("Wrote /tmp/test_ax.xml")

    # Patch existing example
    from pathlib import Path
    repo = Path(__file__).parent.parent
    example = repo / "predicted_mnova_1h (10).xml"
    if example.exists():
        lo, hi = generate_field_pair(example, "/tmp/spinhance_test", stem="example_mol")
        print(f"Low field:  {lo}")
        print(f"High field: {hi}")
