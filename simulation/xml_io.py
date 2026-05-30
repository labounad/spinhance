"""
xml_io.py
=========
Conversion between SpinHance spin-system parameters and the ``mnova-spinsim``
XML format consumed by MestReNova's quantum spin simulator.

This module is **pure** — it has no dependency on MNova or numpy and performs no
orchestration. It only builds, patches, and writes XML trees.

Spin-system representation
---------------------------
A molecule's spin system is described by:

==============  =========================================================
``shifts``      list[float], length ``n`` — chemical shifts δ in ppm
                (the diagonal of the shift+J matrix; field-independent).
``couplings``   list[list[float]], ``n × n`` — scalar couplings *J* in Hz
                (off-diagonal; must be symmetric, zero diagonal).
``degeneracy``  list[int], length ``n`` — protons per spin group
                (e.g. 3 for CH₃, 9 for tert-butyl).
``linewidths``  list[float], length ``n``, optional — Hz, default 1.0 each.
==============  =========================================================

Group labels are auto-assigned A, B, C, … (see :func:`_labels`).

Public API
----------
- :func:`matrix_to_xml`     — build an XML tree from matrix parameters.
- :func:`save_xml`          — write a tree to disk (creates parent dirs).
- :func:`patch_frequency`   — change a tree's spectrometer frequency.
- :func:`generate_field_pair` — emit low- + high-field XMLs from one source.
"""

from __future__ import annotations

import copy
import string
import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = [
    "matrix_to_xml",
    "save_xml",
    "patch_frequency",
    "generate_field_pair",
    "xml_to_matrix",
]

# Default field strengths (MHz). "Low field" is 90 MHz (strongly coupled,
# non-first-order); "high field" 600.15 MHz is the first-order reference.
LOW_FIELD_MHZ = 90.0
HIGH_FIELD_MHZ = 600.15

# Default spectral grid (kept in sync with pipeline.N_POINTS / PPM window).
DEFAULT_POINTS = 16384  # 2**14
DEFAULT_PPM_FROM = 0.0
DEFAULT_PPM_TO = 12.0


# ── Label helpers ─────────────────────────────────────────────────────────────

def _labels(n: int) -> list[str]:
    """Return ``n`` spin-group labels: A, B, …, Z, AA, AB, …"""
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
    frequency_mhz: float = HIGH_FIELD_MHZ,
    points: int = DEFAULT_POINTS,
    ppm_from: float = DEFAULT_PPM_FROM,
    ppm_to: float = DEFAULT_PPM_TO,
    linewidths: list[float] | None = None,
) -> ET.ElementTree:
    """Build a ``mnova-spinsim`` :class:`~xml.etree.ElementTree.ElementTree`.

    Parameters
    ----------
    shifts
        Chemical shifts in ppm (diagonal of the shift+J matrix).
    couplings
        ``n × n`` scalar couplings in Hz (off-diagonal); must be symmetric.
    degeneracy
        Number of protons per spin group.
    frequency_mhz
        Spectrometer frequency (e.g. ``90.0`` or ``600.15``).
    points
        Number of spectral points (recommend ``16384 = 2**14``).
    ppm_from, ppm_to
        Spectral window in ppm.
    linewidths
        Per-group peak linewidths in Hz (default ``1.0`` for each group).

    Returns
    -------
    xml.etree.ElementTree.ElementTree
        A tree rooted at ``<mnova-spinsim>``, ready for :func:`save_xml`.
    """
    n = len(shifts)
    assert len(couplings) == n and len(couplings[0]) == n, "Coupling matrix must be n×n"
    assert len(degeneracy) == n, "Degeneracy list must have length n"

    labels = _labels(n)
    lw = linewidths if linewidths is not None else [1.0] * n

    root = ET.Element("mnova-spinsim")
    ss = ET.SubElement(root, "spin-system")

    # ── summary block (human-readable lower-triangular matrix) ────────────────
    header = "\t" + "\t".join(labels)
    rows = [header]
    for i, lab in enumerate(labels):
        cells = []
        for j in range(i + 1):
            cells.append(f"{shifts[i]:.6f}" if i == j else f"{couplings[i][j]:.6f}")
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


def xml_to_matrix(xml_path: str | Path) -> dict:
    """Parse a ``mnova-spinsim`` XML back into matrix form (inverse of
    :func:`matrix_to_xml`).

    Returns
    -------
    dict with keys:
        ``shifts``      list[float] — δ in ppm, in group order.
        ``couplings``   list[list[float]] — symmetric n×n J in Hz.
        ``degeneracy``  list[int] — protons per group.
        ``labels``      list[str] — group names.
        ``frequency_mhz``, ``points``, ``ppm_from``, ``ppm_to`` — spectrum meta
        (``None`` if absent).
    """
    root = ET.parse(str(xml_path)).getroot()
    groups = root.findall(".//group")
    labels = [g.get("name") for g in groups]
    index = {name: i for i, name in enumerate(labels)}
    n = len(groups)

    shifts = [0.0] * n
    degeneracy = [1] * n
    couplings = [[0.0] * n for _ in range(n)]

    for i, g in enumerate(groups):
        shifts[i] = float(g.find("shift").text)
        degeneracy[i] = int(g.get("number", "1"))
        for jc in g.findall("jCoupling"):
            partner = jc.get("name")
            if partner in index:
                couplings[i][index[partner]] = float(jc.text)

    # Symmetrise (XML stores J in both directions, but be robust to asymmetry).
    for a in range(n):
        for b in range(a + 1, n):
            if couplings[a][b] == 0.0 and couplings[b][a] != 0.0:
                couplings[a][b] = couplings[b][a]
            else:
                couplings[b][a] = couplings[a][b]

    def _meta(tag, cast):
        el = root.find(f".//spectrum/{tag}")
        return cast(el.text) if el is not None else None

    return {
        "shifts": shifts,
        "couplings": couplings,
        "degeneracy": degeneracy,
        "labels": labels,
        "frequency_mhz": _meta("frequency", float),
        "points": _meta("points", int),
        "ppm_from": _meta("from", float),
        "ppm_to": _meta("to", float),
    }


def save_xml(tree: ET.ElementTree, path: str | Path) -> None:
    """Write ``tree`` to ``path`` as UTF-8 XML, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(path), encoding="UTF-8", xml_declaration=True)


# ── Patch an existing XML to a different field ────────────────────────────────

def patch_frequency(xml_path: str | Path, new_freq_mhz: float) -> ET.ElementTree:
    """Return a copy of the XML at ``xml_path`` with its frequency changed.

    The original file is not modified.

    Raises
    ------
    ValueError
        If the XML has no ``<spectrum>/<frequency>`` element.
    """
    tree = ET.parse(str(xml_path))
    root = copy.deepcopy(tree.getroot())
    freq_el = root.find(".//spectrum/frequency")
    if freq_el is None:
        raise ValueError(f"No <frequency> tag found in {xml_path}")
    freq_el.text = str(new_freq_mhz)
    return ET.ElementTree(root)


# ── Generate paired XMLs (low + high field) for one molecule ──────────────────

def generate_field_pair(
    source_xml: str | Path,
    output_dir: str | Path,
    stem: str | None = None,
    low_field_mhz: float = LOW_FIELD_MHZ,
    high_field_mhz: float = HIGH_FIELD_MHZ,
) -> tuple[Path, Path]:
    """From one source XML, write two field-specific XMLs.

    Output files are named ``<stem>_<field>MHz.xml``.

    Returns
    -------
    tuple[Path, Path]
        ``(low_field_path, high_field_path)``.
    """
    source_xml = Path(source_xml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or source_xml.stem

    low_path = output_dir / f"{stem}_{low_field_mhz:.0f}MHz.xml"
    high_path = output_dir / f"{stem}_{high_field_mhz:.0f}MHz.xml"

    save_xml(patch_frequency(source_xml, low_field_mhz), low_path)
    save_xml(patch_frequency(source_xml, high_field_mhz), high_path)

    return low_path, high_path
