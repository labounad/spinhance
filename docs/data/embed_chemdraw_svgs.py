"""
Embed ChemDraw SVGs into spin_viewer.json with accurate atom positions.

Atom positions are derived by:
  1. Parsing <text> elements from the ChemDraw SVG — each has a matrix transform
     whose (tx, ty) values are the exact SVG-pixel position of that atom's label.
  2. Matching those SVG positions to RDKit mol-file atoms of the same element type
     to calibrate the mol→SVG affine transform (scale + offset, Y-flipped).
  3. Applying that calibrated transform to ALL heavy atom positions.
  4. Falling back to bounding-box scaling when no heteroatoms are available.

Run from repo root:
    python docs/data/embed_chemdraw_svgs.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

sys.path.insert(0, ".")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import numpy as np
RDLogger.DisableLog("rdApp.*")

SVG_DIR = Path("docs/data/cd_svgs")
JSON_IN = Path("docs/data/spin_viewer.json")

# ── SVG post-processing ───────────────────────────────────────────────────────

def whiten_svg(svg: str) -> str:
    svg = re.sub(r"<\?xml[^?]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    svg = svg.replace('fill="#000000"', 'fill="#ffffff"')
    svg = svg.replace("fill='#000000'", "fill='#ffffff'")
    svg = svg.replace('stroke="#000000"', 'stroke="#ffffff"')
    svg = svg.replace("stroke='#000000'", "stroke='#ffffff'")
    # Strip ChemDraw enhanced-stereo group labels (&1, &2, or1, or2, abs)
    # These appear as floating text annotations not useful for display
    svg = re.sub(r'<text[^>]*>[^<]*&amp;\d[^<]*</text>', '', svg)
    svg = re.sub(r'<text[^>]*>\s*(?:or|abs)\d*\s*</text>', '', svg)
    # Compact whitespace
    svg = re.sub(r"\n\s*", " ", svg)
    svg = re.sub(r"  +", " ", svg)
    return svg.strip()

# ── Parse ChemDraw SVG text-element positions ─────────────────────────────────
#
# Each labelled atom is a <text x="0" y="0" ... transform="matrix(s 0 0 s tx ty)">LABEL</text>
# The (tx, ty) is the atom's position in SVG pixel coords.

_TEXT_RE = re.compile(
    r'<text[^>]+transform=["\']matrix\([0-9e.Ee+-]+ [0-9e.Ee+-]+ [0-9e.Ee+-]+ [0-9e.Ee+-]+ '
    r'([0-9Ee.+-]+) ([0-9Ee.+-]+)\)["\'][^>]*>\s*([A-Za-z0-9]+)'
)
_VB_RE = re.compile(r'viewBox=["\'](\d+) (\d+) (\d+) (\d+)["\']')


_VALID_ELEMS = {
    "C","N","O","S","P","F","Cl","Br","I","Si","B","Se","Te","As","Sb","Bi"
}

def _parse_svg_labels(svg: str) -> list[tuple[str, float, float]]:
    """Return [(element, svg_x, svg_y), ...] for every atom text label in the SVG."""
    results = []
    for tx, ty, label in _TEXT_RE.findall(svg):
        m = re.match(r'([A-Z][a-z]?)', label)
        if not m:
            continue
        elem = m.group(1)
        if elem not in _VALID_ELEMS:
            continue
        results.append((elem, float(tx), float(ty)))
    return results


def _svg_dims(svg: str) -> tuple[float, float]:
    m = _VB_RE.search(svg)
    if m:
        return float(m.group(3)), float(m.group(4))
    # Fallback: parse width/height attributes
    wm = re.search(r'width=["\']([0-9.]+)px["\']', svg)
    hm = re.search(r'height=["\']([0-9.]+)px["\']', svg)
    return (float(wm.group(1)) if wm else 200.0,
            float(hm.group(1)) if hm else 150.0)


# ── Calibrated atom-position computation ─────────────────────────────────────

def _fit_transform(mol_pts: np.ndarray, svg_pts: np.ndarray
                   ) -> tuple[float, float, float] | None:
    """
    Fit: svg_x = mol_x * s + ox,  svg_y = -mol_y * s + oy
    using least squares over matched (mol_pts, svg_pts) pairs.
    Returns (s, ox, oy) or None if fewer than 2 pairs.
    """
    if len(mol_pts) < 2:
        return None
    # Build linear system for s, ox, oy
    # svg_x = s*mol_x + ox  →  [mol_x, 1, 0][s, ox, oy]^T = svg_x
    # svg_y = -s*mol_y + oy →  [-mol_y, 0, 1][s, ox, oy]^T = svg_y
    A = []
    b = []
    for (mx, my), (sx, sy) in zip(mol_pts, svg_pts):
        A.append([mx,  1.0, 0.0])
        b.append(sx)
        A.append([-my, 0.0, 1.0])
        b.append(sy)
    A = np.array(A); b = np.array(b)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(result[0]), float(result[1]), float(result[2])  # s, ox, oy


def atom_positions(smiles: str, svg: str, heavy_atom_indices: list[int]
                   ) -> list[list[float]]:
    """
    Compute normalised [0-1] (x, y) positions for the requested heavy atom indices.
    Uses ChemDraw SVG text elements for calibration; falls back to bbox scaling.
    """
    W, H = _svg_dims(svg)
    svg_labels = _parse_svg_labels(svg)    # [(elem, tx, ty), ...]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [[0.5, 0.5]] * len(heavy_atom_indices)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    def mol_xy(i):
        p = conf.GetAtomPosition(i)
        return p.x, p.y

    # ── Build calibration pairs ───────────────────────────────────────────────
    # Group mol atoms by element symbol
    mol_by_elem: dict[str, list[tuple[int, float, float]]] = {}
    for i in range(mol.GetNumAtoms()):
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        mol_by_elem.setdefault(sym, []).append((i, *mol_xy(i)))

    # Group SVG labels by element
    svg_by_elem: dict[str, list[tuple[float, float]]] = {}
    for elem, tx, ty in svg_labels:
        svg_by_elem.setdefault(elem, []).append((tx, ty))

    mol_calib: list[tuple[float, float]] = []
    svg_calib: list[tuple[float, float]] = []

    for elem, mol_list in mol_by_elem.items():
        if elem in ("C", "H"):
            continue
        svg_list = svg_by_elem.get(elem, [])
        if not svg_list:
            continue
        if len(mol_list) == 1 and len(svg_list) == 1:
            # Unambiguous single-atom match
            _, mx, my = mol_list[0]
            mol_calib.append((mx, my))
            svg_calib.append(svg_list[0])
        elif len(mol_list) == len(svg_list):
            # Same count: match by sorting on angle from respective centroids
            mc = np.mean([[x, y] for _, x, y in mol_list], axis=0)
            sc = np.mean(svg_list, axis=0)
            def angle_m(tup): return np.arctan2(tup[2]-mc[1], tup[1]-mc[0])
            # SVG Y is down; negate dy so angles match mol (Y-up) convention
            def angle_s(tup): return np.arctan2(-(tup[1]-sc[1]), tup[0]-sc[0])
            for (_, mx, my), (sx, sy) in zip(
                sorted(mol_list, key=angle_m),
                sorted(svg_list, key=angle_s),
            ):
                mol_calib.append((mx, my))
                svg_calib.append((sx, sy))

    transform = _fit_transform(np.array(mol_calib), np.array(svg_calib)) if mol_calib else None

    # ── Apply transform or fall back ─────────────────────────────────────────
    def to_svg(mx, my):
        if transform:
            s, ox, oy = transform
            return s * mx + ox, -s * my + oy
        # Fallback: bbox scaling with 10 % padding (approximate)
        xs = [conf.GetAtomPosition(j).x for j in range(mol.GetNumAtoms())]
        ys = [conf.GetAtomPosition(j).y for j in range(mol.GetNumAtoms())]
        pad = 0.10
        mw = max(xs) - min(xs) or 1e-6
        mh = max(ys) - min(ys) or 1e-6
        sc = min(W*(1-2*pad)/mw, H*(1-2*pad)/mh)
        cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2
        return W/2+(mx-cx)*sc, H/2-(my-cy)*sc

    positions = []
    for idx in heavy_atom_indices:
        mx, my = mol_xy(idx)
        sx, sy = to_svg(mx, my)
        positions.append([round(sx/W, 5), round(sy/H, 5)])
    return positions


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(JSON_IN) as f:
        data = json.load(f)

    ok = skipped = 0
    for mol in data["molecules"]:
        svg_path = SVG_DIR / f"{mol['id']}.svg"
        if not svg_path.exists():
            skipped += 1
            continue

        raw = svg_path.read_text(encoding="utf-8", errors="replace")
        svg = whiten_svg(raw)

        # Store original SVG dimensions so the browser can display at a fixed scale
        W, H = _svg_dims(svg)
        mol["svg_w"] = int(W)
        mol["svg_h"] = int(H)

        # Atom positions as fractions of the ORIGINAL SVG viewBox
        for g in mol["groups"]:
            g["atoms"] = atom_positions(mol["smiles"], svg, g["heavy_atoms"])

        mol["svg"] = svg
        ok += 1

    with open(JSON_IN, "w") as f:
        json.dump(data, f, separators=(",", ":"))

    size_kb = JSON_IN.stat().st_size / 1024
    print(f"Embedded {ok} SVGs ({skipped} skipped)  →  {JSON_IN}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
