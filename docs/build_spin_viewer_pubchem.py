"""Build docs/data/spin_viewer.json (the 8-spin library) from PubChem molecules.

Two visualization aids on top of the basic depiction:
  * STEREO 2D: if a molecule has unassigned tetrahedral stereocentres (a racemic-
    form SMILES), assign ONE enantiomer and depict THAT, so the 2D shows wedge/hash
    bonds. The isomeric SMILES is stored + shown. (Spin-group classification still
    runs on the flat molecule, so the spin system is unchanged — only the drawing
    gains stereo.) Double-bond (E/Z) stereo is left untouched.
  * METHYLENE split: diastereotopic CH2 protons are two distinct groups on the SAME
    carbon, so their highlight circles overlap. Each such group gets a unit `off`
    vector (perpendicular to the C-X bond, opposite signs for the pair) that the
    viewer (docs/assets/viewers.js) uses to nudge the two circles apart so a hover
    distinguishes them.

Runs on the HPC where the consolidated_v2 PubChem records live:
    PYTHONPATH=. python docs/build_spin_viewer_pubchem.py \
        /gpfs/group/shenvi/Users/labounader/spinhance/consolidated_v2/records_train_shuf.json.gz /tmp/spin_viewer.json
"""
from __future__ import annotations

import gzip
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
from rdkit import Chem, RDLogger                                 # noqa: E402
from rdkit.Chem import AllChem, rdMolDescriptors, rdDepictor     # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D                          # noqa: E402
RDLogger.DisableLog("rdApp.*")
from generate.spin_equivalence import classify_spin_groups       # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/group/shenvi/Users/labounader/spinhance/consolidated_v2/records_train_shuf.json.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/data/spin_viewer.json"
N, SEED, W, H = 100, 7, 198, 120
random.seed(SEED)

_SOFT_COLORS = ["#ff2d9e", "#aaff00", "#00e5ff", "#bf5fff", "#ffcc00", "#00ffaa", "#ff6a2f", "#38cfff"]
_HARD_COLOR, _NONE_COLOR = "#c084fc", "#99a4b2"
_TIER_LABEL = {"HARD": "Equivalent", "SOFT": "Enantiotopic", "NONE": "Unique"}


def soft_map(groups):
    seen = {}
    for g in groups:
        if g.tier == "SOFT" and g.class_h_indices not in seen:
            seen[g.class_h_indices] = len(seen)
    return seen


def whiten(svg):
    svg = re.sub(r"<\?xml[^?]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg)
    svg = svg.replace("#000000", "#ffffff")        # bonds, wedges/hashes, heteroatom labels -> white
    svg = re.sub(r"\n\s*", " ", svg)
    return svg.strip()


def assign_one_enantiomer(mol):
    """Return a COPY with every UNASSIGNED tetrahedral centre set to one config
    (pick an enantiomer; it doesn't matter which). Atom indices are preserved, so
    spin-group atom mappings stay valid. Double-bond stereo is left as-is."""
    m = Chem.Mol(mol)
    for idx, lab in Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False):
        if lab == "?":
            m.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return m


def depict(mol):
    rdDepictor.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DSVG(W, H)
    o = d.drawOptions(); o.useBWAtomPalette(); o.clearBackground = False; o.bondLineWidth = 1.5  # 75% of the old 2
    d.DrawMolecule(mol)                              # wedges/hashes drawn from the assigned chiral tags
    d.FinishDrawing()
    return whiten(d.GetDrawingText()), {i: d.GetDrawCoords(i) for i in range(mol.GetNumAtoms())}


def methylene_offsets(mol, coords, groups):
    """Per-group unit offset (perpendicular to the C-X bond) for diastereotopic CH2
    pairs (two groups on the same single carbon); [0,0] otherwise."""
    by_carbon = defaultdict(list)
    for gi, g in enumerate(groups):
        by_carbon[tuple(sorted(int(a) for a in g.heavy_parent_indices))].append(gi)
    offs = [[0.0, 0.0] for _ in groups]
    for key, gis in by_carbon.items():
        if len(gis) == 2 and len(key) == 1 and key[0] in coords:
            c = key[0]
            nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors()
                    if n.GetAtomicNum() > 1 and n.GetIdx() in coords]
            pc = coords[c]; pn = coords[nbrs[0]] if nbrs else None
            vx, vy = (pn.x - pc.x, pn.y - pc.y) if pn else (0.0, 1.0)
            L = math.hypot(vx, vy) or 1.0
            px, py = -vy / L, vx / L                 # perpendicular unit vector (SVG px space)
            offs[gis[0]] = [round(px, 4), round(py, 4)]
            offs[gis[1]] = [round(-px, 4), round(-py, 4)]
    return offs


def main():
    recs = []
    for line in gzip.open(SRC, "rt"):
        line = line.strip().rstrip(",")
        if not line or line in "[]":
            continue
        rec = json.loads(line); smi = rec.get("smiles", "")
        if not smi or re.search(r"\[\d+[A-Z]", smi):     # skip isotopes
            continue
        recs.append(rec)
        if len(recs) >= N * 6:
            break
    random.shuffle(recs)

    mols = []
    for rec in recs:
        if len(mols) >= N:
            break
        smi = rec["smiles"]; mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            _, groups = classify_spin_groups(mol)        # on the FLAT mol (spin system unchanged)
        except Exception:
            continue
        if len(groups) != 8:
            continue
        stereo = assign_one_enantiomer(mol)              # depict one enantiomer -> wedges
        iso_smi = Chem.MolToSmiles(stereo)
        try:
            svg, coords = depict(stereo)
        except Exception:
            continue
        try:
            m3 = Chem.AddHs(mol); p = AllChem.ETKDGv3(); p.randomSeed = 42
            sdf = Chem.MolToMolBlock(Chem.RemoveHs(m3)) if AllChem.EmbedMolecule(m3, p) == 0 and \
                (AllChem.MMFFOptimizeMolecule(m3) or True) else None
        except Exception:
            sdf = None
        sm = soft_map(groups); offs = methylene_offsets(stereo, coords, groups); gd = []
        for gi, g in enumerate(groups):
            col = _HARD_COLOR if g.tier == "HARD" else (
                _SOFT_COLORS[sm.get(g.class_h_indices, 0) % len(_SOFT_COLORS)] if g.tier == "SOFT" else _NONE_COLOR)
            ats = [[round(coords[a].x / W, 5), round(coords[a].y / H, 5)]
                   for a in g.heavy_parent_indices if a in coords]
            gd.append({"label": g.label, "tier": g.tier, "tier_label": _TIER_LABEL[g.tier],
                       "h_count": len(g.h_indices), "color": col,
                       "heavy_atoms": list(g.heavy_parent_indices), "atoms": ats,
                       "off": offs[gi], "methylene": offs[gi] != [0.0, 0.0]})
        mols.append({"id": rec.get("chembl_id"), "smiles": iso_smi,
                     "formula": rdMolDescriptors.CalcMolFormula(mol),
                     "n_protons": sum(len(g.h_indices) for g in groups),
                     "sdf": sdf, "groups": gd, "svg": svg, "svg_w": W, "svg_h": H})
        if len(mols) % 20 == 0:
            print(f"  {len(mols)}/{N}", flush=True)

    json.dump({"meta": {"n": len(mols),
                        "source": "PubChem 8-group; RDKit 2D (one enantiomer, wedge/hash) + ETKDG 3D"},
               "molecules": mols}, open(OUT, "w"), separators=(",", ":"))
    import os
    print(f"wrote {OUT} {os.path.getsize(OUT)/1024:.1f} KB | {len(mols)} mols", flush=True)


if __name__ == "__main__":
    main()
