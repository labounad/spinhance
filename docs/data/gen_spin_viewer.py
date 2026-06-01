"""Generate docs/data/spin_viewer.json for the web molecule browser.

Gallery shows styled tiles; detail view uses 3Dmol.js with 3D coordinates.

Run from repo root:
    python docs/data/gen_spin_viewer.py
"""
from __future__ import annotations
import sys, json, random, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from generate.spin_equivalence import classify_spin_groups

RDLogger.DisableLog("rdApp.*")

N    = 100
SEED = 7

# Neon/vivid palette — deliberately avoids CPK atom hues:
#   CPK: C=grey, H=white, N=blue, O=red, S=yellow, P=orange, Cl=green, Br=brown
#   Our palette: hot pink, lime, cyan, violet, gold, teal, coral, sky — all distinct
_SOFT_COLORS = ["#ff2d9e","#aaff00","#00e5ff","#bf5fff","#ffcc00","#00ffaa","#ff6a2f","#38cfff"]
_HARD_COLOR  = "#c084fc"   # soft violet — distinct from blue N
_NONE_COLOR  = "#99a4b2"
_TIER_LABEL  = {"HARD": "Equivalent", "SOFT": "Enantiotopic", "NONE": "Unique"}


def _build_soft_color_map(groups) -> dict:
    seen: dict = {}
    for g in groups:
        if g.tier == "SOFT" and g.class_h_indices not in seen:
            seen[g.class_h_indices] = len(seen)
    return seen


def _embed_sdf(smiles: str) -> str | None:
    """Return an SDF/molblock string (includes bond orders for aromatic display)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) < 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToMolBlock(mol)


def _process(mol_data: dict):
    smiles    = mol_data["smiles"]
    chembl_id = mol_data["chembl_id"]

    # Skip molecules with isotope labels (e.g. [3H] tritium → renders as "T")
    if re.search(r'\[\d+[A-Z]', smiles):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    _, groups = classify_spin_groups(mol)
    if len(groups) != 8:
        return None

    xyz = _embed_sdf(smiles)
    if xyz is None:
        return None

    formula  = rdMolDescriptors.CalcMolFormula(mol)
    soft_map = _build_soft_color_map(groups)

    group_data = []
    for g in groups:
        if g.tier == "HARD":
            col = _HARD_COLOR
        elif g.tier == "SOFT":
            col = _SOFT_COLORS[soft_map.get(g.class_h_indices, 0) % len(_SOFT_COLORS)]
        else:
            col = _NONE_COLOR
        group_data.append({
            "label":      g.label,
            "tier":       g.tier,
            "tier_label": _TIER_LABEL[g.tier],
            "h_count":    len(g.h_indices),
            "color":      col,
            "heavy_atoms": list(g.heavy_parent_indices),
        })

    n_protons = sum(len(g.h_indices) for g in groups)

    return {
        "id":       chembl_id,
        "smiles":   smiles,
        "formula":  formula,
        "n_protons": n_protons,
        "sdf":      xyz,
        "groups":   group_data,
    }


def main():
    src = Path(__file__).parent / "spin_systems_chembl_8spin_randomized.json"
    out = Path(__file__).parent / "spin_viewer.json"

    print(f"Loading {src.name}…", flush=True)
    with open(src) as f:
        all_data = json.load(f)

    random.seed(SEED)
    sample = random.sample(all_data, min(N * 4, len(all_data)))

    molecules = []
    for i, md in enumerate(sample):
        if len(molecules) >= N:
            break
        if i % 20 == 0:
            print(f"  {i}/{len(sample)} checked, {len(molecules)}/{N} collected…", flush=True)
        result = _process(md)
        if result:
            molecules.append(result)

    output = {"meta": {"n": len(molecules)}, "molecules": molecules}
    with open(out, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_kb = out.stat().st_size / 1024
    print(f"\nWrote {len(molecules)} molecules → {out}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
