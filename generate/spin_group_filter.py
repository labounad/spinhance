"""spin_group_filter.py — apply the deuterium substitution test to count spin systems.

For each candidate molecule, every C-H proton is individually replaced with
deuterium (isotope=2).  A 3D conformer is embedded first so that
AssignStereochemistryFrom3D can tag the newly created CHD stereocentre from
the actual geometry — not from the 2D graph alone.  This correctly separates
diastereotopic methylenes in chiral molecules.

Enantiotopic protons are counted as SEPARATE spin systems (user requirement:
second-order coupling regime).  Homotopic protons (e.g. CH3) produce identical
substituted SMILES and are grouped as one.

Exchangeable protons (NH, OH, SH) are stripped before analysis and ignored.

Molecules with exactly 8 spin systems are written to smiles_8spin_round02.csv.

Input:  generate/data/candidates_8spin_round01.csv
Output: generate/data/smiles_8spin_round02.csv
"""

import csv
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REPO_ROOT   = Path(__file__).resolve().parent.parent
INPUT_FILE  = REPO_ROOT / "generate" / "data" / "candidates_8spin_round01.csv"
OUTPUT_FILE = REPO_ROOT / "generate" / "data" / "smiles_8spin_round02.csv"
TARGET_GROUPS = 8

EXCHANGEABLE_ATOMS = {7, 8, 16}  # N, O, S


# ── 3-D embedding ─────────────────────────────────────────────────────────────

def _embed_3d(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Return (mol_with_all_H, has_3d).

    Adds explicit H, attempts ETKDG v3 + MMFF (UFF fallback).
    ``has_3d`` is True when a usable conformer is present.
    """
    mol_h = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed     = 0xC0FFEE
    params.useRandomCoords = True   # helps with difficult ring systems

    status = AllChem.EmbedMolecule(mol_h, params)
    if status != 0 or mol_h.GetNumConformers() == 0:
        return mol_h, False

    # Geometry optimisation — MMFF first, UFF as fallback
    try:
        if AllChem.MMFFOptimizeMolecule(mol_h, maxIters=300) < 0:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=300)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=300)
        except Exception:
            pass  # keep the unoptimised ETKDG geometry

    return mol_h, mol_h.GetNumConformers() > 0


# ── exchangeable-proton removal ───────────────────────────────────────────────

def strip_exchangeable_protons(mol_h: Chem.Mol) -> Chem.Mol:
    """Remove H atoms bonded to N, O, or S from a molecule with explicit H.

    Conformation (if present) is preserved; atom removal shifts indices
    downward, which is fine because callers iterate over the returned mol.
    """
    to_remove = [
        atom.GetIdx()
        for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() == 1
        and atom.GetNeighbors()[0].GetAtomicNum() in EXCHANGEABLE_ATOMS
    ]
    rw = Chem.RWMol(mol_h)
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


# ── stereo assignment ─────────────────────────────────────────────────────────

def _assign_stereo(mol: Chem.Mol, *, use_3d: bool) -> None:
    """Assign stereochemistry in-place.

    When a 3D conformer is available, use AssignStereochemistryFrom3D to
    write chiral tags from geometry — this is the key step that correctly
    tags a newly created CHD stereocentre after isotope substitution.
    Then call AssignStereochemistry to propagate CIP descriptors.
    """
    if use_3d and mol.GetNumConformers() > 0:
        conf_id = mol.GetConformer(0).GetId()
        try:
            Chem.AssignStereochemistryFrom3D(
                mol, confId=conf_id, replaceExistingTags=True
            )
        except TypeError:
            # Older RDKit: positional argument only
            Chem.AssignStereochemistryFrom3D(mol, conf_id, True)

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


# ── substitution signature ────────────────────────────────────────────────────

def _signature(mol_h: Chem.Mol, atom_idx: int, *, use_3d: bool) -> str:
    """Canonical isomeric SMILES after substituting atom_idx with D."""
    sub = Chem.RWMol(mol_h)
    sub.GetAtomWithIdx(atom_idx).SetIsotope(2)
    sub_mol = sub.GetMol()
    _assign_stereo(sub_mol, use_3d=use_3d)
    return Chem.MolToSmiles(sub_mol, canonical=True, isomericSmiles=True)


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_spin_systems(mol: Chem.Mol) -> tuple[int, list[int]]:
    """Return (n_spin_groups, group_sizes) via the 3D deuterium substitution test.

    Enantiotopic and diastereotopic C-H protons are each counted as distinct
    spin systems.  Only homotopic protons (identical substituted SMILES) share
    a group.
    """
    mol_h, use_3d = _embed_3d(mol)
    mol_h = strip_exchangeable_protons(mol_h)

    smi_to_count: dict[str, int] = {}
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        smi = _signature(mol_h, atom.GetIdx(), use_3d=use_3d)
        smi_to_count[smi] = smi_to_count.get(smi, 0) + 1

    group_sizes = sorted(smi_to_count.values(), reverse=True)
    return len(smi_to_count), group_sizes


# ── batch filter ──────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    embed_failed = 0

    with open(INPUT_FILE, newline="") as f_in, \
         open(OUTPUT_FILE, "w", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["chembl_id", "smiles", "inchikey", "n_groups", "group_sizes"])

        rows = list(reader)
        for row in tqdm(rows, desc="Spin-group filter (3D)", unit=" mol"):
            total += 1
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue

            mol_h, use_3d = _embed_3d(mol)
            if not use_3d:
                embed_failed += 1

            mol_h = strip_exchangeable_protons(mol_h)

            smi_to_count: dict[str, int] = {}
            for atom in mol_h.GetAtoms():
                if atom.GetAtomicNum() != 1:
                    continue
                smi = _signature(mol_h, atom.GetIdx(), use_3d=use_3d)
                smi_to_count[smi] = smi_to_count.get(smi, 0) + 1

            n_groups   = len(smi_to_count)
            group_sizes = sorted(smi_to_count.values(), reverse=True)

            if n_groups == TARGET_GROUPS:
                writer.writerow([
                    row["chembl_id"],
                    row["smiles"],
                    row["inchikey"],
                    n_groups,
                    ";".join(map(str, group_sizes)),
                ])
                kept += 1

    print(f"\nProcessed    : {total:>10,}")
    print(f"Embed failed : {embed_failed:>10,}  ({100 * embed_failed / max(total, 1):.1f}%)")
    print(f"Kept (n=8)   : {kept:>10,}  ({100 * kept / max(total, 1):.1f}%)")
    print(f"Output       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
