"""spin_group_filter.py — apply the deuterium substitution test to count spin systems.

For each candidate molecule, every C-H proton is individually replaced with
deuterium (isotope=2).  The canonical isomeric SMILES of the resulting molecule
is recorded.  Two protons are in the same spin system only if they produce
identical canonical SMILES (i.e. they are homotopic).  Enantiotopic and
diastereotopic protons each produce distinct SMILES and are counted separately.

Molecules with exactly 8 spin systems are written to smiles_8group.csv.

Input:  generate/data/raw/candidates_8spin.csv
Output: generate/data/raw/smiles_8group.csv
"""

import csv
from pathlib import Path

from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE  = REPO_ROOT / "generate" / "data" / "candidates_8spin_round01.csv"
OUTPUT_FILE = REPO_ROOT / "generate" / "data" / "smiles_8spin_round02.csv"
TARGET_GROUPS = 8


EXCHANGEABLE_ATOMS = {7, 8, 16}  # N, O, S


def strip_exchangeable_protons(mol: Chem.Mol) -> Chem.Mol:
    """Return mol with all H atoms on N, O, or S removed.

    These protons are fast-exchanging in solution and invisible as spin
    systems.  Removing them before the deuterium test prevents them from
    influencing stereocentre assignment or SMILES canonicalisation.
    """
    mol_h = Chem.AddHs(mol)
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


def analyze_spin_systems(mol: Chem.Mol) -> tuple[int, list[int]]:
    """Return (n_spin_groups, group_sizes) via the deuterium substitution test.

    Exchangeable protons (NH, OH, SH) are stripped before analysis.
    Two C-H protons share a group iff replacing either with D produces
    the same canonical isomeric SMILES (i.e. they are homotopic).
    """
    mol_h = strip_exchangeable_protons(mol)

    smi_to_count: dict[str, int] = {}

    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue

        rw = Chem.RWMol(mol_h)
        rw.GetAtomWithIdx(atom.GetIdx()).SetIsotope(2)
        deuterated = rw.GetMol()

        Chem.AssignStereochemistry(deuterated, cleanIt=False, force=True)
        smi = Chem.MolToSmiles(deuterated, isomericSmiles=True)

        smi_to_count[smi] = smi_to_count.get(smi, 0) + 1

    group_sizes = sorted(smi_to_count.values(), reverse=True)
    return len(smi_to_count), group_sizes


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0

    with open(INPUT_FILE, newline="") as f_in, \
         open(OUTPUT_FILE, "w", newline="") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["chembl_id", "smiles", "inchikey", "n_groups", "group_sizes"])

        rows = list(reader)  # load into memory — candidates file is manageable
        for row in tqdm(rows, desc="Deuterium test", unit=" mol"):
            total += 1
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue

            n_groups, group_sizes = analyze_spin_systems(mol)

            if n_groups == TARGET_GROUPS:
                writer.writerow([
                    row["chembl_id"],
                    row["smiles"],
                    row["inchikey"],
                    n_groups,
                    ";".join(map(str, group_sizes)),
                ])
                kept += 1

    print(f"\nProcessed : {total:>10,}")
    print(f"Kept (n=8): {kept:>10,}  ({100 * kept / max(total, 1):.1f}%)")
    print(f"Output    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
