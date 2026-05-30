"""filter_chembl.py — pre-filter ChEMBL to molecules with >= 8 proton-bearing carbons.

Streams through the full ChEMBL chemreps file and keeps molecules whose
number of carbon atoms bearing at least one hydrogen is >= 8.  This is a
fast pre-filter; the exact magnetic equivalence analysis (reducing those
carbons to spin groups) happens in the next step.

Output: data/raw/candidates_8spin.csv
"""

import csv
import sys
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")  # suppress RDKit parse warnings

REPO_ROOT = Path(__file__).resolve().parent.parent
CHEMBL_FILE = REPO_ROOT / "generate" / "chembl" / "chembl_37_chemreps.txt"
OUTPUT_FILE = REPO_ROOT / "generate" / "data" / "candidates_8spin_round01.csv"
MAX_PROTON_BEARING_C = 8
MIN_PROTON = 8

def count_proton_bearing_carbons(mol: Chem.Mol) -> int:
    return sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0
    )

def count_protons(mol: Chem.Mol) -> int:
    return sum(
        atom.GetTotalNumHs() for atom in mol.GetAtoms()
    )


def count_lines(path: Path) -> int:
    """Count newlines in file without loading it into memory."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Counting lines in {CHEMBL_FILE.name} ...", end=" ", flush=True)
    total_lines = count_lines(CHEMBL_FILE)
    data_lines = total_lines - 1  # subtract header
    print(f"{total_lines:,} lines ({data_lines:,} molecules)")

    kept = 0
    skipped_parse = 0
    total = 0

    with open(CHEMBL_FILE) as f_in, open(OUTPUT_FILE, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["chembl_id", "smiles", "inchikey", "n_proton_bearing_c", "n_protons"])

        next(f_in)  # skip header row

        with tqdm(
            f_in,
            total=data_lines,
            desc="Filtering ChEMBL",
            unit=" mol",
            mininterval=1.0,
            dynamic_ncols=True,
        ) as pbar:
            for line in pbar:
                total += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue

                chembl_id, smiles, inchikey = parts[0], parts[1], parts[3]

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    skipped_parse += 1
                    continue

                n_C = count_proton_bearing_carbons(mol)
                n_H = count_protons(mol)

                if n_C <= MAX_PROTON_BEARING_C and n_H >= MIN_PROTON:
                    writer.writerow([chembl_id, smiles, inchikey, n_C, n_H])
                    kept += 1

                if total % 10_000 == 0:
                    pbar.set_postfix(
                        kept=f"{kept:,}",
                        kept_pct=f"{100 * kept / total:.1f}%",
                        parse_fail=skipped_parse,
                    )

    print(f"\nProcessed : {total:>10,}")
    print(f"Parse fail: {skipped_parse:>10,}")
    print(f"Kept      : {kept:>10,}  ({100 * kept / max(total, 1):.1f}%)")
    print(f"Output    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
