from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mol_to_matrix.matrix import SpinSystem, build_spin_system, save_spin_system
from mol_to_matrix.shifts import DEFAULT_SOLVENT, make_test_mol_3d


def smiles_to_spin_system(smiles: str, solvent: str = DEFAULT_SOLVENT) -> SpinSystem:
    """SMILES -> 3D embed -> shifts + couplings -> grouped spin-system matrix."""
    mol = make_test_mol_3d(smiles)  # AddHs + ETKDGv3 + MMFF94
    return build_spin_system(mol, solvent=solvent)


def run_batch(
    csv_path: str | Path,
    out_dir: str | Path,
    smiles_col: str = "smiles",
    id_col: str | None = None,
    solvent: str = DEFAULT_SOLVENT,
    max_groups: int = 8,
) -> dict[str, int]:
    """Build and save a spin system for every SMILES in a CSV.

    Molecules that fail to process or exceed max_groups are skipped. Returns a
    summary {ok, skipped, total}.
    """
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    n_ok = n_skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="mol->matrix"):
        smiles = row[smiles_col]
        stem = str(row[id_col]) if id_col else f"mol_{idx:05d}"
        try:
            system = smiles_to_spin_system(smiles, solvent=solvent)
            if system.n_groups > max_groups:
                n_skipped += 1
                continue
            save_spin_system(system, out_dir / stem)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - skip and keep going on bad inputs
            tqdm.write(f"skip {stem} ({smiles}): {exc}")
            n_skipped += 1

    return {"ok": n_ok, "skipped": n_skipped, "total": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser(description="SMILES CSV -> spin-system matrices")
    parser.add_argument("csv", help="input CSV with a SMILES column")
    parser.add_argument("out_dir", help="output directory for .npy/.json matrices")
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--solvent", default=DEFAULT_SOLVENT)
    parser.add_argument("--max-groups", type=int, default=8)
    args = parser.parse_args()

    summary = run_batch(
        args.csv,
        args.out_dir,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        solvent=args.solvent,
        max_groups=args.max_groups,
    )
    print(f"done: {summary}")


if __name__ == "__main__":
    main()
