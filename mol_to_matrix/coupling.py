from __future__ import annotations

from rdkit import Chem

from mol_to_matrix.aromatic import aromatic_couplings
from mol_to_matrix.geminal import geminal_couplings
from mol_to_matrix.long_range import long_range_couplings
from mol_to_matrix.olefinic import olefinic_couplings
from mol_to_matrix.vicinal import vicinal_couplings

_ESTIMATORS = (
    geminal_couplings,    # 2J, same carbon
    vicinal_couplings,    # 3J, H-C-C-H single bond
    olefinic_couplings,   # 3J, H-C=C-H
    aromatic_couplings,   # ortho/meta/para
    long_range_couplings, # 4J allylic
)


def all_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Merge every H-H coupling estimator into one {(atom_i, atom_j): J_Hz}.

    Each estimator covers a distinct topological relationship, so keys do not
    overlap.
    """
    merged: dict[tuple[int, int], float] = {}
    for estimator in _ESTIMATORS:
        for pair, jval in estimator(mol).items():
            merged[pair] = jval
    return merged


if __name__ == "__main__":
    from mol_to_matrix.shifts import make_test_mol_3d

    for smi, name in [("C=CC", "propene"), ("Cc1ccccc1", "toluene")]:
        mol = make_test_mol_3d(smi)
        coups = all_couplings(mol)
        print(f"{name} ({len(coups)} couplings):")
        for (i, j), jval in sorted(coups.items()):
            si = mol.GetAtomWithIdx(i).GetSymbol()
            sj = mol.GetAtomWithIdx(j).GetSymbol()
            print(f"  {i:>2}{si}-{j:>2}{sj}: {jval:+5.1f}")
