from __future__ import annotations

from rdkit import Chem

from mol_to_spin_system.aromatic import aromatic_couplings
from mol_to_spin_system.geminal import geminal_couplings
from mol_to_spin_system.heteroaromatic import heteroaromatic_couplings
from mol_to_spin_system.long_range import long_range_couplings
from mol_to_spin_system.olefinic import olefinic_couplings
from mol_to_spin_system.vicinal import vicinal_couplings

# (estimator, type-tag) — the tag records which mechanism produced each J so a
# per-type sampling sigma can be applied downstream (mol_to_spin_system.augment).
# heteroaromatic runs AFTER aromatic so its ring-specific values override the
# benzene fallback for the hetero rings it covers (same (i,j) keys).
_ESTIMATORS = (
    (geminal_couplings,       "geminal"),    # 2J, same carbon
    (vicinal_couplings,       "vicinal"),     # 3J, H-C-C-H single bond
    (olefinic_couplings,      "olefinic"),    # 3J, H-C=C-H
    (aromatic_couplings,      "aromatic"),    # benzene ortho/meta/para (fallback for all aromatic rings)
    (heteroaromatic_couplings, "aromatic"),   # ring-specific override for heteroaromatics
    (long_range_couplings,    "long_range"),  # 4J allylic/benzylic
)


def all_couplings_typed(
    mol: Chem.Mol,
) -> dict[tuple[int, int], tuple[float, str]]:
    """Merge every H-H estimator into ``{(atom_i, atom_j): (J_Hz, type)}``.

    Estimators are applied in order and a later one overwrites an earlier key:
    most cover disjoint topologies, but ``heteroaromatic`` deliberately follows
    ``aromatic`` to replace the benzene fallback on hetero rings. *type* is the
    producing mechanism's tag.
    """
    merged: dict[tuple[int, int], tuple[float, str]] = {}
    for estimator, tag in _ESTIMATORS:
        for pair, jval in estimator(mol).items():
            merged[pair] = (jval, tag)
    return merged


def all_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Merge every H-H coupling estimator into one {(atom_i, atom_j): J_Hz}."""
    return {pair: jv for pair, (jv, _) in all_couplings_typed(mol).items()}


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    for smi, name in [("C=CC", "propene"), ("Cc1ccccc1", "toluene")]:
        mol = make_test_mol_3d(smi)
        coups = all_couplings(mol)
        print(f"{name} ({len(coups)} couplings):")
        for (i, j), jval in sorted(coups.items()):
            si = mol.GetAtomWithIdx(i).GetSymbol()
            sj = mol.GetAtomWithIdx(j).GetSymbol()
            print(f"  {i:>2}{si}-{j:>2}{sj}: {jval:+5.1f}")
