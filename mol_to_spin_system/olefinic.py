from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# Olefinic 3J across a C=C double bond (Hz). Representative values from Pretsch
# (Tables of Spectral Data, 2009, p.164): cis 4-12 (ethylene 11.6), trans 14-19
# (ethylene 19.1). cis/trans is read from the H-C=C-H dihedral. Geminal =CH2
# (2J ~ +2) is handled in geminal.py.
J_CIS = 11.0
J_TRANS = 17.0


def _heavy_neighbor(mol: Chem.Mol, h_idx: int) -> int | None:
    """The single heavy atom a hydrogen is bonded to (None if isolated)."""
    nbrs = mol.GetAtomWithIdx(h_idx).GetNeighbors()
    return nbrs[0].GetIdx() if nbrs else None


def olefinic_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Estimate olefinic 3J for H's on opposite carbons of a C=C double bond.

    cis (dihedral < 90 deg) vs trans (>= 90 deg) is read from the 3D geometry.
    Returns {(atom_i, atom_j): J_Hz} with i < j, keyed by RDKit atom indices.
    """
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        raise ValueError("mol needs a 3D conformer; embed it first (see make_test_mol_3d).")
    conf = mol.GetConformer()

    hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    couplings: dict[tuple[int, int], float] = {}
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            i, j = hs[a], hs[b]
            ci, cj = _heavy_neighbor(mol, i), _heavy_neighbor(mol, j)
            if ci is None or cj is None or ci == cj:
                continue
            ai, aj = mol.GetAtomWithIdx(ci), mol.GetAtomWithIdx(cj)
            if ai.GetAtomicNum() != 6 or aj.GetAtomicNum() != 6:
                continue
            bond = mol.GetBondBetweenAtoms(ci, cj)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            dih = abs(rdMolTransforms.GetDihedralDeg(conf, i, ci, cj, j))
            couplings[(i, j)] = J_CIS if dih < 90.0 else J_TRANS
    return couplings


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    for smi, name in [("C=C", "ethylene"), ("C=CCl", "vinyl chloride")]:
        mol = make_test_mol_3d(smi)
        js = sorted(olefinic_couplings(mol).values())
        print(f"{name:>16}: {js} Hz")
