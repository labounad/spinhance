from __future__ import annotations

import math

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# Vicinal 3J(H-C-C-H) in Hz, handled in two regimes:
#
#  * rigid C-C bond (in a ring): the dihedral is locked, so use the Karplus
#    relation on the actual 3D geometry. Parameterized as in Pretsch (Tables of
#    Spectral Data, 2009, p.163): 3J = J0/J180 * cos^2(phi) - 0.3, with 8.5/9.5
#    reproducing gauche (~2 Hz) and anti (~9 Hz).
#
#  * freely rotating C-C bond (acyclic single bond): the conformer dihedral is
#    meaningless and the two-parameter Karplus undershoots the rotational
#    average, so use the empirical freely-rotating value instead. Base ~7.3 Hz
#    (ethane), decreasing ~0.5 Hz per electronegative substituent on the
#    coupling carbons (Pretsch p.162 substituent table).
KARPLUS_J0 = 8.5
KARPLUS_J180 = 9.5
KARPLUS_OFFSET = -0.3

ROTATABLE_BASE = 7.3   # ethane-like, freely rotating
EN_REDUCTION = 0.5     # per O/N/halogen on either coupling carbon

_ELECTRONEGATIVE = {7, 8, 9, 17, 35, 53}  # N, O, F, Cl, Br, I


def karplus(phi_deg: float) -> float:
    """Vicinal 3J(H-C-C-H) from the H-C-C-H dihedral angle (degrees)."""
    c = math.cos(math.radians(phi_deg))
    j0 = KARPLUS_J0 if abs(phi_deg) <= 90.0 else KARPLUS_J180
    return j0 * c * c + KARPLUS_OFFSET


def _heavy_neighbor(mol: Chem.Mol, h_idx: int) -> int | None:
    """The single heavy atom a hydrogen is bonded to (None if isolated)."""
    nbrs = mol.GetAtomWithIdx(h_idx).GetNeighbors()
    return nbrs[0].GetIdx() if nbrs else None


def _en_substituent_count(mol: Chem.Mol, ca: int, cb: int) -> int:
    """Electronegative atoms (O/N/halogen) bonded to either coupling carbon."""
    count = 0
    for c in (ca, cb):
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            if nbr.GetAtomicNum() in _ELECTRONEGATIVE:
                count += 1
    return count


def vicinal_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Estimate vicinal 3J(H-C-C-H) across C-C single bonds.

    Ring bonds use Karplus on the 3D dihedral; freely rotating bonds use the
    substituent-adjusted empirical value. Returns {(atom_i, atom_j): J_Hz} with
    i < j, keyed by RDKit atom indices. Olefinic (C=C) and aromatic vicinal
    couplings are left to dedicated handlers.
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
                continue  # only H-C-C-H here
            bond = mol.GetBondBetweenAtoms(ci, cj)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            if bond.IsInRing():
                phi = rdMolTransforms.GetDihedralDeg(conf, i, ci, cj, j)
                j_hz = karplus(phi)
            else:
                n_en = _en_substituent_count(mol, ci, cj)
                j_hz = ROTATABLE_BASE - EN_REDUCTION * n_en
            couplings[(i, j)] = round(j_hz, 1)
    return couplings


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    for smi, name in [("CC", "ethane"), ("CCO", "ethanol"), ("FCCF", "1,2-difluoroethane")]:
        mol = make_test_mol_3d(smi)
        js = sorted(set(vicinal_couplings(mol).values()))
        print(f"{name:>20} (rotatable): {js} Hz")

    mol = make_test_mol_3d("C1CCCCC1")
    js = list(vicinal_couplings(mol).values())
    print(f"{'cyclohexane':>20} (ring/Karplus): {min(js)} .. {max(js)} Hz over {len(js)} pairs")
