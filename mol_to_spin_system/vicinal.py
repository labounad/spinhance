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
#    (ethane), decreasing with electronegative substituents on the coupling
#    carbons, by a substituent-specific decrement read from the Pretsch vicinal
#    substituent table (Tables of Spectral Data, 2009, p.162):
#
#        CH3CHF2  4.5   CH3CH2OH         6.9   CH3CH2CN     7.6
#        CH3CHCl2 6.1   (CH3CH2)3O+BF4-  7.2   (CH3CH2)2S   7.4
#        CH3CH2F  6.9   (CH3CH2)3N       7.1   (CH3CH2)4Si  8.0
#        CH3CH2Cl 7.2   (CH3CH2)4N+I-    7.3   CH3CH2Li     8.4
#
#    The book gives no closed-form rule, only that J drops with substituent
#    electronegativity and number. We reproduce the mono-substituted anchors
#    exactly with a per-element decrement, and add an extra geminal term so the
#    geminal di-halides (CHF2, CHCl2) also match. Electropositive substituents
#    (Si, Li) raise J above ethane; that nuance is left out (no such acyclic
#    H-C-C-H probe is common in the dataset) and falls back to the 7.3 base.
KARPLUS_J0 = 8.5
KARPLUS_J180 = 9.5
KARPLUS_OFFSET = -0.3

ROTATABLE_BASE = 7.3   # ethane-like, freely rotating (sp3-sp3)

# Single bond between two olefinic sp2 carbons (a conjugated =CH-CH= linkage,
# e.g. the C2-C3 bond of a 1,3-diene).  The ethane base (7.3 Hz) is wrong here:
# the rotamer population is dominated by the planar conjugated forms, giving a
# much larger 3J.  Pretsch (Tables of Spectral Data, 2009, p.166) lists the
# s-trans diene 3J at ~10.4 Hz and the s-cis (locked) form lower (~5-7 Hz).
# Acyclic 1,3-dienes overwhelmingly populate s-trans, so we use that value for
# freely-rotating sp2-sp2 single bonds; ring-locked sp2-sp2 bonds still fall
# through to the 3D-dihedral Karplus branch above.  (B5.)
DIENE_SP2_SP2 = 10.4   # s-trans 1,3-butadiene central-bond 3J

# Decrement (Hz) per electronegative substituent bonded to a coupling carbon,
# keyed by atomic number. Tuned to the mono-substituted Pretsch anchors:
#   CH3CH2F 6.9 (-0.4), CH3CH2Cl 7.2 (-0.1), CH3CH2OH 6.9 (-0.4),
#   (CH3CH2)3N 7.1 (-0.2 per ethyl carbon, one N each).
EN_DECREMENT = {
    9: 0.4,   # F
    17: 0.1,  # Cl
    35: 0.2,  # Br  (no direct anchor; between Cl and the trend)
    53: 0.1,  # I   (no direct anchor; ~Cl)
    8: 0.4,   # O
    7: 0.2,   # N
}
# Extra decrement for the 2nd (and each further) electronegative atom on the
# SAME carbon, capturing the strongly nonlinear geminal di-halide drop:
#   CH3CHF2 4.5  -> 2 F need a 2.8 total drop (0.4 + 0.4 + 2.0 extra)
#   CH3CHCl2 6.1 -> 2 Cl need a 1.2 total drop (0.1 + 0.1 + 1.0 extra)
EN_GEMINAL_EXTRA = {
    9: 2.0,   # F
    17: 1.0,  # Cl
}

_ELECTRONEGATIVE = set(EN_DECREMENT)


def karplus(phi_deg: float) -> float:
    """Vicinal 3J(H-C-C-H) from the H-C-C-H dihedral angle (degrees)."""
    c = math.cos(math.radians(phi_deg))
    j0 = KARPLUS_J0 if abs(phi_deg) <= 90.0 else KARPLUS_J180
    return j0 * c * c + KARPLUS_OFFSET


def _heavy_neighbor(mol: Chem.Mol, h_idx: int) -> int | None:
    """The single heavy atom a hydrogen is bonded to (None if isolated)."""
    nbrs = mol.GetAtomWithIdx(h_idx).GetNeighbors()
    return nbrs[0].GetIdx() if nbrs else None


def _is_olefinic_sp2_carbon(atom: Chem.Atom) -> bool:
    """True if a carbon is a non-aromatic sp2 carbon bearing a C=C double bond.

    Aromatic carbons are excluded: ring couplings are handled by the aromatic
    module, not here.
    """
    if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
        return False
    for bond in atom.GetBonds():
        if (bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(atom).GetAtomicNum() == 6):
            return True
    return False


def _rotatable_j(mol: Chem.Mol, ca: int, cb: int) -> float:
    """Freely-rotating vicinal 3J with substituent-specific decrements."""
    j = ROTATABLE_BASE
    for c in (ca, cb):
        per_carbon: dict[int, int] = {}
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            z = nbr.GetAtomicNum()
            if z in _ELECTRONEGATIVE:
                per_carbon[z] = per_carbon.get(z, 0) + 1
        for z, n in per_carbon.items():
            j -= EN_DECREMENT[z] * n
            if n >= 2:
                j -= EN_GEMINAL_EXTRA.get(z, 0.0) * (n - 1)
    return j


def vicinal_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Estimate vicinal 3J(H-C-C-H) across C-C single bonds.

    Ring bonds use Karplus on the 3D dihedral; freely rotating bonds use the
    substituent-adjusted empirical value. Returns {(atom_i, atom_j): J_Hz} with
    i < j, keyed by RDKit atom indices. Olefinic (C=C) and aromatic vicinal
    couplings are left to dedicated handlers. Only protium is emitted (D/T are
    skipped).
    """
    if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
        raise ValueError("mol needs a 3D conformer; embed it first (see make_test_mol_3d).")
    conf = mol.GetConformer()

    hs = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetIsotope() in (0, 1)
    ]
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
                # Ring-locked dihedral: Karplus on the actual 3D geometry.
                # NOTE (B3): this samples a single embedded conformer, so the
                # ring J-set depends on the embed seed (chair vs twist-boat).
                # A proper fix would Karplus-average over an ETKDG ensemble or
                # use ring-type defaults; left as a known limitation here.
                phi = rdMolTransforms.GetDihedralDeg(conf, i, ci, cj, j)
                j_hz = karplus(phi)
            elif _is_olefinic_sp2_carbon(ai) and _is_olefinic_sp2_carbon(aj):
                # Conjugated =CH-CH= single bond (e.g. 1,3-diene C2-C3): not a
                # freely-rotating sp3-sp3 bond.  Use the s-trans diene value
                # rather than the ethane base.  (B5.)
                j_hz = DIENE_SP2_SP2
            else:
                # sp3-sp3 (and sp-sp2 / mixed) freely-rotating single bonds.
                j_hz = _rotatable_j(mol, ci, cj)
            couplings[(i, j)] = round(j_hz, 1)
    return couplings


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    probes = [
        ("CC", "ethane"),
        ("CCO", "ethanol"),
        ("CCF", "fluoroethane"),
        ("CC(F)F", "1,1-difluoroethane"),
        ("CC(Cl)Cl", "1,1-dichloroethane"),
        ("CCC#N", "propionitrile"),
        ("C=CC=C", "1,3-butadiene"),   # sp2-sp2 central bond -> s-trans diene 3J
    ]
    for smi, name in probes:
        mol = make_test_mol_3d(smi)
        js = sorted(set(vicinal_couplings(mol).values()))
        print(f"{name:>20} (rotatable): {js} Hz")

    mol = make_test_mol_3d("C1CCCCC1")
    js = list(vicinal_couplings(mol).values())
    print(f"{'cyclohexane':>20} (ring/Karplus): {min(js)} .. {max(js)} Hz over {len(js)} pairs")
