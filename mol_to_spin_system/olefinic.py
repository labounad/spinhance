from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# Olefinic 3J across a C=C double bond (Hz), substituent dependent.
#
# Pretsch (Tables of Spectral Data, 2009, p.166-167) tabulates the cis (Jab)
# and trans (Jac) couplings of monosubstituted ethylenes CH2=CH-R against R.
# Both shrink with increasing substituent electronegativity; the typical ranges
# (p.164) are Jcis 4-12 and Jtrans 12-19, with ethylene itself at the top
# (cis 11.6, trans 19.1, gem 2.5). We classify the substituent on each C=C
# carbon by the atom directly attached and look up (cis, trans); when the two
# carbons carry different substituent classes we use the one giving the larger
# perturbation (smaller J), matching the "values drop with electronegative
# substituents" trend. The geminal =CH2 2J is handled in geminal.py.
#
# Representative monosubstituted-ethylene values transcribed from p.166-167
# (column Jab = cis, Jac = trans):
#     R            cis    trans
#     -H           11.6   19.1   (ethylene)
#     alkyl/C      10.0   16.8   (CH3; alkyl/vinyl/cyclo all ~10/17)
#     -F            4.7   12.8
#     -Cl           7.5   14.5
#     -Br           7.1   14.9
#     -I            7.8   15.9
#     -O- (OH/OR)   6.4   14.0
#     -OC(=O)-      6.3   14.0   (vinyl ester O)
#     -N<           8.5   15.4   (NH2/NR2)
#     -S-          10.3   16.4   (SCH3)
#     -C(=O)-      10.7   17.6   (CHO/COR carbonyl C)
#     -C#/-CN      11.3   17.8   (alkynyl / nitrile)
J_ETHYLENE = (11.6, 19.1)  # (cis, trans), unsubstituted

# (cis, trans) keyed by a substituent class label.
_SUBST_J = {
    "alkyl": (10.0, 16.8),     # -CH3 and alkyl/vinyl/aryl-CH2/cyclo (~10.0/16.8)
    "vinyl": (10.1, 17.2),     # -CH=CH2 (conjugated diene C)
    "aryl": (11.0, 17.5),      # -phenyl etc. (sp2 aryl C)
    "alkynyl": (11.3, 17.8),   # -C#C-, ~nitrile-like
    "F": (4.7, 12.8),
    "Cl": (7.5, 14.5),
    "Br": (7.1, 14.9),
    "I": (7.8, 15.9),
    "O": (6.4, 14.0),          # -OH, -OR, -OC(=O)R
    "N": (8.5, 15.4),          # -NH2, -NR2, -NHC(=O)R
    "S": (10.3, 16.4),         # -SR
    "carbonyl": (10.7, 17.6),  # -CHO, -COR, -COOH, -CONR2 (sp2 carbonyl C)
    "nitrile": (11.3, 17.8),   # -C#N
}

# Rank used to pick the dominant substituent when the two C=C carbons differ:
# smaller cis coupling = stronger perturbation = dominant.
def _class_cis(label: str) -> float:
    return _SUBST_J[label][0]


def _heavy_neighbor(mol: Chem.Mol, h_idx: int) -> int | None:
    """The single heavy atom a hydrogen is bonded to (None if isolated)."""
    nbrs = mol.GetAtomWithIdx(h_idx).GetNeighbors()
    return nbrs[0].GetIdx() if nbrs else None


def _classify_substituent(atom: Chem.Atom, double_bond_partner: int) -> str | None:
    """Classify one heavy substituent on a vinyl carbon into a Pretsch class.

    `atom` is the substituent's first atom; `double_bond_partner` is the index
    of the other C=C carbon (so we don't reclassify the double bond itself).
    Returns None for plain alkyl carbons (handled by the 'alkyl' default).
    """
    z = atom.GetAtomicNum()
    if z == 9:
        return "F"
    if z == 17:
        return "Cl"
    if z == 35:
        return "Br"
    if z == 53:
        return "I"
    if z == 8:
        return "O"
    if z == 7:
        return "nitrile" if _is_nitrile_carbon(atom) else "N"
    if z == 16:
        return "S"
    if z == 6:
        if atom.GetIsAromatic():
            return "aryl"
        # sp2 carbonyl carbon (C=O)?
        for b in atom.GetBonds():
            o = b.GetOtherAtom(atom)
            if b.GetBondType() == Chem.BondType.DOUBLE and o.GetAtomicNum() == 8:
                return "carbonyl"
            if b.GetBondType() == Chem.BondType.TRIPLE:
                if o.GetAtomicNum() == 7:
                    return "nitrile"
                return "alkynyl"
            if (b.GetBondType() == Chem.BondType.DOUBLE
                    and o.GetAtomicNum() == 6
                    and o.GetIdx() != double_bond_partner):
                return "vinyl"
        return "alkyl"
    return "alkyl"


def _is_nitrile_carbon(n_atom: Chem.Atom) -> bool:
    for b in n_atom.GetBonds():
        if b.GetBondType() == Chem.BondType.TRIPLE and b.GetOtherAtom(n_atom).GetAtomicNum() == 6:
            return True
    return False


def _olefin_j(mol: Chem.Mol, ci: int, cj: int) -> tuple[float, float]:
    """(cis, trans) coupling for H's on the two carbons of a C=C bond.

    Collects the substituent classes on both vinyl carbons and picks the one
    causing the largest reduction (smallest cis); unsubstituted -> ethylene.
    """
    labels: list[str] = []
    for c, partner in ((ci, cj), (cj, ci)):
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            if nbr.GetIdx() == partner or nbr.GetAtomicNum() == 1:
                continue
            label = _classify_substituent(nbr, partner)
            if label is not None:
                labels.append(label)
    if not labels:
        return J_ETHYLENE
    dominant = min(labels, key=_class_cis)
    return _SUBST_J[dominant]


def olefinic_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Estimate olefinic 3J for H's on opposite carbons of a C=C double bond.

    cis (dihedral < 90 deg) vs trans (>= 90 deg) is read from the 3D geometry;
    the magnitude is substituent-dependent (Pretsch p.166-167). Returns
    {(atom_i, atom_j): J_Hz} with i < j, keyed by RDKit atom indices. Only
    protium is emitted (D/T are skipped).
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
                continue
            bond = mol.GetBondBetweenAtoms(ci, cj)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            j_cis, j_trans = _olefin_j(mol, ci, cj)
            dih = abs(rdMolTransforms.GetDihedralDeg(conf, i, ci, cj, j))
            couplings[(i, j)] = round(j_cis if dih < 90.0 else j_trans, 1)
    return couplings


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    for smi, name in [
        ("C=C", "ethylene"),
        ("C=CC", "propene"),
        ("C=CCl", "vinyl chloride"),
        ("C=CF", "vinyl fluoride"),
        ("C=CO", "vinyl alcohol"),
        ("C=CC=O", "acrolein"),
    ]:
        mol = make_test_mol_3d(smi)
        js = sorted(olefinic_couplings(mol).values())
        print(f"{name:>16}: {js} Hz")
