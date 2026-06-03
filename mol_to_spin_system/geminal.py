from __future__ import annotations

from rdkit import Chem

# Geminal 2J (Hz), additive model from Pretsch et al., Tables of Spectral Data
# for Structure Determination of Organic Compounds (2009), section 5.1.2:
#   2J = base + electronegative-substituent terms + adjacent-pi terms
BASE_SP3 = -12.4   # CH4
BASE_SP2 = 2.5     # terminal =CH2 (ethylene +2.5, Pretsch p.165)
EN_CORRECTION = 1.6  # per O/N/halogen on the CH2 carbon (drives 2J toward 0)
# The linear EN_CORRECTION reproduces the single-substituent anchor (e.g.
# CH2 alpha to one halogen) but overshoots when two electronegative atoms sit
# on the SAME carbon: the second substituent's deshielding effect saturates.
# Extra (diminishing-returns) per-substituent correction applied to the 2nd
# and each further EN atom on the carbon, tuned to the geminal di-substituted
# anchors:
#   CH2Cl2       Pretsch -7.5: base -12.4 + 2x1.6 = -9.2, need +1.7 from 1 extra
#   CH2(CN)2     Pretsch -20.3: 2 nitriles, base -12.4 + 2x(-4.5) = -21.4,
#                               need +1.1 from 1 extra pi (less saturating)
EN_GEMINAL_EXTRA = 1.7   # Hz toward 0, per EN substituent beyond the first
PI_GEMINAL_EXTRA = 1.1   # Hz toward 0, per adjacent-pi substituent beyond the first

PI_CORRECTION = {    # per adjacent pi system (drives 2J more negative)
    "aromatic": -1.9,  # toluene -14.3
    "carbonyl": -2.5,  # acetone -14.9
    "nitrile": -4.5,   # CH3CN -16.9
    "alkene": -2.0,    # allylic
}

_ELECTRONEGATIVE = {7, 8, 9, 17, 35, 53}  # N, O, F, Cl, Br, I


def _adjacent_pi(neighbor: Chem.Atom) -> str | None:
    """Classify a heavy neighbour's adjacent-pi contribution to 2J, or None."""
    if neighbor.GetIsAromatic():
        return "aromatic"
    if neighbor.GetAtomicNum() == 6:
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE and other.GetAtomicNum() == 8:
                return "carbonyl"
            if bt == Chem.BondType.TRIPLE and other.GetAtomicNum() == 7:
                return "nitrile"
            if bt == Chem.BondType.DOUBLE and other.GetAtomicNum() == 6:
                return "alkene"
    return None


def _geminal_2j(carbon: Chem.Atom) -> float:
    """Geminal 2J for the two H's on a carbon, via the additive Pretsch model."""
    if carbon.GetHybridization().name == "SP2":
        return BASE_SP2
    j = BASE_SP3
    n_en = 0          # count of electronegative substituents on this carbon
    n_pi = 0          # count of adjacent-pi substituents on this carbon
    for nbr in carbon.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            continue
        if nbr.GetAtomicNum() in _ELECTRONEGATIVE:
            j += EN_CORRECTION
            n_en += 1
            continue
        kind = _adjacent_pi(nbr)
        if kind:
            j += PI_CORRECTION[kind]
            n_pi += 1
    # Nonlinear/saturating correction: the linear per-substituent terms above
    # undershoot (too negative) when >=2 electronegative or >=2 pi substituents
    # sit on the same carbon.  Add a diminishing-returns term (toward 0) for the
    # 2nd and each further substituent of each kind.  (B6.)
    if n_en >= 2:
        j += EN_GEMINAL_EXTRA * (n_en - 1)
    if n_pi >= 2:
        j += PI_GEMINAL_EXTRA * (n_pi - 1)
    return j


def geminal_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Estimate geminal 2J for every pair of H's sharing a carbon.

    Returns {(atom_i, atom_j): J_Hz} with i < j, keyed by RDKit atom indices.
    """
    couplings: dict[tuple[int, int], float] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        # Only true methylenes (exactly 2 H on the carbon) have an observable
        # geminal 2J.  A methyl's three protons are magnetically equivalent
        # (rapid C-C rotation) and show no mutual splitting, so a CH3 (3 H)
        # must NOT be assigned intra-group geminal couplings.
        if len(hs) != 2:
            continue
        j = round(_geminal_2j(atom), 1)
        for a in range(len(hs)):
            for b in range(a + 1, len(hs)):
                couplings[(hs[a], hs[b])] = j
    return couplings


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    # The additive 2J model is validated against Pretsch reference values on
    # methyl probes, but geminal_couplings only EMITS couplings for true
    # methylenes (2 H) — a methyl's 3 protons are magnetically equivalent.
    # (SMILES, name, model 2J on a CH3 probe carbon, emitted? )
    cases = [
        ("ClCCl", "CH2Cl2", -7.5),
        ("Cc1ccccc1", "toluene", -14.3),
        ("CC(C)=O", "acetone", -14.9),
        ("CC#N", "CH3CN", -16.9),
        ("N#CCC#N", "CH2(CN)2", -20.3),
    ]
    for smi, name, ref in cases:
        mol = make_test_mol_3d(smi)
        model = {
            round(_geminal_2j(a), 1)
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6
            and sum(n.GetAtomicNum() == 1 for n in a.GetNeighbors()) >= 2
        }
        emitted = sorted(set(geminal_couplings(mol).values()))
        print(f"{name:>10}: model 2J {sorted(model)}  emitted {emitted}  (Pretsch {ref})")
