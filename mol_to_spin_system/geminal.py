from __future__ import annotations

from rdkit import Chem

# Geminal 2J (Hz), additive model from Pretsch et al., Tables of Spectral Data
# for Structure Determination of Organic Compounds (2009), section 5.1.2:
#   2J = base + electronegative-substituent terms + adjacent-pi terms
BASE_SP3 = -12.4   # CH4
BASE_SP2 = 2.0     # terminal =CH2 (ethylene +2.5)
EN_CORRECTION = 1.6  # per O/N/halogen on the CH2 carbon (drives 2J toward 0)

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
    for nbr in carbon.GetNeighbors():
        if nbr.GetAtomicNum() == 1:
            continue
        if nbr.GetAtomicNum() in _ELECTRONEGATIVE:
            j += EN_CORRECTION
            continue
        kind = _adjacent_pi(nbr)
        if kind:
            j += PI_CORRECTION[kind]
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
        if len(hs) < 2:
            continue
        j = round(_geminal_2j(atom), 1)
        for a in range(len(hs)):
            for b in range(a + 1, len(hs)):
                couplings[(hs[a], hs[b])] = j
    return couplings


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    # (SMILES, the carbon's geminal partner, Pretsch reference 2J)
    cases = [
        ("ClCCl", "CH2Cl2", -7.5),
        ("Cc1ccccc1", "toluene", -14.3),
        ("CC(C)=O", "acetone", -14.9),
        ("CC#N", "CH3CN", -16.9),
        ("N#CCC#N", "CH2(CN)2", -20.3),
    ]
    for smi, name, ref in cases:
        mol = make_test_mol_3d(smi)
        js = set(geminal_couplings(mol).values())
        print(f"{name:>10}: predicted {sorted(js)}  (Pretsch {ref})")
