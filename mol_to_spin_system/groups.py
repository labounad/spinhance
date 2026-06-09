from __future__ import annotations

from collections import defaultdict

from rdkit import Chem


def _mirror_smiles(mol: Chem.Mol) -> str:
    """Canonical isomeric SMILES of the mirror image (invert tetrahedral tags only;
    double-bond E/Z is unchanged by reflection). Used to make the equivalence key
    enantiomer-invariant — enantiotopic protons are equivalent in an achiral solvent."""
    mm = Chem.Mol(mol)
    for a in mm.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(mm, isomericSmiles=True)


def _substitution_key(work: Chem.Mol, h_idx: int) -> str:
    """Rigorous topicity key for a proton: mark it (isotope), perceive the resulting
    centre's configuration from the 3D conformer, and return an enantiomer-invariant
    canonical SMILES. Two protons share a key iff they are homotopic or enantiotopic
    (NMR-equivalent in an achiral solvent); diastereotopic protons get distinct keys.

    This is the substitution (replacement) test — the chemically correct definition of
    NMR topicity — and the only method that reliably separates diastereotopic protons
    (RDKit's CanonicalRankAtoms, even with includeChirality, does NOT)."""
    mm = Chem.Mol(work)
    mm.GetAtomWithIdx(h_idx).SetIsotope(2)         # mark THIS proton
    Chem.AssignStereochemistryFrom3D(mm)           # config of the now-distinct centre, from 3D
    s = Chem.MolToSmiles(mm, isomericSmiles=True)
    return min(s, _mirror_smiles(mm))


def proton_groups(
    mol: Chem.Mol,
    bound_to: tuple[int, ...] = (6,),
) -> tuple[list[list[int]], dict[int, int]]:
    """Partition protons into spin groups by true NMR topicity.

    Protons are equivalent iff homotopic or enantiotopic (related by a proper or, in an
    achiral solvent, improper symmetry); **diastereotopic protons are inequivalent and
    MUST stay in separate groups** (e.g. a CH2 next to a stereocentre, or on a prochiral
    carbon whose two other substituents differ — like a 3,3-disubstituted azetidine).

    Constitutional symmetry (``CanonicalRankAtoms``) is necessary but NOT sufficient:
    diastereotopic protons share a canonical rank, so we form candidate groups by rank
    (cheap) and then split each multi-proton candidate by the substitution test
    (:func:`_substitution_key`), which uses the molecule's 3D conformer. Without a
    conformer we fall back to the constitutional rank (and so cannot resolve
    diastereotopicity — callers should always pass a 3D-embedded molecule).

    Only H's bound to an atom in ``bound_to`` are included (default carbon-bound;
    excludes exchangeable OH/NH).

    Returns (groups, group_of_atom):
      groups        - list of H-atom-index lists, one per spin group
      group_of_atom - {h_atom_idx: group_index}
    """
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    # candidate groups: carbon-bound protium H's sharing a constitutional rank
    by_rank: dict[int, list[int]] = defaultdict(list)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        if atom.GetIsotope() not in (0, 1):          # protium only; D/T are NMR-invisible
            continue
        nbrs = atom.GetNeighbors()
        if not nbrs or nbrs[0].GetAtomicNum() not in bound_to:
            continue
        by_rank[ranks[atom.GetIdx()]].append(atom.GetIdx())

    has_conf = mol.GetNumConformers() > 0
    work = Chem.Mol(mol) if has_conf else None       # substitution test on a copy (no side effects)

    # Refine each candidate by the substitution test; subgroups keyed by topicity.
    final_key: dict[int, tuple] = {}
    for rank, atoms in by_rank.items():
        if len(atoms) == 1 or not has_conf:
            for h in atoms:
                final_key[h] = (rank, "")            # singletons / no-conformer: keep as-is
            continue
        for h in atoms:
            final_key[h] = (rank, _substitution_key(work, h))

    # Build groups deterministically: order by the smallest atom index in each subgroup.
    subgroups: dict[tuple, list[int]] = defaultdict(list)
    for h, key in final_key.items():
        subgroups[key].append(h)
    ordered = sorted(subgroups.values(), key=lambda g: min(g))

    groups: list[list[int]] = []
    group_of_atom: dict[int, int] = {}
    for gi, atoms in enumerate(ordered):
        atoms_sorted = sorted(atoms)
        groups.append(atoms_sorted)
        for h in atoms_sorted:
            group_of_atom[h] = gi
    return groups, group_of_atom


def degeneracies(groups: list[list[int]]) -> list[int]:
    """Number of protons in each spin group (e.g. 3 for CH3, 9 for tBu)."""
    return [len(g) for g in groups]


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    for smi, name in [("CCO", "ethanol (CH2 enantiotopic -> deg2)"),
                      ("CC(O)C", "isopropanol (2 CH3 enantiotopic -> deg6)"),
                      ("CC1(CN(C1)C2=C(C=CC=N2)C(F)(F)F)O", "azetidine (CH2 diastereotopic)"),
                      ("C1=CC(=C(C=C1Br)[C@@H](CCN)N)O", "chiral side chain (CH2 diastereotopic)")]:
        mol = make_test_mol_3d(smi)
        groups, _ = proton_groups(mol)
        print(f"{name:>42}: {len(groups)} groups, degeneracies {degeneracies(groups)}")
