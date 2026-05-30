from __future__ import annotations

from rdkit import Chem


def proton_groups(
    mol: Chem.Mol,
    bound_to: tuple[int, ...] = (6,),
) -> tuple[list[list[int]], dict[int, int]]:
    """Partition protons into spin groups by topological symmetry.

    Protons sharing a canonical rank (Chem.CanonicalRankAtoms, breakTies=False)
    are homotopic/enantiotopic and collapse into one spin group; diastereotopic
    protons get distinct ranks and stay separate.

    Only H's bound to an atom whose atomic number is in `bound_to` are included
    (default: carbon-bound, excluding exchangeable OH/NH).

    Returns (groups, group_of_atom):
      groups        - list of H-atom-index lists, one per spin group
      group_of_atom - {h_atom_idx: group_index}
    """
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    rank_to_group: dict[int, int] = {}
    groups: list[list[int]] = []
    group_of_atom: dict[int, int] = {}

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        nbrs = atom.GetNeighbors()
        if not nbrs or nbrs[0].GetAtomicNum() not in bound_to:
            continue
        rank = ranks[atom.GetIdx()]
        if rank not in rank_to_group:
            rank_to_group[rank] = len(groups)
            groups.append([])
        gi = rank_to_group[rank]
        groups[gi].append(atom.GetIdx())
        group_of_atom[atom.GetIdx()] = gi
    return groups, group_of_atom


def degeneracies(groups: list[list[int]]) -> list[int]:
    """Number of protons in each spin group (e.g. 3 for CH3, 9 for tBu)."""
    return [len(g) for g in groups]


if __name__ == "__main__":
    from mol_to_matrix.shifts import make_test_mol_3d

    for smi, name in [("CCO", "ethanol"), ("Cc1ccccc1", "toluene"),
                      ("c1ccccc1", "benzene"), ("CC(C)(C)O", "tert-butanol")]:
        mol = make_test_mol_3d(smi)
        groups, _ = proton_groups(mol)
        print(f"{name:>14}: {len(groups)} groups, degeneracies {degeneracies(groups)}")
