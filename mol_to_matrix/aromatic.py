from __future__ import annotations

from rdkit import Chem

# Aromatic ring H-H couplings (Hz), keyed by ring-bond separation between the
# carbons bearing the protons: 1 = ortho, 2 = meta, 3 = para. Standard benzene
# values (Pretsch, Tables of Spectral Data, 2009).
AROMATIC = {1: 7.5, 2: 1.5, 3: 0.7}


def _heavy_neighbor(mol: Chem.Mol, h_idx: int) -> int | None:
    """The single heavy atom a hydrogen is bonded to (None if isolated)."""
    nbrs = mol.GetAtomWithIdx(h_idx).GetNeighbors()
    return nbrs[0].GetIdx() if nbrs else None


def _share_aromatic_ring(mol: Chem.Mol, a: int, b: int) -> bool:
    """True if a and b belong to a common fully aromatic ring."""
    for ring in mol.GetRingInfo().AtomRings():
        if a in ring and b in ring and all(
            mol.GetAtomWithIdx(x).GetIsAromatic() for x in ring
        ):
            return True
    return False


def aromatic_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Estimate ortho/meta/para couplings between aromatic ring protons.

    Returns {(atom_i, atom_j): J_Hz} with i < j, keyed by RDKit atom indices.
    Only H's on carbons of a shared aromatic ring are considered.
    """
    hs = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1
        and a.GetNeighbors()
        and a.GetNeighbors()[0].GetIsAromatic()
    ]
    couplings: dict[tuple[int, int], float] = {}
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            i, j = hs[a], hs[b]
            ci, cj = _heavy_neighbor(mol, i), _heavy_neighbor(mol, j)
            if ci is None or cj is None or not _share_aromatic_ring(mol, ci, cj):
                continue
            sep = len(Chem.GetShortestPath(mol, ci, cj)) - 1
            jval = AROMATIC.get(sep)
            if jval is not None:
                couplings[(i, j)] = jval
    return couplings


if __name__ == "__main__":
    from mol_to_matrix.shifts import make_test_mol_3d

    for smi, name in [("c1ccccc1", "benzene"), ("Cc1ccccc1", "toluene")]:
        mol = make_test_mol_3d(smi)
        js = sorted(aromatic_couplings(mol).values())
        print(f"{name:>10}: {js} Hz")
