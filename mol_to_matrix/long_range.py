from __future__ import annotations

from rdkit import Chem

# Long-range 4J couplings (Hz), Pretsch (Tables of Spectral Data, 2009):
#  - allylic  H-C-C=C-H   p.165: -3 to +2  -> representative -1.3
#  - benzylic H-C-C(ar):C(ar)-H (sidechain to ortho ring H), p.177 toluene
#    4J_ab = -0.7
# Both are small but survive the 0.3 Hz pruning threshold. Homoallylic 5J,
# benzylic 5J/6J (meta/para), and saturated W-coupling are geometry-specific
# and omitted.
J_ALLYLIC = -1.3
J_BENZYLIC = -0.7


def _heavy_neighbor(mol: Chem.Mol, h_idx: int) -> int | None:
    """The single heavy atom a hydrogen is bonded to (None if isolated)."""
    nbrs = mol.GetAtomWithIdx(h_idx).GetNeighbors()
    return nbrs[0].GetIdx() if nbrs else None


def long_range_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Estimate long-range 4J couplings: allylic (H-C-C=C-H) and benzylic
    (H-C-C(ar):C(ar)-H, sidechain proton to ortho ring proton).

    Returns {(atom_i, atom_j): J_Hz} with i < j, keyed by RDKit atom indices.
    """
    hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    couplings: dict[tuple[int, int], float] = {}
    for a in range(len(hs)):
        for b in range(a + 1, len(hs)):
            i, j = hs[a], hs[b]
            ca, cc = _heavy_neighbor(mol, i), _heavy_neighbor(mol, j)
            if ca is None or cc is None or ca == cc:
                continue
            if mol.GetBondBetweenAtoms(ca, cc) is not None:
                continue  # bonded -> shorter-range, handled elsewhere
            path = Chem.GetShortestPath(mol, ca, cc)
            if len(path) != 3:  # ca - cb - cc (two bonds between heavy neighbours)
                continue
            cb = path[1]
            if not all(mol.GetAtomWithIdx(x).GetAtomicNum() == 6 for x in (ca, cb, cc)):
                continue
            bonds = (mol.GetBondBetweenAtoms(ca, cb), mol.GetBondBetweenAtoms(cb, cc))
            doubles = sum(1 for bd in bonds if bd.GetBondType() == Chem.BondType.DOUBLE)
            aromatic = [mol.GetAtomWithIdx(x).GetIsAromatic() for x in (ca, cb, cc)]
            if doubles == 1:  # one C=C in the path -> allylic
                couplings[(i, j)] = J_ALLYLIC
            elif aromatic[1] and (aromatic[0] != aromatic[2]):
                # middle is a ring carbon, exactly one endpoint is aromatic:
                # sidechain proton (sp3 end) <-> ortho ring proton (aromatic end)
                couplings[(i, j)] = J_BENZYLIC
    return couplings


if __name__ == "__main__":
    from mol_to_matrix.shifts import make_test_mol_3d

    for smi, name in [("C=CC", "propene (allylic)"), ("Cc1ccccc1", "toluene (benzylic)"),
                      ("CCC", "propane (none)")]:
        mol = make_test_mol_3d(smi)
        js = sorted(long_range_couplings(mol).values())
        print(f"{name:>22}: {js} Hz ({len(js)} pairs)")
