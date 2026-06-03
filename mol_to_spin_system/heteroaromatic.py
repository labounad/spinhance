from __future__ import annotations

from rdkit import Chem

"""Ring-system-specific aromatic H-H couplings (Pretsch 2009, §5.6).

Benzene's ortho/meta/para = 7.5/1.5/0.7 is wrong for heteroaromatics, where the
coupling depends on each proton's ring position relative to the heteroatom(s):
e.g. furan ³J(2,3) ≈ 1.8 Hz but ³J(3,4) ≈ 3.4 Hz (both "ortho"). This module
classifies a monocyclic aromatic ring, numbers its atoms with the heteroatom at
position 1 (Pretsch convention), and looks up J by the unordered position pair.

Values are Pretsch verbatim. Every symmetry-equivalent pair is listed explicitly
so lookup is a direct dict. Carbocyclic rings and ring systems not in the table
fall back to the benzene values in :mod:`mol_to_spin_system.aromatic`.

Scope (v1): non-fused 5- and 6-membered rings with a single heteroatom
(pyridine, furan, thiophene, pyrrole). Diazines/azoles and fused systems are
added by extending ``_RING_J`` + ``_classify`` only.
"""

# {ring name: {frozenset({pos_i, pos_j}): J_Hz}}  — positions 1-based, heteroatom = 1.
_RING_J: dict[str, dict[frozenset, float]] = {
    "furan": {                       # O=1; C2,3,4,5
        frozenset({2, 3}): 1.8, frozenset({4, 5}): 1.8,
        frozenset({3, 4}): 3.4,
        frozenset({2, 4}): 0.9, frozenset({3, 5}): 0.9,
        frozenset({2, 5}): 1.5,
    },
    "thiophene": {                   # S=1
        frozenset({2, 3}): 4.8, frozenset({4, 5}): 4.8,
        frozenset({3, 4}): 3.5,
        frozenset({2, 4}): 1.0, frozenset({3, 5}): 1.0,
        frozenset({2, 5}): 2.8,
    },
    "pyrrole": {                     # NH=1
        frozenset({2, 3}): 2.6, frozenset({4, 5}): 2.6,
        frozenset({3, 4}): 3.5,
        frozenset({2, 4}): 1.3, frozenset({3, 5}): 1.3,
        frozenset({2, 5}): 2.1,
    },
    "pyridine": {                    # N=1; C2,3,4,5,6
        frozenset({2, 3}): 6.0, frozenset({5, 6}): 6.0,
        frozenset({3, 4}): 7.6, frozenset({4, 5}): 7.6,
        frozenset({2, 4}): 1.9, frozenset({4, 6}): 1.9,
        frozenset({3, 5}): 1.6,
        frozenset({2, 5}): 0.9, frozenset({3, 6}): 0.9,
        frozenset({2, 6}): 0.4,
    },
}


def _cycle_order(mol: Chem.Mol, ring: tuple[int, ...]) -> list[int]:
    """Ring atom indices in connectivity (walk) order."""
    rs = set(ring)
    order = [ring[0]]
    prev = None
    while len(order) < len(ring):
        cur = order[-1]
        nxt = None
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j in rs and j != prev and j not in order:
                nxt = j
                break
        if nxt is None:
            break
        prev = cur
        order.append(nxt)
    return order


def _classify(size: int, het: list[tuple[int, int]]) -> str | None:
    """Ring name from size + heteroatom (idx, atomic-num) list. v1: single-het."""
    if len(het) != 1:
        return None
    z = het[0][1]
    if size == 6 and z == 7:
        return "pyridine"
    if size == 5:
        return {8: "furan", 16: "thiophene", 7: "pyrrole"}.get(z)
    return None


def _positions(mol: Chem.Mol, ring: tuple[int, ...]) -> dict[int, int] | None:
    """Map each ring atom -> Pretsch position (heteroatom = 1). Single-het only."""
    order = _cycle_order(mol, ring)
    if len(order) != len(ring):
        return None
    het_k = [k for k, i in enumerate(order)
             if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]
    if len(het_k) != 1:
        return None
    k = het_k[0]
    rot = order[k:] + order[:k]            # heteroatom first
    return {atom: p + 1 for p, atom in enumerate(rot)}


def _classified_rings(mol: Chem.Mol):
    """Yield (ring_atoms, name, {atom_idx: position}) for supported monocyclic
    (non-fused) heteroaromatic rings."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(ri.NumAtomRings(i) > 1 for i in ring):     # skip fused rings (v1)
            continue
        size = len(ring)
        if size not in (5, 6):
            continue
        het = [(i, mol.GetAtomWithIdx(i).GetAtomicNum())
               for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]
        name = _classify(size, het)
        if name is None:
            continue
        pos = _positions(mol, ring)
        if pos:
            yield ring, name, pos


def heteroaromatic_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Position-specific ring couplings for supported heteroaromatic rings.

    Returns {(atom_i, atom_j): J_Hz} with i < j. Overrides the benzene
    fallback for the rings it covers (it is merged AFTER ``aromatic_couplings``
    in ``coupling.all_couplings_typed``). Only protium C-H are coupled.
    """
    out: dict[tuple[int, int], float] = {}
    for ring, name, pos in _classified_rings(mol):
        table = _RING_J[name]
        ch: dict[int, int] = {}                # ring carbon -> its protium H
        for c in ring:
            a = mol.GetAtomWithIdx(c)
            if a.GetAtomicNum() != 6:
                continue
            hs = [n.GetIdx() for n in a.GetNeighbors()
                  if n.GetAtomicNum() == 1 and n.GetIsotope() in (0, 1)]
            if len(hs) == 1:
                ch[c] = hs[0]
        cs = list(ch)
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                ca, cb = cs[a], cs[b]
                jval = table.get(frozenset({pos[ca], pos[cb]}))
                if jval is not None:
                    i, j = ch[ca], ch[cb]
                    out[(min(i, j), max(i, j))] = jval
    return out


if __name__ == "__main__":
    from mol_to_spin_system.shifts import make_test_mol_3d

    for smi, name in [("c1ccncc1", "pyridine"), ("c1ccoc1", "furan"),
                      ("c1ccsc1", "thiophene"), ("c1cc[nH]c1", "pyrrole")]:
        mol = make_test_mol_3d(smi)
        js = sorted(set(heteroaromatic_couplings(mol).values()))
        print(f"{name:>10}: {js} Hz")
