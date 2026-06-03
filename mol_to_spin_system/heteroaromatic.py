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
    "pyridazine": {                  # 1,2-diazine: N1,N2; C3,4,5,6
        frozenset({3, 4}): 4.9, frozenset({5, 6}): 4.9,
        frozenset({4, 5}): 8.4,
        frozenset({3, 5}): 2.0, frozenset({4, 6}): 2.0,
        frozenset({3, 6}): 3.5,
    },
    "pyrimidine": {                  # 1,3-diazine: N1,N3; C2,4,5,6 (J24=J26=0 omitted)
        frozenset({4, 5}): 5.0, frozenset({5, 6}): 5.0,
        frozenset({4, 6}): 2.5,
        frozenset({2, 5}): 1.5,
    },
    # pyrazine (1,4-diazine): four equivalent H -> single magnetic-equivalence
    # group (no observable inter-proton coupling); handled by grouping, no table.
    "oxazole": {                     # 1,3-oxazole: O1,C2,N3,C4,C5
        frozenset({4, 5}): 0.8, frozenset({2, 5}): 0.5,   # J24 = 0 omitted
    },
    "isoxazole": {                   # 1,2-oxazole: O1,N2,C3,C4,C5
        frozenset({3, 4}): 1.7, frozenset({4, 5}): 1.8, frozenset({3, 5}): 0.3,
    },
    "thiazole": {                    # 1,3-thiazole: S1,C2,N3,C4,C5
        frozenset({4, 5}): 3.2, frozenset({2, 5}): 1.9,   # J24 = 0 omitted
    },
    "imidazole": {                   # 1,3-diazole: N1,C2,N3,C4,C5 (tautomer-averaged H4/H5)
        frozenset({4, 5}): 1.5, frozenset({2, 4}): 1.0, frozenset({2, 5}): 1.0,
    },
    "pyrazole": {                    # 1,2-diazole: N1,N2,C3,C4,C5
        frozenset({3, 4}): 2.1, frozenset({4, 5}): 2.1,   # J35 = 0 omitted
    },
}

#: Benzo-fused 5-membered heteroaromatics — the H2-H3 coupling across the
#: hetero ring, keyed by the ring's single heteroatom (Pretsch §5.6.2):
#: indole (N) 3.1, benzofuran (O) 2.5, benzothiophene (S) 5.5. The benzo ring's
#: ortho/meta couplings (~7.9/1.2) stay on the benzene fallback (close enough);
#: small cross-ring/peri couplings are not modeled (v1).
_FUSED5_J23 = {7: 3.1, 8: 2.5, 16: 5.5}


#: (ring size, sorted ((position, atomic-num), ...)) -> ring name, using the
#: canonical numbering from ``_canonical_positions`` (heteroatoms at lowest
#: locants; O<S<N seniority breaks ties, matching IUPAC).
_NAME: dict[tuple, str] = {
    (6, ((1, 7),)): "pyridine",
    (6, ((1, 7), (2, 7))): "pyridazine",
    (6, ((1, 7), (3, 7))): "pyrimidine",
    (5, ((1, 8),)): "furan",
    (5, ((1, 16),)): "thiophene",
    (5, ((1, 7),)): "pyrrole",
    (5, ((1, 8), (3, 7))): "oxazole",
    (5, ((1, 8), (2, 7))): "isoxazole",
    (5, ((1, 16), (3, 7))): "thiazole",
    (5, ((1, 7), (3, 7))): "imidazole",
    (5, ((1, 7), (2, 7))): "pyrazole",
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


#: heteroatom seniority for lowest-locant tiebreak (IUPAC: O before S before N)
_SENIORITY = {8: 0, 16: 1, 7: 2}


def _canonical_positions(mol: Chem.Mol, ring: tuple[int, ...]) -> dict[int, int] | None:
    """Number ring atoms 1..n giving heteroatoms the lowest locants (then O<S<N
    seniority) — IUPAC-style. Returns {atom_idx: position} or None."""
    order = _cycle_order(mol, ring)
    n = len(order)
    if n != len(ring):
        return None
    z = {a: mol.GetAtomWithIdx(a).GetAtomicNum() for a in order}
    best_key = None
    best_seq = None
    for start in range(n):
        for step in (1, -1):
            seq = [order[(start + step * j) % n] for j in range(n)]
            het_locants = tuple(p + 1 for p, a in enumerate(seq) if z[a] != 6)
            seniority = tuple(_SENIORITY.get(z[a], 9) for a in seq if z[a] != 6)
            key = (het_locants, seniority)
            if best_key is None or key < best_key:
                best_key, best_seq = key, seq
    return {a: p + 1 for p, a in enumerate(best_seq)}


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
        pos = _canonical_positions(mol, ring)
        if pos is None:
            continue
        het = tuple(sorted((pos[i], mol.GetAtomWithIdx(i).GetAtomicNum())
                           for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() != 6))
        name = _NAME.get((size, het))
        if name is not None:
            yield ring, name, pos


def _fused5_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """H2-H3 coupling across a benzo-fused 5-membered hetero ring (indole,
    benzofuran, benzothiophene). Couples adjacent protium-bearing carbons in a
    fused, single-heteroatom 5-ring (the fusion carbons bear no H, so this is
    the 2,3-pair). 2-heteroatom fused 5-rings (benzimidazole/indazole) have no
    such pair and are skipped."""
    out: dict[tuple[int, int], float] = {}
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) != 5 or not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if not any(ri.NumAtomRings(i) > 1 for i in ring):     # must be fused
            continue
        het = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring
               if mol.GetAtomWithIdx(i).GetAtomicNum() != 6]
        if len(het) != 1 or het[0] not in _FUSED5_J23:
            continue
        jval = _FUSED5_J23[het[0]]
        rs = set(ring)
        h_of: dict[int, int] = {}
        for c in ring:
            a = mol.GetAtomWithIdx(c)
            if a.GetAtomicNum() != 6:
                continue
            hs = [n.GetIdx() for n in a.GetNeighbors()
                  if n.GetAtomicNum() == 1 and n.GetIsotope() in (0, 1)]
            if len(hs) == 1:
                h_of[c] = hs[0]
        for c in h_of:
            for nb in mol.GetAtomWithIdx(c).GetNeighbors():
                d = nb.GetIdx()
                if d in rs and d > c and d in h_of:
                    i, j = h_of[c], h_of[d]
                    out[(min(i, j), max(i, j))] = jval
    return out


def heteroaromatic_couplings(mol: Chem.Mol) -> dict[tuple[int, int], float]:
    """Position-specific ring couplings for supported heteroaromatic rings.

    Returns {(atom_i, atom_j): J_Hz} with i < j. Overrides the benzene
    fallback for the rings it covers (it is merged AFTER ``aromatic_couplings``
    in ``coupling.all_couplings_typed``). Only protium C-H are coupled.
    Covers monocyclic rings (`_RING_J`) plus the H2-H3 coupling of benzo-fused
    5-membered hetero rings.
    """
    out: dict[tuple[int, int], float] = dict(_fused5_couplings(mol))
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
