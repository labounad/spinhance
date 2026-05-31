"""generate/spin_equivalence.py — ¹H spin-group analysis and molecule screening.

Spin-group counting (why this is non-trivial)
---------------------------------------------
The naive approach — count unique D-substitution canonical SMILES — correctly
handles homotopic protons (CH₃) but silently under-counts aromatic spin
systems.  Consider a 1,4-disubstituted benzene: the two Ha and Ha' protons
give *identical* substituted SMILES (they are enantiotopic by ring symmetry),
so the naive test folds them into one spin group.  But in MNova they must be
*separate* groups: J(Ha, Hx) ≠ J(Ha′, Hx) (one coupling is ortho, the other
is meta), making them magnetically inequivalent even though they are
chemically equivalent.  The same issue arises for any AA′BB′/AA′XX′ aromatic
system, mirrored ring methylenes, and vinyl protons.

Three-tier classification
-------------------------
To handle this correctly we assume every C-H proton is its own spin group
**by default**, then promote sets of protons to a single shared group only
when fast molecular rotation guarantees genuine magnetic equivalence:

* **HARD** — a complete methyl (CH₃), gem-dimethyl (6H), or tert-butyl (9H)
  unit whose rapid rotation makes all members truly magnetically equivalent.
  MNova collapses these into one group with ``number=N``, averaged shift, and
  averaged J.  Identified structurally (not from J-vectors), so it is robust.

* **SOFT** — protons that are chemically equivalent (same or enantiomeric
  D-substitution SMILES) but are kept as separate spin groups because magnetic
  equivalence is *not* guaranteed.  Each emits as its own group but they share
  a common averaged chemical shift.  Covers aromatic AA′BB′ pairs, CH₂
  methylenes, vinyl pairs, and mirrored ring protons.

* **NONE** — protons that are chemically distinct (different D-substitution
  SMILES even after enantiomer folding).  Each is its own group with its own
  unique shift.  Covers diastereotopic CH₂ protons, isolated sp³ CH, vinyl
  H-cis vs H-trans in asymmetric contexts, etc.

Spin-group count
----------------
Because SOFT and NONE protons each contribute exactly 1 spin group, and HARD
groups contribute exactly 1 spin group regardless of how many protons they
contain, the count is::

    n_groups = n_HARD_groups + n_non_HARD_protons

Role of the deuterium test
--------------------------
The 3-D deuterium substitution test determines chemical equivalence classes
(distinguishing SOFT from NONE), not the spin-group count.  The count itself
follows from the HARD/SOFT/NONE tier assignment above.  3-D embedding via
AssignStereochemistryFrom3D is still required so that diastereotopic CH₂
protons in chiral molecules are correctly assigned to NONE (they give
diastereomeric, not enantiomeric, substituted SMILES).

Public API
----------
Heuristic pre-filter:

- :func:`passes_heuristic` — fast atom-count check.

Core analysis:

- :func:`embed_3d` — 3-D conformer generation with MMFF/UFF fallback.
- :func:`strip_exchangeable_protons` — remove N-H, O-H, S-H atoms.
- :func:`equivalence_classes_by_substitution` — chemical equivalence groups.
- :func:`analyze_spin_systems` — spin-group count using the HARD/SOFT/NONE tiers.

Low-level (used by the viewer and tests):

- :func:`substitution_signature` — canonical SMILES after one H→D swap.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from .config import (
    EMBED_MAX_OPT_ITERS,
    EMBED_RANDOM_SEED,
    MAX_PROTON_BEARING_C,
    MIN_PROTONS,
)

_EXCHANGEABLE_PARENTS: frozenset[int] = frozenset({7, 8, 16})


# ── Heuristic pre-filter ──────────────────────────────────────────────────────

def _count_proton_bearing_carbons(mol: Chem.Mol) -> int:
    return sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() > 0
    )


def _count_ch_protons(mol: Chem.Mol) -> int:
    return sum(
        atom.GetTotalNumHs()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
    )


def passes_heuristic(
    mol: Chem.Mol,
    max_carbons: int = MAX_PROTON_BEARING_C,
    min_protons: int = MIN_PROTONS,
) -> tuple[bool, int, int]:
    """Return whether *mol* passes the fast proton-count pre-filter.

    A molecule passes when both conditions hold:

    * ``n_proton_bearing_c ≤ max_carbons`` — at most the target maximum number
      of CH carbons.  A molecule with more would need extreme symmetry to
      collapse to the target spin-group count.
    * ``n_protons ≥ min_protons`` — enough C-H protons to fill every group.

    *max_carbons* and *min_protons* default to the legacy single-target
    constants; a categorising scan passes the range bounds
    (``max_carbons=max_groups``, ``min_protons=min_groups``) so one pass keeps
    every molecule that could fall anywhere in the requested range.

    Returns ``(passes, n_proton_bearing_c, n_protons)``.
    """
    n_c = _count_proton_bearing_carbons(mol)
    n_h = _count_ch_protons(mol)
    return (n_c <= max_carbons) and (n_h >= min_protons), n_c, n_h


# ── 3-D conformer generation ──────────────────────────────────────────────────

def embed_3d(mol: Chem.Mol) -> tuple[Chem.Mol, bool]:
    """Return ``(mol_with_H, has_3d)`` — explicit H + MMFF/UFF conformer.

    Parameters
    ----------
    mol:
        Input molecule.  Explicit H are added internally.

    Returns
    -------
    mol_with_H:
        All H explicit.  Carries a MMFF94/UFF-minimised conformer when
        embedding succeeds; returned without a conformer on failure.
    has_3d:
        ``True`` when a valid 3-D conformer is present.  Pass this flag to
        :func:`substitution_signature` and
        :func:`equivalence_classes_by_substitution` so that stereo
        perception can degrade gracefully on embedding failure.

    Notes
    -----
    ETKDG v3 with ``useRandomCoords=True`` improves success on rigid
    fused-ring systems.  MMFF94 is preferred; UFF is the fallback.  Raw
    ETKDG geometry is retained when both force fields fail — it is still
    better than no 3-D information for stereochemistry perception.
    """
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed      = EMBED_RANDOM_SEED
    params.useRandomCoords = True

    if AllChem.EmbedMolecule(mol_h, params) != 0 or mol_h.GetNumConformers() == 0:
        return mol_h, False

    try:
        if AllChem.MMFFOptimizeMolecule(mol_h, maxIters=EMBED_MAX_OPT_ITERS) < 0:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=EMBED_MAX_OPT_ITERS)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=EMBED_MAX_OPT_ITERS)
        except Exception:
            pass

    return mol_h, mol_h.GetNumConformers() > 0


# ── Exchangeable-proton removal ───────────────────────────────────────────────

def strip_exchangeable_protons(mol_h: Chem.Mol) -> Chem.Mol:
    """Remove H atoms bonded to N, O, or S from a molecule with explicit H.

    N-H, O-H, and S-H protons exchange rapidly in deuterated solvents and are
    invisible to solution-state ¹H NMR.  Retaining them would (a) count
    phantom spin groups and (b) allow heteroatom geometry to perturb
    stereocentre assignment via
    :func:`~rdkit.Chem.AssignStereochemistryFrom3D`.

    The 3-D conformer (if present) is preserved after atom removal.
    """
    to_remove = [
        atom.GetIdx()
        for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() == 1
        and atom.GetNeighbors()[0].GetAtomicNum() in _EXCHANGEABLE_PARENTS
    ]
    rw = Chem.RWMol(mol_h)
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


# ── Stereo assignment ─────────────────────────────────────────────────────────

def _assign_stereo(mol: Chem.Mol, *, use_3d: bool) -> None:
    """Assign stereochemistry on *mol* in-place.

    When *use_3d* is ``True``, calls
    :func:`~rdkit.Chem.AssignStereochemistryFrom3D` to write chiral tags
    from 3-D atom coordinates before the standard CIP propagation step.
    This is the critical call that assigns the CHD stereocentre created by
    isotope substitution — the 2-D molecular graph cannot determine which
    face of a methylene the D occupies.

    Falls back to graph-based :func:`~rdkit.Chem.AssignStereochemistry`
    when 3-D is unavailable.
    """
    if use_3d and mol.GetNumConformers() > 0:
        conf_id = mol.GetConformer(0).GetId()
        try:
            Chem.AssignStereochemistryFrom3D(
                mol, confId=conf_id, replaceExistingTags=True
            )
        except TypeError:
            Chem.AssignStereochemistryFrom3D(mol, conf_id, True)

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def _invert_tetrahedral_stereochemistry(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy of *mol* with every tetrahedral chiral tag inverted.

    Used to generate the enantiomeric SMILES of a D-substituted molecule so
    that :func:`equivalence_classes_by_substitution` can detect whether two
    substituted structures are enantiomers (→ SOFT) vs diastereomers (→ NONE).
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    inverted = rw.GetMol()
    Chem.AssignStereochemistry(inverted, cleanIt=True, force=True)
    return inverted


# ── Substitution signatures ───────────────────────────────────────────────────

def substitution_signature(
    mol_h: Chem.Mol,
    atom_idx: int,
    *,
    use_3d: bool,
) -> str:
    """Canonical isomeric SMILES after replacing H at *atom_idx* with D.

    Two protons are homotopic iff this string is identical for both
    substitutions.  Different strings indicate enantiotopic or diastereotopic
    protons.

    Used by the viewer to build per-group deuterated representatives.  The
    pipeline uses :func:`equivalence_classes_by_substitution` instead, which
    also handles enantiomeric pairs.
    """
    sub = Chem.RWMol(mol_h)
    sub.GetAtomWithIdx(atom_idx).SetIsotope(2)
    sub_mol = sub.GetMol()
    _assign_stereo(sub_mol, use_3d=use_3d)
    return Chem.MolToSmiles(sub_mol, canonical=True, isomericSmiles=True)


def _both_signatures(
    mol_h: Chem.Mol,
    atom_idx: int,
    *,
    use_3d: bool,
) -> tuple[str, str]:
    """Return ``(forward_smi, enantiomer_smi)`` for H→D substitution.

    *forward_smi* is the canonical SMILES of the substituted molecule.
    *enantiomer_smi* is the canonical SMILES with every tetrahedral tag
    inverted.  Comparing enantiomer signatures across atom pairs allows
    :func:`equivalence_classes_by_substitution` to detect whether two
    substitutions produce enantiomers (enantiotopic → SOFT) or diastereomers
    (diastereotopic → NONE).
    """
    sub = Chem.RWMol(mol_h)
    sub.GetAtomWithIdx(atom_idx).SetIsotope(2)
    sub_mol = sub.GetMol()
    _assign_stereo(sub_mol, use_3d=use_3d)
    fwd = Chem.MolToSmiles(sub_mol, canonical=True, isomericSmiles=True)
    inv = Chem.MolToSmiles(
        _invert_tetrahedral_stereochemistry(sub_mol),
        canonical=True, isomericSmiles=True,
    )
    return fwd, inv


# ── Chemical equivalence classes ─────────────────────────────────────────────

def equivalence_classes_by_substitution(
    mol_h: Chem.Mol,
    *,
    use_3d: bool,
    merge_enantiotopic: bool = True,
    candidate_atoms: list[int] | None = None,
) -> list[list[int]]:
    """Group C-H protons by chemical equivalence via the deuterium test.

    Parameters
    ----------
    mol_h:
        Molecule with explicit H and an optional 3-D conformer.  May contain
        exchangeable protons (N-H, O-H, S-H) — they are preserved as
        structural context and are NOT substituted, but their presence in the
        molecule ensures they correctly break apparent ring symmetry during
        the test.  Pass the full output of :func:`embed_3d` rather than the
        stripped output of :func:`strip_exchangeable_protons`.
    use_3d:
        Whether to use :func:`~rdkit.Chem.AssignStereochemistryFrom3D`.
        Pass the ``has_3d`` flag returned by :func:`embed_3d`.
    merge_enantiotopic:
        When ``True`` (default), protons whose substituted structures differ
        only as enantiomers are placed in the same class (→ SOFT: same
        averaged chemical shift, separate spin groups).  When ``False``,
        only homotopic protons are grouped (every enantiotopic pair is its
        own class).
    candidate_atoms:
        Explicit list of H atom indices to test.  When provided, only these
        atoms are included in the output classes; other H atoms (e.g.
        exchangeable N-H, O-H) remain in the molecule as structural context
        but are not classified.  When ``None``, all H atoms are tested.

    Returns
    -------
    list of lists of int
        Each inner list is a chemical-equivalence class sorted by atom index.
        The outer list is sorted by the smallest atom index in each class.

    Notes
    -----
    **Why exchangeable protons must stay in the molecule** (not be stripped
    first): removing an indole N-H before the test makes the two junction
    carbons of the indole 5-membered ring appear equivalent, which propagates
    a false symmetry to the aromatic CH protons on the 6-membered ring — they
    appear enantiotopic (SOFT) instead of chemically distinct (NONE).  By
    keeping N-H in the molecule and passing only C-H atoms as
    ``candidate_atoms``, the structural asymmetry is preserved.
    """
    h_idxs: list[int] = (
        candidate_atoms
        if candidate_atoms is not None
        else [atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() == 1]
    )
    if not h_idxs:
        return []

    fwd: dict[int, str] = {}
    inv: dict[int, str] = {}
    for idx in h_idxs:
        fwd[idx], inv[idx] = _both_signatures(mol_h, idx, use_3d=use_3d)

    # Union-Find
    parent = {i: i for i in h_idxs}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for i, a in enumerate(h_idxs):
        for b in h_idxs[i + 1:]:
            homotopic    = fwd[a] == fwd[b]
            enantiomeric = (fwd[a] == inv[b]) or (inv[a] == fwd[b])
            if homotopic or (merge_enantiotopic and enantiomeric):
                union(a, b)

    by_root: dict[int, list[int]] = {}
    for idx in h_idxs:
        by_root.setdefault(find(idx), []).append(idx)

    return sorted(
        (sorted(grp) for grp in by_root.values()),
        key=lambda g: g[0],
    )


# ── HARD/SOFT/NONE classification ────────────────────────────────────────────

def _hydrogen_neighbors(atom: Chem.Atom) -> list[int]:
    """Sorted list of H atom indices directly bonded to *atom*."""
    return sorted(
        n.GetIdx() for n in atom.GetNeighbors()
        if n.GetAtomicNum() == 1
    )


def _is_magnetically_equivalent(
    cls: list[int],
    fwd_sigs: dict[int, str],
    mol_h: Chem.Mol,
    all_candidate_h: list[int],
) -> bool:
    """Return ``True`` iff every member of *cls* is both homotopic and
    magnetically equivalent to every other member.

    Two C-H protons are magnetically equivalent iff:

    1. **Homotopic** — they produce *identical* (not merely enantiomeric)
       D-substitution canonical SMILES.  This rules out enantiotopic pairs
       (which give enantiomeric SMILES) and diastereotopic pairs immediately.

    2. **Same coupling profile** — every H atom outside *cls* is at the same
       shortest-bond-path distance from every member of *cls*.  Equal distances
       imply equal *J* couplings (since *J* falls off steeply with path length),
       so the full coupling pattern is identical for every class member.

    This single test subsumes the old methyl-rotor check while also handling
    symmetric aromatic protons (e.g. the two equivalent H's in a
    1,3,5-trisubstituted benzene that share a C₂ rotation axis).

    Contrast with toluene's two ortho protons: they are homotopic (condition 1
    passes) but one ortho-H sees the meta-H three bonds away while the other
    sees it five bonds away (condition 2 fails) → correctly classified SOFT.
    """
    # Condition 1: all members produce the same forward SMILES (homotopic).
    first = fwd_sigs[cls[0]]
    if not all(fwd_sigs[h] == first for h in cls[1:]):
        return False

    # Condition 2: identical distance profile to all external H atoms.
    cls_set  = set(cls)
    ext_h    = [h for h in all_candidate_h if h not in cls_set]
    if not ext_h:
        return True   # no external H → trivially magnetically equivalent

    ref      = cls[0]
    ref_prof = tuple(
        len(Chem.GetShortestPath(mol_h, ref, e)) - 1 for e in ext_h
    )
    for member in cls[1:]:
        prof = tuple(
            len(Chem.GetShortestPath(mol_h, member, e)) - 1 for e in ext_h
        )
        if prof != ref_prof:
            return False
    return True


def _classify_equivalence_classes(
    mol_h: Chem.Mol,
    *,
    use_3d: bool,
    candidate_atoms: list[int],
) -> list[tuple[str, list[int]]]:
    """Group *candidate_atoms* into HARD/SOFT/NONE spin-group classes.

    Returns a list of ``(tier, atom_indices)`` pairs ordered by the smallest
    atom index in each class.

    Algorithm
    ---------
    1. Compute the forward and enantiomer D-substitution SMILES for every
       candidate H atom.
    2. Union-Find groups homotopic pairs (identical forward SMILES) and —
       with ``merge_enantiotopic`` logic — enantiotopic pairs (one's forward
       SMILES equals the other's enantiomer SMILES) into the same class.
    3. Each class is then tiered:

       * ``NONE``  — singleton (unique chemical environment).
       * ``HARD``  — all members are homotopic *and* magnetically equivalent
                     (see :func:`_is_magnetically_equivalent`).
       * ``SOFT``  — all other multi-member classes (enantiotopic, or homotopic
                     but magnetically inequivalent like toluene's ortho-H pair).
    """
    if not candidate_atoms:
        return []

    fwd: dict[int, str] = {}
    inv: dict[int, str] = {}
    for idx in candidate_atoms:
        fwd[idx], inv[idx] = _both_signatures(mol_h, idx, use_3d=use_3d)

    # Union-Find — merge homotopic and enantiotopic pairs.
    parent = {i: i for i in candidate_atoms}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for i, a in enumerate(candidate_atoms):
        for b in candidate_atoms[i + 1:]:
            if fwd[a] == fwd[b] or fwd[a] == inv[b] or inv[a] == fwd[b]:
                union(a, b)

    by_root: dict[int, list[int]] = {}
    for idx in candidate_atoms:
        by_root.setdefault(find(idx), []).append(idx)
    classes = sorted(
        (sorted(g) for g in by_root.values()), key=lambda g: g[0]
    )

    # Determine tier for each class.
    result: list[tuple[str, list[int]]] = []
    for cls in classes:
        if len(cls) == 1:
            tier = "NONE"
        elif _is_magnetically_equivalent(cls, fwd, mol_h, candidate_atoms):
            tier = "HARD"
        else:
            tier = "SOFT"
        result.append((tier, cls))

    return result


def _is_rotationally_hard_hydrogen_class(
    class_atoms: list[int],
    mol_h: Chem.Mol,
) -> bool:
    """Return ``True`` iff *class_atoms* forms one or more complete methyl groups.

    A HARD class consists of CH₃ groups (or sets thereof) where rapid
    rotation about the C-C bond makes every proton genuinely magnetically
    equivalent.  Valid patterns include CH₃ (3H), gem-dimethyl (6H),
    tert-butyl (9H), and neopentyl-like (12H).

    The invariant checked: every H in the class lives on a carbon with
    exactly three H neighbours (a methyl carbon), and *all* three H on
    each such carbon are present in the class.  A partial set (e.g. only
    2 of 3 methyl H) is NOT HARD — this prevents accidental collapsing
    of prochiral groups.
    """
    if len(class_atoms) < 3 or len(class_atoms) % 3 != 0:
        return False

    parent_to_class_hs: dict[int, list[int]] = {}
    for idx in class_atoms:
        atom = mol_h.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 1:
            return False
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            return False
        parent = neighbors[0]
        if parent.GetAtomicNum() != 6:
            return False
        parent_to_class_hs.setdefault(parent.GetIdx(), []).append(idx)

    for parent_idx, hs_in_class in parent_to_class_hs.items():
        all_hs = _hydrogen_neighbors(mol_h.GetAtomWithIdx(parent_idx))
        if len(all_hs) != 3 or set(all_hs) != set(hs_in_class):
            return False

    return True


# ── Main analysis entry point ─────────────────────────────────────────────────

def analyze_spin_systems(mol: Chem.Mol) -> tuple[int, list[int]]:
    """Count magnetically distinct ¹H spin groups using the HARD/SOFT/NONE tiers.

    Parameters
    ----------
    mol:
        RDKit molecule.  Explicit H and stereo annotations are optional.

    Returns
    -------
    n_groups : int
        Number of spin groups.
    group_sizes : list[int]
        Proton count per spin group, sorted descending.

    Notes
    -----
    **Counting rule** (the key change vs. the naive D-substitution count):

    Every C-H proton is its own spin group *by default*, with one exception:
    a complete methyl (or tert-butyl) rotor is counted as a **single** HARD
    group.  All other protons — whether chemically equivalent (SOFT: same
    averaged shift but separate groups) or distinct (NONE: individual shift)
    — each contribute exactly 1 to the group count.

    Concretely::

        n_groups = n_HARD_groups + n_non_HARD_protons

    This correctly handles aromatic AA′BB′ systems: the two chemically
    equivalent ortho protons of a monosubstituted benzene are enantiotopic
    (SOFT), so they are merged into one chemical-equivalence class for shift
    averaging but counted as **two** separate spin groups.

    The deuterium test (with ``merge_enantiotopic=True``) provides the
    chemical-equivalence classes used to assign SOFT vs NONE tiers, and to
    determine the averaged shift.  It does **not** set the spin-group count.
    """
    mol_h, use_3d = embed_3d(mol)

    # Keep exchangeable protons (N-H, O-H, S-H) in the molecule so they
    # act as structural context during the D-substitution test.  Only
    # non-exchangeable C-H atoms are tested as spin-group candidates.
    candidate_h = [
        a.GetIdx() for a in mol_h.GetAtoms()
        if a.GetAtomicNum() == 1
        and a.GetNeighbors()[0].GetAtomicNum() not in _EXCHANGEABLE_PARENTS
    ]

    classified = _classify_equivalence_classes(
        mol_h, use_3d=use_3d, candidate_atoms=candidate_h,
    )

    group_sizes: list[int] = []
    for tier, cls in classified:
        if tier == "HARD":
            group_sizes.append(len(cls))
        else:
            group_sizes.extend([1] * len(cls))

    group_sizes.sort(reverse=True)
    return len(group_sizes), group_sizes


# ── Atom-level classification (used by the viewer) ────────────────────────────

from dataclasses import dataclass as _dataclass  # noqa: E402


def _index_to_excel_letters(idx: int) -> str:
    """0 → 'A', 25 → 'Z', 26 → 'AA', … (bijective base-26)."""
    letters: list[str] = []
    n = idx
    while True:
        letters.append(chr(ord("A") + n % 26))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(letters))


@_dataclass(frozen=True)
class SpinGroup:
    """Atom-level record for one ¹H spin group, used by the viewer.

    Attributes
    ----------
    label:
        Excel-style letter (A, B, C …).
    tier:
        ``'HARD'`` — complete methyl/t-Bu rotor (single spin group, N protons).
        ``'SOFT'`` — chemically equivalent but magnetically inequivalent
        (each atom is its own spin group; they share one averaged shift).
        ``'NONE'`` — chemically distinct singleton.
    h_indices:
        Atom indices of the H atoms that belong to *this* spin group,
        within ``mol_h`` (output of :func:`strip_exchangeable_protons`).
        A HARD group contains N indices; SOFT and NONE groups contain one.
    class_h_indices:
        Atom indices of *all* H atoms in the chemical-equivalence class.
        For HARD and NONE groups this equals ``h_indices``.  For SOFT groups
        multiple ``SpinGroup`` objects share the same ``class_h_indices``
        — they use one averaged chemical shift in the XML emitter.
    heavy_parent_indices:
        Heavy-atom indices (in the original mol, same numbering as mol_h)
        whose H atoms belong to this spin group.  Used by the viewer to
        annotate and highlight atoms in the 2-D structure.
    """

    label: str
    tier: str
    h_indices: tuple[int, ...]
    class_h_indices: tuple[int, ...]
    heavy_parent_indices: tuple[int, ...]


def classify_spin_groups(mol: Chem.Mol) -> tuple[Chem.Mol, list[SpinGroup]]:
    """Return ``(mol_h, groups)`` — atom-level spin-group classification.

    Parameters
    ----------
    mol:
        Input molecule (explicit H optional).

    Returns
    -------
    mol_h:
        The molecule with explicit H and a 3-D conformer (if embedding
        succeeded).  Exchangeable protons are retained as structural context.
        Heavy-atom indices match the original *mol*.
    groups:
        One :class:`SpinGroup` per spin group, in label order (A, B, C …).
        The group count equals ``analyze_spin_systems(mol)[0]``.

    Notes
    -----
    Exchangeable protons (N-H, O-H, S-H) are kept in *mol_h* so that the
    D-substitution test sees the full structural context.  They are excluded
    from ``candidate_atoms`` so they are never classified as spin groups.
    """
    mol_h, use_3d = embed_3d(mol)

    # Non-exchangeable C-H protons only; exchangeable H stay for context.
    candidate_h = [
        a.GetIdx() for a in mol_h.GetAtoms()
        if a.GetAtomicNum() == 1
        and a.GetNeighbors()[0].GetAtomicNum() not in _EXCHANGEABLE_PARENTS
    ]

    h_to_heavy: dict[int, int] = {
        h: mol_h.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
        for h in candidate_h
    }

    classified = _classify_equivalence_classes(
        mol_h, use_3d=use_3d, candidate_atoms=candidate_h,
    )

    groups: list[SpinGroup] = []
    label_idx = 0

    for tier, cls in classified:
        class_tuple = tuple(cls)
        if tier == "HARD":
            heavies = tuple(sorted({h_to_heavy[h] for h in cls}))
            groups.append(SpinGroup(
                label              = _index_to_excel_letters(label_idx),
                tier               = "HARD",
                h_indices          = class_tuple,
                class_h_indices    = class_tuple,
                heavy_parent_indices = heavies,
            ))
            label_idx += 1
        else:
            for h_idx in cls:
                groups.append(SpinGroup(
                    label              = _index_to_excel_letters(label_idx),
                    tier               = tier,
                    h_indices          = (h_idx,),
                    class_h_indices    = class_tuple,
                    heavy_parent_indices = (h_to_heavy[h_idx],),
                ))
                label_idx += 1

    return mol_h, groups
