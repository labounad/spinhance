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


def passes_heuristic(mol: Chem.Mol) -> tuple[bool, int, int]:
    """Return whether *mol* passes the fast proton-count pre-filter.

    A molecule passes when both conditions hold:

    * ``n_proton_bearing_c ≤ MAX_PROTON_BEARING_C`` — at most the target
      number of CH carbons.  A molecule with more would need extreme symmetry
      to reach exactly the target spin-group count.
    * ``n_protons ≥ MIN_PROTONS`` — enough C-H protons to fill every group.

    Returns ``(passes, n_proton_bearing_c, n_protons)``.
    """
    n_c = _count_proton_bearing_carbons(mol)
    n_h = _count_ch_protons(mol)
    return (n_c <= MAX_PROTON_BEARING_C) and (n_h >= MIN_PROTONS), n_c, n_h


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
) -> list[list[int]]:
    """Group C-H protons by chemical equivalence via the deuterium test.

    Parameters
    ----------
    mol_h:
        Molecule with explicit H (output of :func:`strip_exchangeable_protons`).
        Should carry a 3-D conformer when available.
    use_3d:
        Whether to use :func:`~rdkit.Chem.AssignStereochemistryFrom3D`.
        Pass the ``has_3d`` flag returned by :func:`embed_3d`.
    merge_enantiotopic:
        When ``True`` (default), protons whose substituted structures differ
        only as enantiomers are placed in the same class (→ SOFT: same
        averaged chemical shift, separate spin groups).  When ``False``,
        only homotopic protons are grouped (every enantiotopic pair is its
        own class).

    Returns
    -------
    list of lists of int
        Each inner list is a chemical-equivalence class sorted by atom index.
        The outer list is sorted by the smallest atom index in each class.

    Notes
    -----
    Enantiomeric detection: for each H→D substituted molecule, we also
    compute the SMILES with all tetrahedral chiral tags inverted (the
    "enantiomer SMILES").  Two protons belong to the same SOFT class iff
    one's forward SMILES matches the other's enantiomer SMILES (or vice
    versa).  This correctly identifies enantiotopic protons in achiral
    molecules while leaving diastereotopic protons in separate classes.
    """
    h_idxs = [
        atom.GetIdx() for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() == 1
    ]
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


# ── HARD-group detection ──────────────────────────────────────────────────────

def _hydrogen_neighbors(atom: Chem.Atom) -> list[int]:
    """Sorted list of H atom indices directly bonded to *atom*."""
    return sorted(
        n.GetIdx() for n in atom.GetNeighbors()
        if n.GetAtomicNum() == 1
    )


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
    mol_h = strip_exchangeable_protons(mol_h)

    classes = equivalence_classes_by_substitution(
        mol_h, use_3d=use_3d, merge_enantiotopic=True
    )

    group_sizes: list[int] = []
    for cls in classes:
        if _is_rotationally_hard_hydrogen_class(cls, mol_h):
            # HARD: one spin group containing all N protons.
            group_sizes.append(len(cls))
        else:
            # SOFT (size > 1) or NONE (size == 1): each proton is its own group.
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
        succeeded), after exchangeable protons have been stripped.  Atom
        indices of heavy atoms match the original *mol*.
    groups:
        One :class:`SpinGroup` per spin group, in label order (A, B, C …).
        The group count equals ``analyze_spin_systems(mol)[0]``.

    Notes
    -----
    Heavy-atom indices in *mol_h* are identical to those in the original
    *mol* because :func:`embed_3d` appends H after all heavy atoms and
    :func:`strip_exchangeable_protons` only removes H atoms.
    """
    mol_h, use_3d = embed_3d(mol)
    mol_h = strip_exchangeable_protons(mol_h)

    # H atom idx → parent heavy atom idx (stable because H indices > heavy)
    h_to_heavy: dict[int, int] = {
        atom.GetIdx(): atom.GetNeighbors()[0].GetIdx()
        for atom in mol_h.GetAtoms()
        if atom.GetAtomicNum() == 1
    }

    classes = equivalence_classes_by_substitution(
        mol_h, use_3d=use_3d, merge_enantiotopic=True
    )

    groups: list[SpinGroup] = []
    label_idx = 0

    for cls in classes:
        class_tuple = tuple(cls)
        if _is_rotationally_hard_hydrogen_class(cls, mol_h):
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
            tier = "SOFT" if len(cls) > 1 else "NONE"
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
