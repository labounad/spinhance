"""
model.splits
===============
Molecule-level 70/20/10 train/val/test splitter for Task 4 (Decision 8).

Guarantees / design (see model/DESIGN.md §8):
  * Folds are assigned at the MOLECULE level, so every spectrum derived from a
    molecule (90 + 600 MHz + augmentations) lands in the same fold. No 90/600
    leakage.
  * Two molecules are forced into the SAME fold if they share either
      - a Bemis-Murcko scaffold (structural near-relatives), or
      - a near-duplicate shift+J+degeneracy matrix (after canonical ordering),
    because the model only ever sees the spectrum/matrix and near-identical
    systems straddling folds would inflate the test score.
  * Assignment is stratified by (degeneracy pattern, coupling-density bin) so
    those regimes are balanced across folds.

Scaffold computation needs RDKit; it is imported lazily and only when a record
lacks a precomputed ``scaffold``. Everything else (canonical ordering, dedup,
stratified assignment) is pure numpy and unit-tested without RDKit.

A "record" is a dict:
    {
      "mol_id":     str,                # unique per molecule
      "shifts":     (G,) array  ppm,
      "couplings":  (G, G) array Hz,    # symmetric, 0 diagonal
      "degeneracy": (G,) int array,
      "smiles":     str,                # optional if "scaffold" given
      "scaffold":   str,                # optional; else computed from smiles
    }
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["canonical_order", "matrix_feature", "dedup_key",
           "bemis_murcko_scaffold", "make_splits"]


# -----------------------------------------------------------------------------
# Canonical ordering (Decision 3) and dedup feature
# -----------------------------------------------------------------------------

def canonical_order(shifts, couplings, degeneracy):
    """Return the permutation that sorts groups by shift desc, tie-broken by
    degeneracy desc, then |J| row-sum desc. Deterministic."""
    shifts = np.asarray(shifts, float)
    couplings = np.asarray(couplings, float)
    degeneracy = np.asarray(degeneracy, float)
    jrow = np.abs(couplings).sum(axis=1)
    # np.lexsort sorts by the LAST key primary; negate for descending.
    order = np.lexsort((-jrow, -degeneracy, -shifts))
    return order


def reorder(shifts, couplings, degeneracy, order):
    shifts = np.asarray(shifts, float)[order]
    degeneracy = np.asarray(degeneracy)[order]
    couplings = np.asarray(couplings, float)[np.ix_(order, order)]
    return shifts, couplings, degeneracy


def matrix_feature(shifts, couplings, degeneracy):
    """Canonical 1-D invariant of the spin system: [shifts | upper-tri J | deg]
    after canonical ordering. Used for near-duplicate detection."""
    order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    iu = np.triu_indices(len(s), 1)
    return np.concatenate([s, c[iu], d.astype(float)])


def dedup_key(shifts, couplings, degeneracy,
              shift_tol=0.02, j_tol=0.5):
    """Hashable key that collapses near-identical matrices into one bucket.

    Rounds the canonical feature to the tolerance grid (shifts in ppm, J in Hz)
    and keeps degeneracy exact. O(1) per molecule -> O(N) dedup overall.
    NOTE: rounding can split a true near-dup across a bin edge; for a harder
    guarantee, swap this for threshold clustering / LSH (left as an upgrade).
    """
    order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    iu = np.triu_indices(len(s), 1)
    s_q = np.round(s / shift_tol).astype(np.int64)
    j_q = np.round(c[iu] / j_tol).astype(np.int64)
    return (tuple(s_q.tolist()), tuple(j_q.tolist()), tuple(int(x) for x in d))


# -----------------------------------------------------------------------------
# Scaffold (RDKit, lazy)
# -----------------------------------------------------------------------------

def bemis_murcko_scaffold(smiles):
    """Bemis-Murcko scaffold SMILES via RDKit. Raises if RDKit is missing."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "RDKit required for scaffold computation; install it or precompute "
            "a 'scaffold' field on each record."
        ) from e
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""  # unparseable -> its own degenerate scaffold bucket below
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


# -----------------------------------------------------------------------------
# Union-find
# -----------------------------------------------------------------------------

class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


# -----------------------------------------------------------------------------
# Stratification key
# -----------------------------------------------------------------------------

def _stratum_key(rec, n_density_bins=4):
    """(degeneracy pattern, coupling-density bin) — what we balance across folds."""
    deg = tuple(sorted((int(x) for x in rec["degeneracy"]), reverse=True))
    c = np.asarray(rec["couplings"], float)
    G = c.shape[0]
    n_off = G * (G - 1) / 2
    density = (np.abs(c[np.triu_indices(G, 1)]) > 0).sum() / max(n_off, 1)
    dbin = min(int(density * n_density_bins), n_density_bins - 1)
    return (deg, dbin)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def make_splits(records, ratios=(0.7, 0.2, 0.1), seed=0,
                shift_tol=0.02, j_tol=0.5, compute_scaffold=True,
                drop_exact_dups=False):
    """Assign each molecule to 'train' / 'val' / 'test'.

    Returns (assignment, report):
      assignment: {mol_id: fold}
      report:     dict of summary stats (counts, ratios, #groups, leakage checks)

    Grouping: molecules are merged (must share a fold) if they share a scaffold
    or a near-duplicate matrix. Atomic groups are then assigned to folds with a
    stratified, size-aware greedy rule to hit the target ratios.
    """
    recs = list(records)
    n = len(recs)
    fold_names = ("train", "val", "test")
    assert abs(sum(ratios) - 1.0) < 1e-9 and len(ratios) == 3

    # scaffolds + dedup keys
    scaffolds = []
    keys = []
    for r in recs:
        if "scaffold" in r and r["scaffold"] is not None:
            scaffolds.append(r["scaffold"])
        elif compute_scaffold:
            scaffolds.append(bemis_murcko_scaffold(r["smiles"]))
        else:
            scaffolds.append(None)
        keys.append(dedup_key(r["shifts"], r["couplings"], r["degeneracy"],
                              shift_tol, j_tol))

    # optional: drop exact dup matrices (keep first occurrence)
    keep = list(range(n))
    if drop_exact_dups:
        seen, keep = set(), []
        for i, k in enumerate(keys):
            if k not in seen:
                seen.add(k); keep.append(i)

    # union-find merges by shared scaffold and by shared dedup key
    uf = _UF(n)
    by_scaffold = defaultdict(list)
    by_key = defaultdict(list)
    for i in keep:
        if scaffolds[i]:                      # empty/None scaffold -> stays singleton
            by_scaffold[scaffolds[i]].append(i)
        by_key[keys[i]].append(i)
    for members in list(by_scaffold.values()) + list(by_key.values()):
        for j in members[1:]:
            uf.union(members[0], j)

    # gather atomic groups
    groups = defaultdict(list)
    for i in keep:
        groups[uf.find(i)].append(i)
    group_list = list(groups.values())

    # stratified, size-aware greedy assignment
    rng = np.random.default_rng(seed)
    # representative stratum per group = most common member stratum
    def group_stratum(g):
        ks = [_stratum_key(recs[i]) for i in g]
        # pick deterministically: the lexicographically smallest most-common key
        cnt = defaultdict(int)
        for k in ks:
            cnt[k] += 1
        best = max(sorted(cnt), key=lambda k: cnt[k])
        return best

    strata = defaultdict(list)
    for g in group_list:
        strata[group_stratum(g)].append(g)

    total_keep = sum(len(g) for g in group_list)
    target = np.array(ratios) * total_keep
    fold_count = np.zeros(3)
    assignment = {}

    for stratum in sorted(strata.keys()):
        gs = strata[stratum]
        # within a stratum, big groups first (less room to balance later), random tie-break
        order = sorted(range(len(gs)),
                       key=lambda idx: (-len(gs[idx]), rng.random()))
        for idx in order:
            g = gs[idx]
            size = len(g)
            # assign to the fold furthest below its global target share
            deficit = (target - fold_count) / np.maximum(target, 1)
            f = int(np.argmax(deficit))
            fold_count[f] += size
            for i in g:
                assignment[recs[i]["mol_id"]] = fold_names[f]

    # ---- report + leakage self-checks ----
    counts = {fn: int(fold_count[k]) for k, fn in enumerate(fold_names)}
    achieved = {fn: counts[fn] / max(total_keep, 1) for fn in fold_names}

    # no scaffold straddles folds; no dedup key straddles folds
    scaf_folds = defaultdict(set)
    key_folds = defaultdict(set)
    for i in keep:
        mid = recs[i]["mol_id"]
        if mid not in assignment:
            continue
        if scaffolds[i]:
            scaf_folds[scaffolds[i]].add(assignment[mid])
        key_folds[keys[i]].add(assignment[mid])
    scaffold_leaks = sum(1 for v in scaf_folds.values() if len(v) > 1)
    dup_leaks = sum(1 for v in key_folds.values() if len(v) > 1)

    report = dict(
        n_molecules=n, n_kept=total_keep, n_groups=len(group_list),
        n_strata=len(strata), counts=counts, ratios=achieved,
        target_ratios=dict(zip(fold_names, ratios)),
        scaffold_leaks=scaffold_leaks, dup_matrix_leaks=dup_leaks,
        seed=seed,
    )
    return assignment, report
