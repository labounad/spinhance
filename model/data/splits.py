"""
model.data.splits
=================
Molecule-level 70/20/10 train/val/test splitter (master plan / DESIGN §8),
ported from the legacy package unchanged in behavior.

Guarantees:
  * Folds assigned at the MOLECULE level (all derived spectra stay together).
  * Two molecules forced into the same fold if they share a Bemis-Murcko scaffold
    or a near-duplicate canonical shift+J+degeneracy matrix.
  * Stratified by (degeneracy pattern, coupling-density bin).

Scaffold computation needs RDKit (imported lazily, only when a record lacks a
precomputed ``scaffold``). Canonical ordering / dedup / assignment are pure numpy.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["canonical_order", "reorder", "matrix_feature", "dedup_key",
           "bemis_murcko_scaffold", "make_splits"]


# ── Canonical ordering + dedup feature ─────────────────────────────────────────

def canonical_order(shifts, couplings, degeneracy):
    """Permutation sorting groups by shift desc, tie-broken by degeneracy desc,
    then |J| row-sum desc. Deterministic."""
    shifts = np.asarray(shifts, float)
    couplings = np.asarray(couplings, float)
    degeneracy = np.asarray(degeneracy, float)
    jrow = np.abs(couplings).sum(axis=1)
    return np.lexsort((-jrow, -degeneracy, -shifts))


def reorder(shifts, couplings, degeneracy, order):
    shifts = np.asarray(shifts, float)[order]
    degeneracy = np.asarray(degeneracy)[order]
    couplings = np.asarray(couplings, float)[np.ix_(order, order)]
    return shifts, couplings, degeneracy


def matrix_feature(shifts, couplings, degeneracy):
    """Canonical 1-D invariant [shifts | upper-tri J | deg] for near-dup detection."""
    order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    iu = np.triu_indices(len(s), 1)
    return np.concatenate([s, c[iu], d.astype(float)])


def dedup_key(shifts, couplings, degeneracy, shift_tol=0.02, j_tol=0.5):
    """Hashable key collapsing near-identical matrices onto a tolerance grid."""
    order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    iu = np.triu_indices(len(s), 1)
    s_q = np.round(s / shift_tol).astype(np.int64)
    j_q = np.round(c[iu] / j_tol).astype(np.int64)
    return (tuple(s_q.tolist()), tuple(j_q.tolist()), tuple(int(x) for x in d))


# ── Scaffold (RDKit, lazy) ─────────────────────────────────────────────────────

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
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


# ── Union-find ─────────────────────────────────────────────────────────────────

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


# ── Stratification ─────────────────────────────────────────────────────────────

def _stratum_key(rec, n_density_bins=4):
    deg = tuple(sorted((int(x) for x in rec["degeneracy"]), reverse=True))
    c = np.asarray(rec["couplings"], float)
    G = c.shape[0]
    n_off = G * (G - 1) / 2
    density = (np.abs(c[np.triu_indices(G, 1)]) > 0).sum() / max(n_off, 1)
    dbin = min(int(density * n_density_bins), n_density_bins - 1)
    return (deg, dbin)


# ── Main entry point ───────────────────────────────────────────────────────────

def make_splits(records, ratios=(0.7, 0.2, 0.1), seed=0,
                shift_tol=0.02, j_tol=0.5, compute_scaffold=True,
                drop_exact_dups=False):
    """Assign each molecule to 'train'/'val'/'test'. Returns (assignment, report)."""
    recs = list(records)
    n = len(recs)
    fold_names = ("train", "val", "test")
    assert abs(sum(ratios) - 1.0) < 1e-9 and len(ratios) == 3

    scaffolds, keys = [], []
    for r in recs:
        if "scaffold" in r and r["scaffold"] is not None:
            scaffolds.append(r["scaffold"])
        elif compute_scaffold:
            scaffolds.append(bemis_murcko_scaffold(r["smiles"]))
        else:
            scaffolds.append(None)
        keys.append(dedup_key(r["shifts"], r["couplings"], r["degeneracy"],
                              shift_tol, j_tol))

    keep = list(range(n))
    if drop_exact_dups:
        seen, keep = set(), []
        for i, k in enumerate(keys):
            if k not in seen:
                seen.add(k); keep.append(i)

    uf = _UF(n)
    by_scaffold = defaultdict(list)
    by_key = defaultdict(list)
    for i in keep:
        if scaffolds[i]:
            by_scaffold[scaffolds[i]].append(i)
        by_key[keys[i]].append(i)
    for members in list(by_scaffold.values()) + list(by_key.values()):
        for j in members[1:]:
            uf.union(members[0], j)

    groups = defaultdict(list)
    for i in keep:
        groups[uf.find(i)].append(i)
    group_list = list(groups.values())

    rng = np.random.default_rng(seed)

    def group_stratum(g):
        ks = [_stratum_key(recs[i]) for i in g]
        cnt = defaultdict(int)
        for k in ks:
            cnt[k] += 1
        return max(sorted(cnt), key=lambda k: cnt[k])

    strata = defaultdict(list)
    for g in group_list:
        strata[group_stratum(g)].append(g)

    total_keep = sum(len(g) for g in group_list)
    target = np.array(ratios) * total_keep
    fold_count = np.zeros(3)
    assignment = {}

    for stratum in sorted(strata.keys()):
        gs = strata[stratum]
        order = sorted(range(len(gs)), key=lambda idx: (-len(gs[idx]), rng.random()))
        for idx in order:
            g = gs[idx]
            size = len(g)
            deficit = (target - fold_count) / np.maximum(target, 1)
            f = int(np.argmax(deficit))
            fold_count[f] += size
            for i in g:
                assignment[recs[i]["mol_id"]] = fold_names[f]

    counts = {fn: int(fold_count[k]) for k, fn in enumerate(fold_names)}
    achieved = {fn: counts[fn] / max(total_keep, 1) for fn in fold_names}

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
        scaffold_leaks=scaffold_leaks, dup_matrix_leaks=dup_leaks, seed=seed,
    )
    return assignment, report
