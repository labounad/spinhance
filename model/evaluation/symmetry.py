"""
model.evaluation.symmetry
=========================
Label-invariant coupling comparison.

The spin-system graph carries **arbitrary labels** for chemically-equivalent groups.
The canonical shift-sort orders nodes by shift, but among nodes with **equal shift**
(chemical equivalence — e.g. the A,A' and X,X' of a para-ring's AA'XX' system) the
order is an **arbitrary tie-break**, chosen independently for the prediction and for the
target. Shift-MAE is unaffected (the tied shifts are equal), but the coupling matrix's
rows/columns get permuted, so an element-wise J-MAE unfairly penalizes a prediction that
happened to break a tie the other way — even when its predicted spectrum is identical
(the AA'XX' case where swapping A<->A' flips the ortho/para couplings: 0.21 vs 2.59 Hz
for the same spectrum).

Fix: make J-MAE invariant to that tie-breaking by aligning the prediction to the target
over the group of within-class label permutations before scoring:

    J-MAE = min over π (permuting only equal-shift+degeneracy nodes) of
            masked | π(pred_couplings) − target_couplings |

This is the coupling analogue of the canonical shift-sort that already makes shift-MAE
label-invariant. It is fair (it only ever re-labels nodes the sort itself left ambiguous)
and does not over-credit: a permutation can only *rearrange* coupling values, never change
them, so a prediction with genuinely wrong couplings cannot be relabeled into a match.

Only **exactly-equal** shifts are grouped (chemical equivalence is exact by construction;
``shift_tol`` just guards float noise). Genuine *accidental* near-degeneracies (distinct
shifts a few hundredths apart) keep a deterministic, learnable sort order and are NOT
grouped — matching the soft-equivalence philosophy (symmetry, not shift proximity).
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def label_permutations(shifts, deg, *, shift_tol=1e-3, max_perms=720):
    """Permutations of node indices that only reorder nodes sharing (shift, degeneracy)
    — i.e. the canonical sort's arbitrary tie-break group. Always includes the identity.

    Args:
      shifts: (G,) chemical shifts (ppm).
      deg:    (G,) degeneracy per group.
      shift_tol: nodes whose shifts agree within this (ppm) are tie-broken (grouped).
      max_perms: bound on the search; pathological symmetry -> identity only.
    Returns: list of (G,) int permutation arrays.
    """
    shifts = np.asarray(shifts, float); deg = np.asarray(deg)
    G = len(shifts)
    ident = np.arange(G)

    classes = {}
    for i in range(G):
        k = (round(float(shifts[i]) / shift_tol), int(deg[i]))
        classes.setdefault(k, []).append(i)
    multi = [idx for idx in classes.values() if len(idx) > 1]
    if not multi:
        return [ident]

    total = 1
    for idx in multi:
        total *= math.factorial(len(idx))
        if total > max_perms:                       # too degenerate to enumerate cheaply
            return [ident]

    perms = []
    per_class = [list(itertools.permutations(idx)) for idx in multi]
    for combo in itertools.product(*per_class):
        p = ident.copy()
        for idx, order in zip(multi, combo):
            p[np.asarray(idx)] = np.asarray(order)
        perms.append(p)
    return perms


def align_pred_couplings(pred_cm, tgt_cm, tgt_mask, shifts, deg, **kw):
    """Relabel ``pred_cm`` (G, G) by the within-class label permutation minimizing the
    masked coupling discrepancy to ``tgt_cm``. Returns the aligned prediction (G, G).

    ``shifts``/``deg`` define the tie-break classes and come from the TARGET; ``tgt_mask``
    is the (G, G) ground-truth coupling presence mask. Identity is always a candidate, so
    the result is never worse than the unaligned prediction."""
    perms = label_permutations(shifts, deg, **kw)
    if len(perms) == 1:
        return pred_cm
    pred_cm = np.asarray(pred_cm, float)
    tgt_cm = np.asarray(tgt_cm, float); tgt_mask = np.asarray(tgt_mask, float)
    best, best_err = pred_cm, np.inf
    for p in perms:
        cand = pred_cm[np.ix_(p, p)]
        err = float((np.abs(cand - tgt_cm) * tgt_mask).sum())
        if err < best_err:
            best_err, best = err, cand
    return best
