"""
model.evaluation.hungarian
==========================
Optimal spin-group assignment for permutation-invariant metrics. The G group
labels are arbitrary, so canonical-order element comparison understates accuracy;
Hungarian matching finds the permutation minimizing shift error and gives a
tighter, honest bound.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hungarian_perm"]


def hungarian_perm(pred_shifts: np.ndarray, tgt_shifts: np.ndarray) -> np.ndarray:
    """Per-sample permutation (B, G): perms[b, i] = the PRED index assigned to
    TARGET position i, so pred[b, perms[b]] aligns to the target ordering.
    cost[i, j] = |tgt[i] - pred[j]| (rows = target, cols = pred)."""
    from scipy.optimize import linear_sum_assignment
    B, G = pred_shifts.shape
    perms = np.zeros((B, G), dtype=int)
    for b in range(B):
        cost = np.abs(tgt_shifts[b, :, None] - pred_shifts[b, None, :])
        _, ci = linear_sum_assignment(cost)
        perms[b] = ci
    return perms
