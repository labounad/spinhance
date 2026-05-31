"""
model.metrics
================
Torch-free decode + evaluation metrics (kept numpy so they're unit-tested and
shared as the eval oracle). The train loop passes detached predictions as numpy.

decode(): standardized/logit model outputs -> physical matrix
          (shifts ppm, couplings Hz w/ presence threshold, degeneracy values).
compute_metrics(): physical-unit errors vs the (standardized) targets.
wasserstein1_np(): eval-side 1-D Wasserstein (mirrors losses.wasserstein1).
"""

from __future__ import annotations

import numpy as np

__all__ = ["decode", "compute_metrics", "wasserstein1_np"]

# Cache triu_indices per G value — avoids recomputation on every validation batch
_TRIU_CACHE: dict[int, tuple] = {}


def _hungarian_perm(pred_shifts: np.ndarray, tgt_shifts: np.ndarray) -> np.ndarray:
    """Compute per-sample optimal node permutation (B, G) that minimises shift MAE.

    Returns perms[b, i] = the PRED index assigned to TARGET position i, so that
    dec["shifts"][b, perms[b]] aligns pred to the target ordering.
    cost[i, j] = |tgt[i] - pred[j]|: rows = target groups, cols = pred groups.
    """
    from scipy.optimize import linear_sum_assignment
    B, G = pred_shifts.shape
    perms = np.zeros((B, G), dtype=int)
    for b in range(B):
        cost = np.abs(tgt_shifts[b, :, None] - pred_shifts[b, None, :])
        _, ci = linear_sum_assignment(cost)
        perms[b] = ci
    return perms


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def _pairs_to_matrix(jmag, G):
    """(B, n_pairs) upper-tri -> (B, G, G) symmetric, zero diagonal."""
    B = jmag.shape[0]
    iu = np.triu_indices(G, 1)
    M = np.zeros((B, G, G), float)
    M[:, iu[0], iu[1]] = jmag
    M[:, iu[1], iu[0]] = jmag
    return M


def decode(pred, standardizer, vocab, presence_thresh=0.5):
    """pred: dict of numpy arrays (shifts (B,G), j_mag (B,P), j_presence logits
    (B,P), deg_logits (B,G,C)). Returns physical dict."""
    G = pred["shifts"].shape[1]
    shifts = standardizer.inverse_shifts(pred["shifts"])
    present = _sigmoid(pred["j_presence"]) > presence_thresh
    jmag = standardizer.inverse_j(pred["j_mag"]) * present       # zero if absent
    couplings = _pairs_to_matrix(jmag, G)
    deg_idx = np.argmax(pred["deg_logits"], axis=-1)             # (B, G)
    degeneracy = np.stack([vocab.from_index(deg_idx[b]) for b in range(deg_idx.shape[0])])
    return dict(shifts=shifts, couplings=couplings, degeneracy=degeneracy,
                presence=present.astype(np.float32))


def compute_metrics(pred, target, standardizer, vocab, presence_thresh=0.5):
    """pred & target: numpy dicts in STANDARDIZED space (target as produced by
    Standardizer.transform; deg as class indices). Returns physical metrics."""
    G = pred["shifts"].shape[1]
    dec = decode(pred, standardizer, vocab, presence_thresh)

    # ground-truth physical
    tgt_shifts = standardizer.inverse_shifts(target["shifts"])
    tgt_present = target["j_presence"] > 0.5
    tgt_jmag = standardizer.inverse_j(target["j_mag"]) * tgt_present

    shift_mae = float(np.abs(dec["shifts"] - tgt_shifts).mean())

    # J MAE over TRUE-present couplings — reuse cached triu indices for this G
    if G not in _TRIU_CACHE:
        _TRIU_CACHE[G] = np.triu_indices(G, 1)
    iu = _TRIU_CACHE[G]
    pred_jmag_ut = dec["couplings"][:, iu[0], iu[1]]
    m = tgt_present
    j_mae = float(np.abs(pred_jmag_ut[m] - tgt_jmag[m]).mean()) if m.any() else 0.0

    # presence accuracy / F1
    pp = dec["presence"] > 0.5
    tp = (pp & tgt_present).sum()
    fp = (pp & ~tgt_present).sum()
    fn = (~pp & tgt_present).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    pres_acc = float((pp == tgt_present).mean())

    # degeneracy accuracy — vectorized vocab lookup over full (B, G) array at once
    tgt_deg = vocab.from_index(target["deg_class"])   # (B, G)
    pred_deg = dec["degeneracy"]
    deg_acc = float((pred_deg == tgt_deg).mean())
    recalls = []
    for v in np.unique(tgt_deg):
        m = tgt_deg == v
        recalls.append(float((pred_deg[m] == v).mean()))
    deg_acc_balanced = float(np.mean(recalls)) if recalls else 0.0

    base = dict(shift_mae_ppm=shift_mae, j_mae_hz=j_mae,
                presence_acc=pres_acc, presence_f1=float(f1),
                deg_acc=deg_acc, deg_acc_balanced=deg_acc_balanced)

    # Hungarian-matched metrics (scipy optional — silently omitted if unavailable)
    try:
        B = dec["shifts"].shape[0]
        perms = _hungarian_perm(dec["shifts"], tgt_shifts)       # (B, G)
        bi = np.arange(B)[:, None]
        h_shift_mae = float(np.abs(dec["shifts"][bi, perms] - tgt_shifts).mean())

        tgt_C = _pairs_to_matrix(tgt_jmag, G)                   # (B, G, G)
        h_j_errs: list[float] = []
        for b in range(B):
            p  = perms[b]
            pC = dec["couplings"][b][p][:, p]
            m  = tgt_present[b]
            if m.any():
                h_j_errs.extend(
                    np.abs(pC[iu[0], iu[1]][m] - tgt_C[b][iu[0], iu[1]][m]).tolist()
                )
        h_j_mae = float(np.mean(h_j_errs)) if h_j_errs else 0.0

        h_deg = dec["degeneracy"][bi, perms]
        h_deg_acc = float((h_deg == tgt_deg).mean())

        base.update(h_shift_mae_ppm=h_shift_mae, h_j_mae_hz=h_j_mae, h_deg_acc=h_deg_acc)
    except Exception:
        pass

    return base


def wasserstein1_np(spec_a, spec_b, dx=1.0, eps=1e-12):
    a = spec_a / (spec_a.sum(-1, keepdims=True) + eps)
    b = spec_b / (spec_b.sum(-1, keepdims=True) + eps)
    return np.abs(np.cumsum(a, -1) - np.cumsum(b, -1)).sum(-1) * dx
