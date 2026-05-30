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

    # J MAE over TRUE-present couplings (upper triangle)
    iu = np.triu_indices(G, 1)
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

    # degeneracy accuracy (raw + BALANCED = mean per-class recall).
    # Raw acc is misleading under heavy imbalance (~89% d=1): a model that always
    # predicts 1 scores ~0.89. Balanced acc exposes that collapse (would be ~1/#classes).
    tgt_deg = np.stack([vocab.from_index(target["deg_class"][b])
                        for b in range(target["deg_class"].shape[0])])
    pred_deg = dec["degeneracy"]
    deg_acc = float((pred_deg == tgt_deg).mean())
    recalls = []
    for v in np.unique(tgt_deg):
        m = tgt_deg == v
        recalls.append(float((pred_deg[m] == v).mean()))
    deg_acc_balanced = float(np.mean(recalls)) if recalls else 0.0

    return dict(shift_mae_ppm=shift_mae, j_mae_hz=j_mae,
                presence_acc=pres_acc, presence_f1=float(f1),
                deg_acc=deg_acc, deg_acc_balanced=deg_acc_balanced)


def wasserstein1_np(spec_a, spec_b, dx=1.0, eps=1e-12):
    a = spec_a / (spec_a.sum(-1, keepdims=True) + eps)
    b = spec_b / (spec_b.sum(-1, keepdims=True) + eps)
    return np.abs(np.cumsum(a, -1) - np.cumsum(b, -1)).sum(-1) * dx
