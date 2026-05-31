"""
model.evaluation.metrics
========================
Physical-unit evaluation metrics (ported from legacy metrics.py). The numpy core
(``decode`` / ``compute_metrics``) is shared as the eval oracle; ``evaluate_output``
is the typed adapter that extracts arrays from a ``ModelOutput`` + ``SpinBatch``.

Predictions/targets live in STANDARDIZED space; metrics are reported in ppm / Hz
via the standardizer's inverse transforms.
"""
from __future__ import annotations

import numpy as np

from model.evaluation.hungarian import hungarian_perm

__all__ = ["decode", "compute_metrics", "evaluate_output"]

_TRIU_CACHE: dict[int, tuple] = {}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def _triu(G):
    if G not in _TRIU_CACHE:
        _TRIU_CACHE[G] = np.triu_indices(G, 1)
    return _TRIU_CACHE[G]


def _pairs_to_matrix(jmag, G):
    B = jmag.shape[0]
    iu = _triu(G)
    M = np.zeros((B, G, G), float)
    M[:, iu[0], iu[1]] = jmag
    M[:, iu[1], iu[0]] = jmag
    return M


def decode(pred, standardizer, vocab, presence_thresh=0.5):
    """pred: numpy dict (shifts (B,G), j_mag (B,E), j_presence logits (B,E),
    deg_logits (B,G,C)). Returns physical dict (shifts ppm, couplings Hz, deg, presence)."""
    G = pred["shifts"].shape[1]
    shifts = standardizer.inverse_shifts(pred["shifts"])
    present = _sigmoid(pred["j_presence"]) > presence_thresh
    jmag = standardizer.inverse_j(pred["j_mag"]) * present
    couplings = _pairs_to_matrix(jmag, G)
    deg_idx = np.argmax(pred["deg_logits"], axis=-1)
    degeneracy = np.stack([vocab.from_index(deg_idx[b]) for b in range(deg_idx.shape[0])])
    return dict(shifts=shifts, couplings=couplings, degeneracy=degeneracy,
                presence=present.astype(np.float32))


def compute_metrics(pred, target, standardizer, vocab, presence_thresh=0.5):
    """pred & target: numpy dicts in STANDARDIZED space. Returns physical metrics
    incl. Hungarian-matched shift/J/degeneracy (scipy optional)."""
    G = pred["shifts"].shape[1]
    dec = decode(pred, standardizer, vocab, presence_thresh)

    tgt_shifts = standardizer.inverse_shifts(target["shifts"])
    tgt_present = target["j_presence"] > 0.5
    tgt_jmag = standardizer.inverse_j(target["j_mag"]) * tgt_present

    shift_mae = float(np.abs(dec["shifts"] - tgt_shifts).mean())

    iu = _triu(G)
    pred_jmag_ut = dec["couplings"][:, iu[0], iu[1]]
    m = tgt_present
    j_mae = float(np.abs(pred_jmag_ut[m] - tgt_jmag[m]).mean()) if m.any() else 0.0

    pp = dec["presence"] > 0.5
    tp = (pp & tgt_present).sum()
    fp = (pp & ~tgt_present).sum()
    fn = (~pp & tgt_present).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    pres_acc = float((pp == tgt_present).mean())

    tgt_deg = vocab.from_index(target["deg_class"])
    pred_deg = dec["degeneracy"]
    deg_acc = float((pred_deg == tgt_deg).mean())
    recalls = []
    for v in np.unique(tgt_deg):
        mm = tgt_deg == v
        recalls.append(float((pred_deg[mm] == v).mean()))
    deg_acc_balanced = float(np.mean(recalls)) if recalls else 0.0

    base = dict(shift_mae_ppm=shift_mae, j_mae_hz=j_mae,
                presence_acc=pres_acc, presence_f1=float(f1),
                presence_precision=float(prec), presence_recall=float(rec),
                deg_acc=deg_acc, deg_acc_balanced=deg_acc_balanced)

    try:
        B = dec["shifts"].shape[0]
        perms = hungarian_perm(dec["shifts"], tgt_shifts)
        bi = np.arange(B)[:, None]
        base["h_shift_mae_ppm"] = float(np.abs(dec["shifts"][bi, perms] - tgt_shifts).mean())

        tgt_C = _pairs_to_matrix(tgt_jmag, G)
        h_j_errs: list[float] = []
        for b in range(B):
            p = perms[b]
            pC = dec["couplings"][b][p][:, p]
            mb = tgt_present[b]
            if mb.any():
                h_j_errs.extend(np.abs(pC[iu[0], iu[1]][mb] - tgt_C[b][iu[0], iu[1]][mb]).tolist())
        base["h_j_mae_hz"] = float(np.mean(h_j_errs)) if h_j_errs else 0.0
        base["h_deg_acc"] = float((pred_deg[bi, perms] == tgt_deg).mean())
    except Exception:
        pass

    return base


# ── Typed adapter ──────────────────────────────────────────────────────────────

def _np_pred(output):
    G = output.n_groups
    iu = _triu(G)
    cm = output.coupling_matrix().detach().float().cpu().numpy()
    pm = output.presence_matrix().detach().float().cpu().numpy()
    return {
        "shifts": output.shifts.detach().float().cpu().numpy(),
        "j_mag": cm[:, iu[0], iu[1]],
        "j_presence": pm[:, iu[0], iu[1]],
        "deg_logits": output.degeneracy_logits.detach().float().cpu().numpy(),
    }


def _np_target(batch):
    G = batch.n_groups
    iu = _triu(G)
    cm = batch.couplings.detach().float().cpu().numpy()
    mask = batch.coupling_mask.detach().float().cpu().numpy()
    return {
        "shifts": batch.shifts.detach().float().cpu().numpy(),
        "j_mag": cm[:, iu[0], iu[1]],
        "j_presence": mask[:, iu[0], iu[1]],
        "deg_class": batch.degeneracy_classes.detach().cpu().numpy(),
    }


def evaluate_output(output, batch, standardizer, vocab, presence_thresh=0.5) -> dict:
    """Compute metrics from a typed ModelOutput + SpinBatch."""
    return compute_metrics(_np_pred(output), _np_target(batch),
                           standardizer, vocab, presence_thresh)
