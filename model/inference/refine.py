"""
model.inference.refine
=======================
Test-time spectral refinement ("analysis-by-synthesis").

The inverse map (90 MHz spectrum -> spin system) is under-determined, but the
forward map (spin system -> spectrum) is exact and differentiable
(``model.renderers._torch_exact.simulate``). The network identifies the discrete
structure well (degeneracies, coupling topology) and gets shifts *almost* right;
we polish the predicted chemical shifts by gradient descent to maximise overlap
with the target spectrum — while staying inside a trust region of the model's
prediction (the prior that keeps us in the correct basin of the under-determined
problem). Degeneracy and coupling presence stay FIXED from the model.

This is a decode-time step: no retraining. ``refine_shifts`` is the core; it is
non-regressing on the spectral objective (reverts if it can't improve it).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from model.renderers._torch_exact import simulate, _structure
from model.evaluation.spectral_metrics import wasserstein1, cosine_similarity


def _spec_loss(sp, tgt, dx, w1_w, cos_w):
    w1 = wasserstein1(sp[None], tgt[None], dx=dx).mean()
    cos = cosine_similarity(sp[None], tgt[None]).mean()
    return w1_w * w1 + cos_w * (1.0 - cos)


def _eigh_cost(struct):
    """Cheap proxy for one simulate()'s work: sum of block eigendecomposition cost
    (~n^3) over all manifold combos. Dense/high-symmetry systems blow this up; we use
    it to skip refinement on molecules that would be too slow to polish cheaply."""
    cost = 0
    for (_, _, sb) in struct["combos"]:
        for _, blk in sb["blocks"].items():
            cost += int(blk["n"]) ** 3
    return cost


def refine_shifts(shifts0, couplings, degeneracy, target, *, field_mhz=90.0,
                  n_steps=120, lr=0.02, trust=0.3, w1_w=1.0, cos_w=0.5, reg=2.0,
                  points=16384, ppm_from=0.0, ppm_to=12.0, accept=True,
                  patience=15, plateau_tol=1e-4, max_seconds=8.0, max_cost=6e7):
    """Refine the chemical shifts (ppm) to match ``target`` spectrum; couplings and
    degeneracy held fixed.

    Args:
      shifts0:    (G,) model-predicted shifts (ppm) — the warm start / prior centre.
      couplings:  (G,G) Hz, symmetric (fixed).
      degeneracy: (G,) protons per group (fixed).
      target:     (points,) the observed/input spectrum to match (any scale; the
                  objective is scale-invariant).
      trust:      hard ppm bound on |refined - shifts0| (trust region).
      reg:        soft L2 pull toward shifts0 (extra regularisation inside the box).
      accept:     if True, revert to shifts0 when refinement fails to improve the
                  spectral loss (guarantees the step never hurts the objective).
      patience/plateau_tol: early-stop when the loss hasn't improved by plateau_tol
                  for `patience` consecutive steps (most molecules converge < 60 steps).
      max_seconds: wall-clock budget per molecule — break out with the best-so-far if
                  exceeded (bounds the cost on dense, slow-to-simulate systems).
      max_cost:   skip refinement entirely (return the prediction unchanged) when one
                  simulate() would cost more than this (see _eigh_cost) — avoids burning
                  the time budget on hopeless cases.

    Returns:
      (refined_shifts (G,) np.float, info) where info has spectral loss/cosine before
      ('*0') and after ('*1'), plus `reverted` and `skipped` flags.
    """
    dt = torch.float64
    s0 = torch.as_tensor(np.asarray(shifts0), dtype=dt)
    cpl = torch.as_tensor(np.asarray(couplings), dtype=dt)
    deg = torch.as_tensor(np.asarray(degeneracy))
    tgt = torch.as_tensor(np.asarray(target), dtype=dt)
    dx = (ppm_to - ppm_from) / points
    struct = _structure(deg, s0.device, dt)        # degeneracy fixed -> build the plan once

    def render(s):
        _, sp = simulate(s, cpl, deg, field_mhz, points, ppm_from, ppm_to, struct=struct)
        return sp

    with torch.no_grad():
        sp0 = render(s0)
        loss0 = _spec_loss(sp0, tgt, dx, w1_w, cos_w).item()
        cos0 = cosine_similarity(sp0[None], tgt[None]).item()

    # cost guard: too expensive to polish cheaply -> leave the prediction untouched.
    if max_cost is not None and _eigh_cost(struct) > max_cost:
        return np.asarray(shifts0, float), {
            "loss0": loss0, "loss1": loss0, "cos0": cos0, "cos1": cos0,
            "reverted": False, "skipped": True, "steps": 0}

    s = s0.clone().requires_grad_(True)
    opt = torch.optim.Adam([s], lr=lr)
    lo, hi = s0 - trust, s0 + trust
    best, stale, t0, step = loss0, 0, time.time(), 0
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        loss = _spec_loss(render(s), tgt, dx, w1_w, cos_w) + reg * ((s - s0) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            s.copy_(torch.min(torch.max(s, lo), hi))   # project back into the trust box
        lv = float(loss.detach())
        if lv < best - plateau_tol:
            best, stale = lv, 0
        else:
            stale += 1
        if stale >= patience or (time.time() - t0) > max_seconds:   # converged or out of time
            break

    with torch.no_grad():
        sp1 = render(s)
        loss1 = _spec_loss(sp1, tgt, dx, w1_w, cos_w).item()
        cos1 = cosine_similarity(sp1[None], tgt[None]).item()

    refined = s.detach()
    reverted = False
    if accept and loss1 > loss0:                       # non-regressing guard
        refined, loss1, cos1, reverted = s0, loss0, cos0, True
    return refined.cpu().numpy(), {
        "loss0": loss0, "loss1": loss1, "cos0": cos0, "cos1": cos1,
        "reverted": reverted, "skipped": False, "steps": step,
    }


# ──────────────────────────────────────────────────────────────────────────────
# refine_system — graduated-non-convexity joint shift + J refinement
# ──────────────────────────────────────────────────────────────────────────────
# refine_shifts() only moves shifts and gets trapped on multiplets: when a rendered
# multiplet is offset by ~J, W1/cosine settle on a *partial* sub-peak overlap (a local
# min "off by a J"). Two changes fix that and add J:
#   1. Graduated non-convexity — score the fit at an artificially broadened linewidth,
#      annealed broad->sharp. At the coarse scale a whole multiplet is one blob, so the
#      objective is unimodal and aligning the blobs aligns the multiplet CENTROID (=the
#      chemical shift), with no off-by-J trap; sharpening then resolves the J splitting.
#      This also gives "shifts first, J second" for free (coarse=shift, fine=J).
#   2. Joint shift + J — optimize the chemical shifts AND only the couplings the model
#      predicted NONZERO (never turn a coupling on); degeneracy + topology stay fixed.
# Both stay inside a trust region + soft prior centred on the model's prediction (MAP).


def _gaussian_blur(y, sigma_pts):
    """1-D Gaussian convolution along the spectrum (differentiable). sigma in points."""
    if sigma_pts <= 0.5:
        return y
    r = max(1, int(round(3 * sigma_pts)))
    x = torch.arange(-r, r + 1, dtype=y.dtype, device=y.device)
    k = torch.exp(-0.5 * (x / sigma_pts) ** 2)
    k = k / k.sum()
    import torch.nn.functional as F
    return F.conv1d(y.view(1, 1, -1), k.view(1, 1, -1), padding=r).view(-1)


def _coarse_centroid_fix(shifts0, sp_pred, tgt, ppm_from, ppm_to, gap_ppm=0.15, thresh_frac=0.05):
    """Gradient-free coarse shift fix via multiplet centroids. A multiplet is symmetric
    about its chemical shift, so the intensity-weighted centroid of the multiplet IS the
    shift — robust to the off-by-J sub-peak cross-alignment that traps gradient descent on
    blurred spectra. We cluster the predicted groups by shift, split the ppm axis at the
    midpoints between cluster centres (so each cluster owns a window), and shift each
    cluster by (target_centroid − predicted_centroid) in its window. Returns new shifts.
    No simulation: uses the already-rendered predicted spectrum + the target."""
    shifts0 = np.asarray(shifts0, float)
    sp_pred = np.asarray(sp_pred, float); tgt = np.asarray(tgt, float)
    G, P = len(shifts0), len(sp_pred)
    ppm = np.linspace(ppm_from, ppm_to, P)
    order = np.argsort(shifts0); ss = shifts0[order]
    clusters = [[order[0]]]
    for k in range(1, G):
        if ss[k] - ss[k - 1] <= gap_ppm:
            clusters[-1].append(order[k])
        else:
            clusters.append([order[k]])
    centers = np.array([shifts0[cl].mean() for cl in clusters])
    bnd = np.concatenate([[ppm_from], (centers[:-1] + centers[1:]) / 2.0, [ppm_to]])
    tp, tt = thresh_frac * sp_pred.max(), thresh_frac * tgt.max()
    out = shifts0.copy()
    for ci, cl in enumerate(clusters):
        m = (ppm >= bnd[ci]) & (ppm < bnd[ci + 1])
        wp = np.clip(sp_pred[m] - tp, 0, None); wt = np.clip(tgt[m] - tt, 0, None)
        if wp.sum() <= 0 or wt.sum() <= 0:
            continue                                    # empty window -> leave this cluster as predicted
        c_pred = float((ppm[m] * wp).sum() / wp.sum())
        c_tgt = float((ppm[m] * wt).sum() / wt.sum())
        for g in cl:
            out[g] += (c_tgt - c_pred)
    return out


def equiv_classes_from_softequiv(se_prob, G, thresh=0.5):
    """Union-find on the soft-equivalence edge probabilities (``se_prob``: (E,) upper-tri
    order) -> list of group-index lists. Groups the model flags chemically equivalent end
    up in one class (singletons for the rest). These classes share one shift in refinement."""
    iu = np.triu_indices(G, 1)
    parent = list(range(G))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for e, (i, j) in enumerate(zip(iu[0], iu[1])):
        if se_prob[e] > thresh:
            parent[find(int(i))] = find(int(j))
    cl = {}
    for n in range(G):
        cl.setdefault(find(n), []).append(n)
    return list(cl.values())


def refine_system(shifts0, couplings, degeneracy, target, *, equiv=None, field_mhz=90.0,
                  points=16384, ppm_from=0.0, ppm_to=12.0,
                  scales_ppm=(0.05, 0.0), steps_per_scale=45,
                  lr_shift=0.02, lr_j=0.3, trust_shift=0.3, trust_j=4.0,
                  reg_shift=2.0, reg_j=0.02, w1_w=1.0, cos_w=0.5, base_lw_hz=1.0,
                  presence_tol=0.5, patience=8, plateau_tol=1e-4,
                  coarse_centroid=True, coarse_gap_ppm=0.15,
                  accept=True, max_seconds=15.0, max_cost=6e7):
    """Jointly refine shifts (ppm) + the predicted-nonzero couplings (Hz) to match
    ``target``, via graduated linewidth annealing. ``couplings`` (G,G) symmetric: only
    entries with |J|>presence_tol are optimized (the model's predicted topology); the
    rest stay 0. Returns (shifts (G,), couplings (G,G), info).

    Speed: the broad scales are produced by rendering the prediction at a wide
    ``linewidth_hz`` (free in the lineshape build) instead of convolving it with a wide
    Gaussian every step; each scale early-stops on a patience plateau. Both cut work
    without changing the outcome — the target is still Gaussian-blurred to match scale,
    and the final/objective scale is the native linewidth."""
    dt = torch.float64
    s0 = torch.as_tensor(np.asarray(shifts0), dtype=dt)
    C0 = torch.as_tensor(np.asarray(couplings), dtype=dt)
    deg = torch.as_tensor(np.asarray(degeneracy))
    tgt = torch.as_tensor(np.asarray(target), dtype=dt)
    G = s0.numel()
    dx = (ppm_to - ppm_from) / points
    ppm_per_pt = (ppm_to - ppm_from) / points
    struct = _structure(deg, s0.device, dt)

    # equivalence classes: groups the model flags chemically-equivalent share ONE shift, so
    # they move together and never split into the off-by-J artifact. equiv = list of group-
    # index lists; default = each group its own class (no tying). Optimize one shift per class.
    if equiv is None:
        equiv = [[g] for g in range(G)]
    cls_of = torch.zeros(G, dtype=torch.long)
    for ci, members in enumerate(equiv):
        for g in members:
            cls_of[g] = ci
    def expand(sf):                                     # (n_cls,) class shifts -> (G,) per-group
        return sf[cls_of]
    s0_free = torch.stack([s0[torch.tensor(m)].mean() for m in equiv])   # (n_cls,) class-mean of prediction

    iu = torch.triu_indices(G, G, 1, device=s0.device)
    j0 = C0[iu[0], iu[1]].clone()                       # (E,) upper-tri couplings
    pres = j0.abs() > presence_tol                      # predicted-nonzero edges (the only ones we move)

    def build_C(jv):
        C = torch.zeros(G, G, dtype=dt)
        C[iu[0], iu[1]] = jv; C[iu[1], iu[0]] = jv
        return C

    def render(s, jv, lw):
        _, sp = simulate(s, build_C(jv), deg, field_mhz, points, ppm_from, ppm_to,
                         linewidth_hz=lw, struct=struct)
        return sp

    # per scale: render the prediction at a broad linewidth (Hz) and Gaussian-blur the
    # target to match. sc=0 -> native linewidth, unblurred target (== the real objective).
    sigmas = [(sc / ppm_per_pt if sc > 0 else 0.0) for sc in scales_ppm]
    lws = [base_lw_hz + sc * field_mhz for sc in scales_ppm]
    tgt_blur = {sc: (_gaussian_blur(tgt, sg) if sg > 0 else tgt) for sc, sg in zip(scales_ppm, sigmas)}

    def loss_at(sp, sc):
        b = tgt_blur[sc]
        w1 = wasserstein1(sp[None], b[None], dx=dx).mean()
        cos = cosine_similarity(sp[None], b[None]).mean()
        return w1_w * w1 + cos_w * (1.0 - cos)

    with torch.no_grad():
        sp0 = render(s0, j0, base_lw_hz)
        loss0 = _spec_loss(sp0, tgt, dx, w1_w, cos_w).item()   # sharp-scale loss (the real objective)
        cos0 = cosine_similarity(sp0[None], tgt[None]).item()

    if max_cost is not None and _eigh_cost(struct) > max_cost:
        return (np.asarray(shifts0, float), np.asarray(couplings, float),
                {"loss0": loss0, "loss1": loss0, "cos0": cos0, "cos1": cos0,
                 "reverted": False, "skipped": True, "steps": 0})

    # coarse basin-fix: jump each multiplet to its target centroid (gradient-free, no eigh),
    # then collapse to ONE shift per equivalence class (so equivalent groups stay locked together).
    if coarse_centroid:
        s_init = _coarse_centroid_fix(np.asarray(shifts0, float), sp0.cpu().numpy(),
                                      np.asarray(target, float), ppm_from, ppm_to, gap_ppm=coarse_gap_ppm)
    else:
        s_init = np.asarray(shifts0, float)
    s_init_t = torch.as_tensor(s_init, dtype=dt)
    sfree_init = torch.stack([s_init_t[torch.tensor(m)].mean() for m in equiv])     # per-class init
    sfree_init = torch.min(torch.max(sfree_init, s0_free - trust_shift), s0_free + trust_shift)
    sfree = sfree_init.clone().requires_grad_(True)    # ONE shift per equivalence class
    j = j0.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [sfree], "lr": lr_shift}, {"params": [j], "lr": lr_j}])
    sflo, sfhi = s0_free - trust_shift, s0_free + trust_shift
    jlo, jhi = j0 - trust_j, j0 + trust_j
    t0 = time.time(); steps = 0
    for sc, lw in zip(scales_ppm, lws):
        best_sc, stale = float("inf"), 0
        for _ in range(steps_per_scale):
            opt.zero_grad()
            loss = (loss_at(render(expand(sfree), j, lw), sc)
                    + reg_shift * ((sfree - s0_free) ** 2).mean()
                    + reg_j * (((j - j0) * pres) ** 2).sum() / pres.sum().clamp_min(1))
            loss.backward()
            with torch.no_grad():
                if j.grad is not None:
                    j.grad[~pres] = 0.0                 # never move an absent coupling
            opt.step()
            with torch.no_grad():
                sfree.copy_(torch.min(torch.max(sfree, sflo), sfhi))   # trust boxes
                j.copy_(torch.min(torch.max(j, jlo), jhi))
                j[~pres] = j0[~pres]                    # keep absent couplings at 0
            steps += 1
            lv = float(loss.detach())
            if lv < best_sc - plateau_tol:
                best_sc, stale = lv, 0
            else:
                stale += 1
            if stale >= patience or (time.time() - t0) > max_seconds:   # converged / out of time
                break
        if (time.time() - t0) > max_seconds:
            break

    with torch.no_grad():
        sp1 = render(expand(sfree), j, base_lw_hz)
        loss1 = _spec_loss(sp1, tgt, dx, w1_w, cos_w).item()
        cos1 = cosine_similarity(sp1[None], tgt[None]).item()
    s_out, j_out, reverted = expand(sfree).detach(), j.detach(), False
    if accept and loss1 > loss0:                        # non-regressing guard (sharp scale)
        s_out, j_out, loss1, cos1, reverted = s0, j0, loss0, cos0, True
    return (s_out.cpu().numpy(), build_C(j_out).cpu().numpy(),
            {"loss0": loss0, "loss1": loss1, "cos0": cos0, "cos1": cos1,
             "reverted": reverted, "skipped": False, "steps": steps})
