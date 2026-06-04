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
