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

import numpy as np
import torch

from model.renderers._torch_exact import simulate, _structure
from model.evaluation.spectral_metrics import wasserstein1, cosine_similarity


def _spec_loss(sp, tgt, dx, w1_w, cos_w):
    w1 = wasserstein1(sp[None], tgt[None], dx=dx).mean()
    cos = cosine_similarity(sp[None], tgt[None]).mean()
    return w1_w * w1 + cos_w * (1.0 - cos)


def refine_shifts(shifts0, couplings, degeneracy, target, *, field_mhz=90.0,
                  n_steps=150, lr=0.02, trust=0.3, w1_w=1.0, cos_w=0.5, reg=2.0,
                  points=16384, ppm_from=0.0, ppm_to=12.0, accept=True):
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

    Returns:
      (refined_shifts (G,) np.float, info) where info has spectral loss/cosine
      before ('*0') and after ('*1') refinement.
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

    s = s0.clone().requires_grad_(True)
    opt = torch.optim.Adam([s], lr=lr)
    lo, hi = s0 - trust, s0 + trust
    for _ in range(n_steps):
        opt.zero_grad()
        loss = _spec_loss(render(s), tgt, dx, w1_w, cos_w) + reg * ((s - s0) ** 2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            s.copy_(torch.min(torch.max(s, lo), hi))   # project back into the trust box

    with torch.no_grad():
        sp1 = render(s)
        loss1 = _spec_loss(sp1, tgt, dx, w1_w, cos_w).item()
        cos1 = cosine_similarity(sp1[None], tgt[None]).item()

    refined = s.detach()
    reverted = False
    if accept and loss1 > loss0:                       # non-regressing guard
        refined, loss1, cos1, reverted = s0, loss0, cos0, True
    return refined.cpu().numpy(), {
        "loss0": loss0, "loss1": loss1, "cos0": cos0, "cos1": cos1, "reverted": reverted,
    }
