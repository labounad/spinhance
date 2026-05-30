"""
model.losses
===============
Stage-1 matrix loss and Stage-2 spectral-consistency loss (Decisions 4, 6, 7).

Matrix loss (per element, canonical-ordered, standardized space):
  shifts      -> smooth-L1 (Huber)
  j_mag       -> smooth-L1, MASKED by ground-truth presence (only real couplings)
  j_presence  -> BCE-with-logits
  deg         -> cross-entropy over the degeneracy vocab
weighted sum; component weights are explicit (Decision 4).

Spectral loss (Stage 2): decode the prediction to physical units, render it with
the differentiable renderer (diff_renderer_torch) at 90 (+optional 600) MHz, and
compare to the reference spectrum with a 1-D Wasserstein distance (+ optional MSE
lineshape term). Permutation-invariant by construction.

The torch-free schedules live in model.schedules.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from model.schedules import curriculum_weights  # re-export convenience

__all__ = ["matrix_loss", "wasserstein1", "spectral_loss", "curriculum_weights"]


# -----------------------------------------------------------------------------
# Stage 1 — matrix loss
# -----------------------------------------------------------------------------

def matrix_loss(pred, target, weights=None, huber_beta=1.0):
    """pred: model output dict; target: batch dict (standardized).
    Returns (total, components)."""
    w = {"shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5}
    if weights:
        w.update(weights)

    shift = F.smooth_l1_loss(pred["shifts"], target["shifts"], beta=huber_beta)

    mask = target["j_presence"]                      # (B, n_pairs) in {0,1}
    jmag_el = F.smooth_l1_loss(pred["j_mag"], target["j_mag"],
                               beta=huber_beta, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    jmag = (jmag_el * mask).sum() / denom            # masked mean over present

    presence = F.binary_cross_entropy_with_logits(pred["j_presence"], mask)

    B, G, C = pred["deg_logits"].shape
    deg = F.cross_entropy(pred["deg_logits"].reshape(B * G, C),
                          target["deg_class"].reshape(B * G))

    total = (w["shift"] * shift + w["jmag"] * jmag
             + w["presence"] * presence + w["deg"] * deg)
    comps = {"shift": shift.detach(), "jmag": jmag.detach(),
             "presence": presence.detach(), "deg": deg.detach()}
    return total, comps


# -----------------------------------------------------------------------------
# Stage 2 — spectral consistency
# -----------------------------------------------------------------------------

def wasserstein1(spec_a, spec_b, dx=1.0, eps=1e-12):
    """1-D Wasserstein-1 between two non-negative spectra over a common grid.

    Normalizes each to a probability distribution, then integrates |CDF_a-CDF_b|.
    Batched over the leading dim. Differentiable in both inputs.
    """
    a = spec_a / (spec_a.sum(dim=-1, keepdim=True) + eps)
    b = spec_b / (spec_b.sum(dim=-1, keepdim=True) + eps)
    ca = torch.cumsum(a, dim=-1)
    cb = torch.cumsum(b, dim=-1)
    return (ca - cb).abs().sum(dim=-1) * dx


def spectral_loss(pred_phys, ref_spectra, degeneracy, field_mhz, renderer,
                  struct=None, points=16384, ppm_from=0.0, ppm_to=12.0,
                  linewidth_hz=1.0, eigh_eps=1.0, lineshape_weight=0.0):
    """Render predicted matrices and compare to reference spectra.

    pred_phys   : dict with physical 'shifts' (B,G) ppm and 'couplings' (B,G,G) Hz
    ref_spectra : (B, points) reference spectra at field_mhz (unit integral)
    degeneracy  : (B, G) int (shared within a bucket -> pass struct once)
    renderer    : model.diff_renderer_torch module (or .simulate callable)
    Returns (loss, per_sample_w1).
    """
    B = ref_spectra.shape[0]
    dx = (ppm_to - ppm_from) / points
    sims = []
    for i in range(B):
        deg_i = [int(x) for x in degeneracy[i].tolist()]
        _, spec = renderer.simulate(
            pred_phys["shifts"][i], pred_phys["couplings"][i], deg_i, field_mhz,
            points=points, ppm_from=ppm_from, ppm_to=ppm_to,
            linewidth_hz=linewidth_hz, eigh_eps=eigh_eps,
            struct=struct if struct is not None else None)
        sims.append(spec)
    sim = torch.stack(sims)                          # (B, points)
    w1 = wasserstein1(sim, ref_spectra, dx=dx)
    loss = w1.mean()
    if lineshape_weight > 0:
        loss = loss + lineshape_weight * F.mse_loss(sim, ref_spectra)
    return loss, w1.detach()
