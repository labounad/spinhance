"""
model.data.transforms
=====================
Torch-free target encoding and spectrum augmentation (ported from legacy
targets.py). Kept torch-free so it is unit-testable without torch and shared by
the dataset.

Target layout (per molecule, G groups, canonical-ordered):
  shifts      (G,)              ppm                 -> regression (standardized)
  j_mag       (G*(G-1)/2,)      Hz, upper triangle  -> regression (standardized, masked)
  j_presence  (G*(G-1)/2,)      {0,1}               -> binary classification
  deg_class   (G,)              vocab index         -> classification
"""
from __future__ import annotations

import numpy as np

from model.data.splits import canonical_order, reorder

__all__ = ["encode_target", "augment_spectrum", "bucket_key"]


# ── Encode one molecule's matrix into target components (canonical-ordered) ─────

def encode_target(shifts, couplings, degeneracy, vocab, j_zero_tol=1e-6, order=None):
    if order is None:
        order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    G = len(s)
    iu = np.triu_indices(G, 1)
    j_mag = c[iu].astype(float)
    j_presence = (np.abs(j_mag) > j_zero_tol).astype(np.float32)
    deg_class = vocab.to_index(d)
    return dict(shifts=s.astype(np.float32), j_mag=j_mag.astype(np.float32),
                j_presence=j_presence, deg_class=deg_class, order=order)


# ── On-the-fly spectrum augmentation (train only); preserves length + unit ∫ ────

def _renorm(spec, dx):
    area = spec.sum() * dx
    return spec / area if area > 0 else spec


# Peak/sum ratio of ONE simulated line (pseudo-Voigt, η=0.8, ~1 Hz @ 90 MHz on
# the 12 ppm / 16384-pt grid → ~7.6-pt HWHM). Converts a group's per-proton
# integral into its per-proton PEAK height.  It only calibrates the absolute
# noise scale — i.e. what `frac` means as 1/SNR of a single proton; the
# per-molecule 1/N scaling (the actual fix vs the old spec.max() reference) is
# independent of this constant.  Recompute if the simulation linewidth changes.
_LINE_PEAK_TO_SUM = 0.047


def _one_h_height(spec, n_protons):
    """Peak height a single-proton singlet would have in this unit-∫ spectrum.

    The spectrum integrates to 1 over its ``n_protons`` protons, so one proton's
    integral is ``Σspec/n_protons`` and its peak height is that times the line's
    peak/sum ratio.  Referencing noise to THIS (not ``spec.max()``) means a
    molecule with a tall 9H tert-butyl singlet no longer gets 9× the noise on
    its minor peaks.
    """
    return _LINE_PEAK_TO_SUM * float(np.sum(spec)) / max(int(n_protons), 1)


def augment_spectrum(spec, ppm_from=0.0, ppm_to=12.0, rng=None, *,
                     n_protons=None, noise_frac_range=(0.003, 0.015),
                     broaden_sigma_pts=0.0):
    """Augmented copy of a normalized spectrum (unit integral).

    The model operates on **processed, referenced** spectra, so we only model
    distortions that survive good processing:

    noise_frac_range   (lo, hi) for the intensity-noise level, sampled
                       LOG-UNIFORMLY per spectrum.  The level is the noise RMS
                       as a fraction of a **1H-singlet** peak height — i.e.
                       1/SNR of a single proton — so it is independent of how
                       the molecule's protons are distributed among peaks
                       (a 9H tert-butyl singlet no longer inflates the noise).
                       Pass equal endpoints for a fixed level; ``None`` to skip.
    n_protons          total protons (Σ degeneracy); sets the 1H reference
                       height.  If ``None``, falls back to ``spec.max()``
                       (standalone use only — the dataset always passes it).
    broaden_sigma_pts  optional Gaussian broadening (points) ~ linewidth jitter

    Deliberately NOT modeled:
      * baseline drift — modern processing baseline-corrects accurately; the
        input is already a clean baseline.
      * global referencing shift — the spectrum is already referenced, so its
        peak positions ARE the true shifts.  Sliding the spectrum without
        moving the labels would inject pure label noise (~the magnitude of the
        target shift error), teaching the model to predict a position other
        than where the peak actually is.
    """
    rng = rng or np.random.default_rng()
    spec = np.asarray(spec, float).copy()
    P = len(spec)
    dx = (ppm_to - ppm_from) / P

    if broaden_sigma_pts > 0:
        k = int(max(3, round(6 * broaden_sigma_pts)))
        t = np.arange(-k, k + 1)
        g = np.exp(-0.5 * (t / broaden_sigma_pts) ** 2)
        g /= g.sum()
        spec = np.convolve(spec, g, mode="same")

    if noise_frac_range is not None:
        lo, hi = noise_frac_range
        frac = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if hi > lo else float(lo)
        ref = (_one_h_height(spec, n_protons) if n_protons
               else (spec.max() if spec.max() > 0 else 1.0))
        if ref > 0:
            spec = spec + rng.normal(0, frac * ref, P)

    spec = np.clip(spec, 0.0, None)
    return _renorm(spec, dx).astype(np.float32)


# ── Bucket key for renderer struct-sharing (Stage-2 surrogate) ─────────────────

def bucket_key(shifts, couplings, degeneracy, order=None):
    """Canonical-ordered degeneracy vector; samples with the same key share a
    renderer ``struct`` (same Hilbert space)."""
    if order is None:
        order = canonical_order(shifts, couplings, degeneracy)
    _, _, d = reorder(shifts, couplings, degeneracy, order)
    return tuple(int(x) for x in d)
