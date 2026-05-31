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


def augment_spectrum(spec, ppm_from=0.0, ppm_to=12.0, rng=None,
                     noise_sigma_frac=0.01, max_ref_shift_ppm=0.01,
                     baseline_amp_frac=0.02, broaden_sigma_pts=0.0):
    """Augmented copy of a normalized spectrum (unit integral).

    noise_sigma_frac   Gaussian noise std as fraction of peak height
    max_ref_shift_ppm  random global referencing shift (sub-pixel, interpolated)
    baseline_amp_frac  low-frequency baseline drift amplitude (fraction of peak)
    broaden_sigma_pts  optional Gaussian broadening (points) ~ linewidth jitter
    """
    rng = rng or np.random.default_rng()
    spec = np.asarray(spec, float).copy()
    P = len(spec)
    dx = (ppm_to - ppm_from) / P
    peak = spec.max() if spec.max() > 0 else 1.0

    if max_ref_shift_ppm > 0:
        shift_ppm = rng.uniform(-max_ref_shift_ppm, max_ref_shift_ppm)
        x = np.arange(P)
        spec = np.interp(x - shift_ppm / dx, x, spec, left=0.0, right=0.0)

    if broaden_sigma_pts > 0:
        k = int(max(3, round(6 * broaden_sigma_pts)))
        t = np.arange(-k, k + 1)
        g = np.exp(-0.5 * (t / broaden_sigma_pts) ** 2)
        g /= g.sum()
        spec = np.convolve(spec, g, mode="same")

    if baseline_amp_frac > 0:
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 2.0)
        base = baseline_amp_frac * peak * np.sin(np.linspace(0, freq * np.pi, P) + phase)
        spec = spec + (base - base.min())

    if noise_sigma_frac > 0:
        spec = spec + rng.normal(0, noise_sigma_frac * peak, P)

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
