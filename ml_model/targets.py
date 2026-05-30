"""
ml_model.targets
=================
Torch-free target encoding/decoding, standardization, and spectrum augmentation
for Task 4. Kept torch-free so it can be unit-tested without torch and shared by
the torch Dataset in ``ml_model.dataset``.

Target layout (per molecule, G groups, canonical-ordered — Decision 3):
  shifts      (G,)              ppm                 -> regression (standardized)
  j_mag       (G*(G-1)/2,)      Hz, upper triangle  -> regression (standardized, masked)
  j_presence  (G*(G-1)/2,)      {0,1}               -> binary classification
  deg_class   (G,)              vocab index         -> classification

Standardization (Decision 4): z-score shifts over all groups and couplings over
PRESENT entries only, using TRAIN-set statistics. Absent couplings are masked
out of the loss, so their standardized value is irrelevant (set to 0).
"""

from __future__ import annotations

import numpy as np

from ml_model.splits import canonical_order, reorder

__all__ = ["DegeneracyVocab", "encode_target", "Standardizer",
           "augment_spectrum", "bucket_key", "DEFAULT_DEG_VOCAB"]

DEFAULT_DEG_VOCAB = (1, 2, 3, 4, 6, 9, 12, 18)


# -----------------------------------------------------------------------------
# Degeneracy vocabulary (classification target)
# -----------------------------------------------------------------------------

class DegeneracyVocab:
    def __init__(self, vocab=DEFAULT_DEG_VOCAB):
        self.vocab = tuple(int(v) for v in vocab)
        self._to = {v: i for i, v in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def to_index(self, degeneracy):
        out = []
        for d in np.asarray(degeneracy).ravel():
            d = int(round(float(d)))
            if d not in self._to:
                raise KeyError(f"degeneracy {d} not in vocab {self.vocab}")
            out.append(self._to[d])
        return np.array(out, dtype=np.int64)

    def from_index(self, idx):
        return np.array([self.vocab[int(i)] for i in np.asarray(idx).ravel()],
                        dtype=np.int64)


# -----------------------------------------------------------------------------
# Encode one molecule's matrix into target components (canonical-ordered)
# -----------------------------------------------------------------------------

def encode_target(shifts, couplings, degeneracy, vocab: DegeneracyVocab,
                  j_zero_tol=1e-6):
    order = canonical_order(shifts, couplings, degeneracy)
    s, c, d = reorder(shifts, couplings, degeneracy, order)
    G = len(s)
    iu = np.triu_indices(G, 1)
    j_mag = c[iu].astype(float)
    j_presence = (np.abs(j_mag) > j_zero_tol).astype(np.float32)
    deg_class = vocab.to_index(d)
    return dict(shifts=s.astype(np.float32), j_mag=j_mag.astype(np.float32),
                j_presence=j_presence, deg_class=deg_class, order=order)


# -----------------------------------------------------------------------------
# Standardizer (fit on TRAIN only)
# -----------------------------------------------------------------------------

class Standardizer:
    """Z-score shifts (all) and coupling magnitudes (present entries only)."""

    def __init__(self):
        self.shift_mean = self.shift_std = None
        self.j_mean = self.j_std = None

    def fit(self, train_records, vocab: DegeneracyVocab):
        shifts_all, j_present = [], []
        for r in train_records:
            t = encode_target(r["shifts"], r["couplings"], r["degeneracy"], vocab)
            shifts_all.append(t["shifts"])
            mask = t["j_presence"] > 0
            j_present.append(t["j_mag"][mask])
        shifts_all = np.concatenate(shifts_all)
        j_present = np.concatenate(j_present) if any(len(x) for x in j_present) \
            else np.array([0.0])
        self.shift_mean, self.shift_std = float(shifts_all.mean()), float(shifts_all.std() + 1e-8)
        self.j_mean, self.j_std = float(j_present.mean()), float(j_present.std() + 1e-8)
        return self

    def transform(self, t):
        out = dict(t)
        out["shifts"] = (t["shifts"] - self.shift_mean) / self.shift_std
        jm = (t["j_mag"] - self.j_mean) / self.j_std
        out["j_mag"] = (jm * t["j_presence"]).astype(np.float32)   # zero where absent
        return out

    def inverse_shifts(self, x):
        return np.asarray(x) * self.shift_std + self.shift_mean

    def inverse_j(self, x):
        return np.asarray(x) * self.j_std + self.j_mean


# -----------------------------------------------------------------------------
# On-the-fly spectrum augmentation (train only); preserves length & unit integral
# -----------------------------------------------------------------------------

def _renorm(spec, dx):
    area = spec.sum() * dx
    return spec / area if area > 0 else spec


def augment_spectrum(spec, ppm_from=0.0, ppm_to=12.0, rng=None,
                     noise_sigma_frac=0.01, max_ref_shift_ppm=0.01,
                     baseline_amp_frac=0.02, broaden_sigma_pts=0.0):
    """Return an augmented copy of a normalized spectrum (unit integral).

    noise_sigma_frac     : Gaussian noise std as fraction of peak height
    max_ref_shift_ppm    : random global referencing shift (sub-pixel, interpolated)
    baseline_amp_frac    : low-frequency baseline drift amplitude (fraction of peak)
    broaden_sigma_pts    : optional Gaussian broadening (points) ~ linewidth jitter
    """
    rng = rng or np.random.default_rng()
    spec = np.asarray(spec, float).copy()
    P = len(spec)
    dx = (ppm_to - ppm_from) / P
    peak = spec.max() if spec.max() > 0 else 1.0

    # referencing shift: sub-pixel via linear interpolation of the grid
    if max_ref_shift_ppm > 0:
        shift_ppm = rng.uniform(-max_ref_shift_ppm, max_ref_shift_ppm)
        x = np.arange(P)
        spec = np.interp(x - shift_ppm / dx, x, spec, left=0.0, right=0.0)

    # optional broadening (linewidth jitter proxy)
    if broaden_sigma_pts > 0:
        k = int(max(3, round(6 * broaden_sigma_pts)))
        t = np.arange(-k, k + 1)
        g = np.exp(-0.5 * (t / broaden_sigma_pts) ** 2)
        g /= g.sum()
        spec = np.convolve(spec, g, mode="same")

    # smooth baseline drift (low-order sinusoid)
    if baseline_amp_frac > 0:
        phase = rng.uniform(0, 2 * np.pi)
        freq = rng.uniform(0.5, 2.0)
        base = baseline_amp_frac * peak * np.sin(
            np.linspace(0, freq * np.pi, P) + phase)
        spec = spec + (base - base.min())   # keep non-negative

    # additive noise
    if noise_sigma_frac > 0:
        spec = spec + rng.normal(0, noise_sigma_frac * peak, P)

    spec = np.clip(spec, 0.0, None)
    return _renorm(spec, dx).astype(np.float32)


# -----------------------------------------------------------------------------
# Bucket key for structure-sharing in the Stage-2 renderer
# -----------------------------------------------------------------------------

def bucket_key(shifts, couplings, degeneracy):
    """Canonical-ordered degeneracy vector -> the renderer ``struct`` is shared
    across all samples with the same key (same Hilbert space + operators)."""
    order = canonical_order(shifts, couplings, degeneracy)
    _, _, d = reorder(shifts, couplings, degeneracy, order)
    return tuple(int(x) for x in d)
