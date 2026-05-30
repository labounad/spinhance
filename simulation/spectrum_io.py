"""
spectrum_io.py
==============
Canonical save/load for the three SpinHance spectrum representations, in order
of increasing sparsity:

1. **dense**  — full intensity array on the grid (``.npy``). What the engines
   return; what Task 4 ultimately consumes.
2. **sparse** — the dense spectrum with points ≤ ``cutoff·max`` dropped, stored
   as ``(idx, val)`` and renormalised to ∫=1 (``.npz``). ~7× fewer points.
3. **peaks**  — the line list itself: ``(centers_ppm, amps)`` plus the lineshape
   parameters (``linewidth_hz``, ``field_mhz``, grid). The lineshape is applied
   **on the fly** at load time (convolve), so storage is just the transitions —
   the most compact form. ``.npz``.

``load_spectrum`` reads any of the three into a dense array, so downstream code
(plotting, Task 4) is representation-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simulation.pyspin.simulator import peaks_to_spectrum

__all__ = [
    "sparsify", "save_dense", "save_sparse", "save_peaks",
    "load_spectrum", "DEFAULT_CUTOFF",
]

DEFAULT_CUTOFF = 0.001  # drop points ≤ 0.1% of the per-spectrum max


# ── dense ─────────────────────────────────────────────────────────────────────

def save_dense(path, y) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(y, dtype=np.float32))
    return path


# ── sparse (thresholded broadened spectrum) ──────────────────────────────────

def sparsify(y, cutoff: float = DEFAULT_CUTOFF, renormalize: bool = True):
    """Drop points ≤ ``cutoff`` × max; return ``(idx int32, val float32)``.

    With ``renormalize``, kept values are rescaled so ∫ over the (unchanged) ppm
    grid stays 1.
    """
    y = np.asarray(y, dtype=np.float64)
    thr = cutoff * y.max() if y.size else 0.0
    idx = np.nonzero(y > thr)[0]
    val = y[idx]
    if renormalize and val.sum() > 0:
        val = val * (y.sum() / val.sum())
    return idx.astype(np.int32), val.astype(np.float32)


def save_sparse(path, y, cutoff: float = DEFAULT_CUTOFF, renormalize: bool = True,
                compressed: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx, val = sparsify(y, cutoff, renormalize)
    saver = np.savez_compressed if compressed else np.savez
    saver(path, idx=idx, val=val, n=np.int32(len(y)), cutoff=np.float32(cutoff))
    return path


# ── peaks (line list; lineshape applied at load) ─────────────────────────────

def save_peaks(path, centers_ppm, amps, *, linewidth_hz: float, field_mhz: float,
               points: int = 16384, ppm_from: float = 0.0, ppm_to: float = 12.0,
               rel_threshold: float = 1e-4, compressed: bool = True) -> Path:
    """Store a peak list + lineshape parameters (the most compact form).

    Lines with ``amp ≤ rel_threshold × max`` are dropped. Reconstruction
    (``load_spectrum``) bins to the grid and Lorentzian-broadens on the fly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = np.asarray(centers_ppm, dtype=np.float64)
    a = np.asarray(amps, dtype=np.float64)
    if a.size:
        keep = a > rel_threshold * a.max()
        c, a = c[keep], a[keep]
    saver = np.savez_compressed if compressed else np.savez
    saver(path, centers=c.astype(np.float32), amps=a.astype(np.float32),
          linewidth_hz=np.float32(linewidth_hz), field_mhz=np.float32(field_mhz),
          points=np.int32(points), ppm_from=np.float32(ppm_from),
          ppm_to=np.float32(ppm_to))
    return path


# ── unified loader ────────────────────────────────────────────────────────────

def load_spectrum(path) -> np.ndarray:
    """Load any representation (.npy dense, .npz sparse, .npz peaks) → dense array."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    d = np.load(path)
    keys = set(d.files)
    if "centers" in keys:                       # peaks → convolve on the fly
        return peaks_to_spectrum(
            d["centers"], d["amps"], points=int(d["points"]),
            ppm_from=float(d["ppm_from"]), ppm_to=float(d["ppm_to"]),
            linewidth_hz=float(d["linewidth_hz"]), field_mhz=float(d["field_mhz"]),
            normalize=True)
    if "idx" in keys:                           # sparse broadened
        y = np.zeros(int(d["n"]), dtype=np.float32)
        y[d["idx"]] = d["val"]
        return y
    raise ValueError(f"unrecognised spectrum npz: keys={sorted(keys)}")
