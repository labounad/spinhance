"""
model.data.regions
===================
Support-region tokenization for 90 MHz ¹H NMR spectra (IDEAS Families D/E/H).
A spectrum is read as a set of spectral *objects* (contiguous above-baseline
regions) rather than 16384 unrelated bins. Each region becomes a fixed-length
token: 16 metadata scalars (incl. raw + relative INTEGRATION — the proton-count
cue, never normalized away) followed by a 64-pt locally-peak-normalized window
(local shape / multiplicity). The architecture's region branch consumes these
alongside the global spectral context.

Design constraints (IDEAS D): keep raw + relative integral; do NOT assume one
region == one spin group; the model always retains global context, so missing a
weak region is non-fatal.

Per-item output is already padded to ``max_regions`` so the collate just stacks:
  features (R_max, F=80) float32, mask (R_max,) float32 {1=real, 0=pad}.
"""
from __future__ import annotations

import numpy as np

N_SCALAR = 16
WINDOW = 64
FEAT_DIM = N_SCALAR + WINDOW          # 80


def _contiguous_regions(above: np.ndarray):
    """Start/end (exclusive) index pairs of contiguous True runs in a bool array."""
    if not above.any():
        return []
    d = np.diff(above.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(above)]
    return list(zip(starts, ends))


def _merge_and_filter(regions, min_gap_pts, margin_pts, min_width_pts, P):
    """Merge regions separated by < min_gap, expand by margin, drop too-narrow."""
    if not regions:
        return []
    regions = sorted(regions)
    merged = [list(regions[0])]
    for s, e in regions[1:]:
        if s - merged[-1][1] < min_gap_pts:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if e - s < min_width_pts:
            continue
        out.append((max(0, s - margin_pts), min(P, e + margin_pts)))
    return out


def _region_features(spec, s, e, ppm_from, dppm, total_sum, global_max):
    """16 metadata scalars + 64-pt window for region [s, e)."""
    seg = np.clip(spec[s:e].astype(np.float64), 0.0, None)
    width_pts = e - s
    seg_sum = float(seg.sum())
    center_ppm = ppm_from + (s + e) * 0.5 * dppm
    start_ppm = ppm_from + s * dppm
    end_ppm = ppm_from + e * dppm
    width_ppm = width_pts * dppm

    # intensity-weighted shape moments over local position
    pos = np.arange(width_pts, dtype=np.float64)
    w = seg / (seg_sum + 1e-12)
    mean = float((w * pos).sum())
    var = float((w * (pos - mean) ** 2).sum())
    std = var ** 0.5
    if std > 1e-6:
        z = (pos - mean) / std
        skew = float((w * z ** 3).sum())
        kurt = float((w * z ** 4).sum()) - 3.0
    else:
        skew = kurt = 0.0

    # local maxima count (3-pt, above 10% of the region's own peak)
    seg_max = float(seg.max()) if width_pts else 0.0
    if width_pts >= 3 and seg_max > 0:
        thr = 0.1 * seg_max
        lm = (seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:]) & (seg[1:-1] > thr)
        n_max = int(lm.sum()) + 1
    else:
        n_max = 1

    raw_integral = seg_sum * dppm                      # ≈ proton fraction (unit-integral spec)
    rel_integral = seg_sum / (total_sum + 1e-12)
    argmax_local = float(np.argmax(seg)) / max(1, width_pts - 1) if width_pts else 0.0
    centroid_ppm = start_ppm + mean * dppm

    scalars = np.array([
        center_ppm / 12.0, start_ppm / 12.0, end_ppm / 12.0, width_ppm / 12.0,
        raw_integral, rel_integral,
        seg_max / (global_max + 1e-12), float(seg.mean()) / (global_max + 1e-12),
        min(n_max, 20) / 20.0, std / max(1.0, width_pts),
        np.tanh(skew), np.tanh(kurt),
        argmax_local, centroid_ppm / 12.0,
        np.log1p(seg_sum), width_pts / 200.0,
    ], dtype=np.float32)

    # 64-pt locally peak-normalized window (resample region shape)
    if width_pts >= 2:
        xp = np.linspace(0, width_pts - 1, WINDOW)
        win = np.interp(xp, np.arange(width_pts), seg)
    else:
        win = np.full(WINDOW, seg[0] if width_pts else 0.0)
    win = (win / (seg_max + 1e-12)).astype(np.float32)
    return np.concatenate([scalars, win]), raw_integral


def extract_support_regions(spec, ppm_from=0.0, ppm_to=12.0, *, threshold_frac=0.02,
                            min_gap_pts=8, margin_pts=4, min_width_pts=4,
                            max_regions=48):
    """Spectrum (P,) -> (features (max_regions, 80), mask (max_regions,)).

    Regions are kept by descending raw integral and padded/truncated to
    ``max_regions``. Robust to empty spectra (returns all-pad)."""
    spec = np.asarray(spec, dtype=np.float64)
    P = spec.shape[0]
    dppm = (ppm_to - ppm_from) / P
    gmax = float(spec.max()) if P else 0.0
    feats = np.zeros((max_regions, FEAT_DIM), dtype=np.float32)
    mask = np.zeros((max_regions,), dtype=np.float32)
    if gmax <= 0:
        return feats, mask

    above = spec > (threshold_frac * gmax)
    regions = _merge_and_filter(_contiguous_regions(above),
                                min_gap_pts, margin_pts, min_width_pts, P)
    if not regions:
        return feats, mask

    total_sum = float(spec.sum())
    built = [_region_features(spec, s, e, ppm_from, dppm, total_sum, gmax)
             for s, e in regions]
    built.sort(key=lambda fr: fr[1], reverse=True)          # by raw integral desc
    for j, (fv, _) in enumerate(built[:max_regions]):
        feats[j] = fv
        mask[j] = 1.0
    return feats, mask
