"""mol_to_spin_system/augment.py — randomized chemical-shift sampling.

The NMRShiftDB HOSE predictor returns one deterministic shift per environment,
so the same molecule always yields the same spectrum.  For data augmentation we
instead sample each spin group's shift from ``N(mean, sigma)``, where *sigma* is
derived from the empirical spread (min/max) of that HOSE environment in the
database (stored by Task 2 as ``shift_range``)::

    sigma = clip((max - min) / k, floor, cap)

Why floor and cap (measured on real molecules):

* The raw spread is usually tiny (median ~0.07 ppm) and ~28% of environments
  are single-observation (spread 0), so the spread alone barely randomizes.
  A **floor** injects realistic jitter reflecting the predictor's intrinsic
  ~0.2 ppm ¹H error, so even well-determined shifts vary.
* A few many-observation environments have huge spreads (up to ~17 ppm); a
  **cap** keeps those from producing nonsense.

Sample at simulation/training time so every epoch sees a fresh draw.
"""

from __future__ import annotations

import numpy as np

#: range -> sigma divisor ((max-min) ~ 4 sigma covers ~95% of a normal)
DEFAULT_K = 4.0
#: minimum sigma (ppm) — the predictor's intrinsic uncertainty floor
DEFAULT_FLOOR = 0.05
#: maximum sigma (ppm) — tame rare wide-spread environments
DEFAULT_CAP = 0.4
#: clip sampled shifts to a plausible ¹H window
SHIFT_CLIP = (-1.0, 13.0)

#: per-type coupling sigma (Hz).  Unlike shifts, J has no database spread — it
#: comes from literature tables/equations — so the jitter reflects each
#: mechanism's typical literature uncertainty (geminal ²J varies most;
#: long-range ⁴J least).  Tunable at sampling time.
COUPLING_SIGMA = {
    "geminal":    1.2,
    "vicinal":    1.0,
    "olefinic":   1.0,
    "aromatic":   0.6,
    "long_range": 0.3,
}
#: sigma for a coupling whose type is missing/unknown
DEFAULT_COUPLING_SIGMA = 0.8


def shift_sigma(
    lo: float, hi: float,
    k: float = DEFAULT_K, floor: float = DEFAULT_FLOOR, cap: float = DEFAULT_CAP,
) -> float:
    """Per-group sampling sigma from its [min, max] spread."""
    return float(np.clip((hi - lo) / k, floor, cap))


def sample_shifts(
    means,
    ranges,
    *,
    k: float = DEFAULT_K,
    floor: float = DEFAULT_FLOOR,
    cap: float = DEFAULT_CAP,
    clip: tuple[float, float] | None = SHIFT_CLIP,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomized shifts ``N(mean_i, sigma_i)`` for every group.

    Parameters
    ----------
    means: (n,) group mean shifts (ppm).
    ranges: (n, 2) per-group ``[min, max]`` spread.
    k, floor, cap: sigma = clip((max-min)/k, floor, cap).
    clip: optional (lo, hi) ppm window to clamp the sampled shift.
    rng: numpy Generator (pass one seeded per-sample for reproducibility).
    """
    rng = rng if rng is not None else np.random.default_rng()
    means = np.asarray(means, dtype=float)
    ranges = np.asarray(ranges, dtype=float).reshape(-1, 2)
    sigma = np.clip((ranges[:, 1] - ranges[:, 0]) / k, floor, cap)
    out = rng.normal(means, sigma)
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


def sample_couplings(
    jvals,
    types,
    *,
    sigma: dict[str, float] = COUPLING_SIGMA,
    default: float = DEFAULT_COUPLING_SIGMA,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomized couplings ``N(J_i, sigma_type_i)`` for every group pair."""
    rng = rng if rng is not None else np.random.default_rng()
    jvals = np.asarray(jvals, dtype=float)
    sig = np.array([sigma.get(t, default) for t in types], dtype=float)
    return rng.normal(jvals, sig)


def sample_record(
    record: dict,
    *,
    rng: np.random.Generator | None = None,
    shift_kw: dict | None = None,
    coupling_kw: dict | None = None,
) -> dict:
    """Return a copy of *record* with randomized shifts and couplings.

    Shifts are drawn from the stored ``shift_range`` (falls back to floor-only
    jitter if absent); couplings from their per-type sigma (falls back to the
    default sigma if ``coupling_types`` is absent).  Pass one *rng* for a
    reproducible draw.
    """
    rng = rng if rng is not None else np.random.default_rng()
    out = dict(record)

    sg = record["spin_groups"]
    means = [g[0] for g in sg]
    ranges = record.get("shift_range") or [[m, m] for m in means]
    shifts = sample_shifts(means, ranges, rng=rng, **(shift_kw or {}))
    out["spin_groups"] = [[round(float(s), 3), int(g[1])]
                          for s, g in zip(shifts, sg)]

    coups = record.get("couplings", [])
    if coups:
        types = record.get("coupling_types") or [None] * len(coups)
        jvals = sample_couplings([c[2] for c in coups], types,
                                 rng=rng, **(coupling_kw or {}))
        out["couplings"] = [[c[0], c[1], round(float(j), 2)]
                            for c, j in zip(coups, jvals)]
    return out
