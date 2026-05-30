#!/usr/bin/env python3
"""
precompute_field_sweep.py
=========================
Generate the compact JSON dataset that powers the SpinHance website's
scroll-driven "field sweep" hero animation.

For a curated subset of molecules from mol_to_matrix/data/spin_systems.json we
run the pure-Python pyspin composite simulator across a GEOMETRIC sweep of
spectrometer fields (90 -> 600 MHz) and store downsampled, quantized intensity
arrays. The site interpolates between frames as the user scrolls.

Output: docs/data/field_sweep.json
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

# This script lives in <repo>/docs/ ; the repo root is one level up.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from simulation.graph_io import record_to_arrays, molecule_id  # noqa: E402
from simulation.pyspin.composite import (  # noqa: E402
    simulate_spectrum_composite,
    largest_component_spins,
)

# ── config ────────────────────────────────────────────────────────────────────
LOW_MHZ, HIGH_MHZ = 90.0, 600.0
N_FIELDS = 16            # geometric frames between low and high field
PPM_FROM, PPM_TO = 0.0, 12.0
SIM_POINTS = 32768       # 2**15 native resolution, simulated over the TIGHT window
DISP_POINTS = 2048       # stored points across the (narrow) per-molecule window
LINEWIDTH_HZ = 1.3       # crisp lines for high-field resolution
N_MOLECULES = 30         # how many molecules to ship
MAX_FRAGMENT_SPINS = 11  # skip very large coupled fragments (slow / huge)
SIGNAL_FRAC = 2e-3       # signal-extent threshold (fraction of peak) for windowing
SCAN_POINTS = 16384      # resolution of the full-range scan that finds the window

SPIN = REPO / "mol_to_matrix" / "data" / "spin_systems.json"
OUT = REPO / "docs" / "data" / "field_sweep.json"


def geometric_fields(lo, hi, n):
    return [round(float(lo * (hi / lo) ** (i / (n - 1))), 2) for i in range(n)]


def maxpool_downsample(y, target):
    """Downsample by max-pooling so sharp peaks survive."""
    n = len(y)
    factor = n // target
    usable = factor * target
    head = y[:usable].reshape(target, factor).max(axis=1)
    if usable < n:  # fold any remainder into the last bin
        head[-1] = max(head[-1], y[usable:].max())
    return head


def signal_window(shifts, couplings, deg):
    """Data-driven ppm window: where protons actually resonate.

    Scan the broad LOW-field spectrum (widest peaks) over the full 0-12 ppm range
    and return [lo, hi] covering all signal above SIGNAL_FRAC of the peak, padded.
    Falls back to the shift range if the scan is empty.
    """
    _, spec = simulate_spectrum_composite(
        shifts, couplings, deg, LOW_MHZ,
        points=SCAN_POINTS, ppm_from=PPM_FROM, ppm_to=PPM_TO, linewidth_hz=LINEWIDTH_HZ,
    )
    ppm = np.linspace(PPM_FROM, PPM_TO, SCAN_POINTS)
    thr = spec.max() * SIGNAL_FRAC
    idx = np.where(spec > thr)[0]
    if len(idx):
        lo, hi = ppm[idx[0]], ppm[idx[-1]]
    else:
        lo, hi = min(shifts), max(shifts)
    pad = max(0.18, 0.05 * (hi - lo))
    lo, hi = max(PPM_FROM, lo - pad), min(PPM_TO, hi + pad)
    if hi - lo < 0.8:                       # don't over-zoom a lone singlet
        c = 0.5 * (lo + hi); lo, hi = c - 0.45, c + 0.45
    return round(float(lo), 3), round(float(hi), 3)


def second_order_score(shifts, couplings, deg):
    """Higher = more dramatic, OBSERVABLE low->high field change.

    The sweet spot is a pair of groups with DISTINCT but close shifts strongly
    coupled to each other: overlapped/second-order at 90 MHz, cleanly resolved at
    600 MHz. Couplings between groups at the same shift are NMR-silent (equivalent
    spins don't split each other) and must NOT count. We also reward shift
    diversity and spread so the spectrum has several visible multiplets.
    """
    n = len(shifts)
    score = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            J = abs(couplings[i][j])
            if J < 1.0:
                continue
            dnu_lo = abs(shifts[i] - shifts[j]) * LOW_MHZ   # Hz @ 90
            if dnu_lo < 1.0:
                continue  # effectively equivalent shift -> no observable splitting
            ratio = dnu_lo / J
            # peak when ratio ~ 1-4: second order at 90, resolved at 600
            score += J * math.exp(-((math.log(ratio / 2.0)) ** 2) / 1.2)
    # diversity: number of distinct shift clusters (0.05 ppm tolerance)
    clusters = []
    for s in sorted(shifts):
        if not clusters or abs(s - clusters[-1]) > 0.05:
            clusters.append(s)
    score *= (1.0 + 0.12 * len(clusters))
    return score


def main():
    records = json.loads(SPIN.read_text())
    scored = []
    for rec in records:
        try:
            labels, shifts, couplings, deg = record_to_arrays(rec)
        except Exception:
            continue
        # keep things fast & web-friendly
        if largest_component_spins(couplings, deg) > MAX_FRAGMENT_SPINS:
            continue
        # require signal spread inside a sensible window
        lo, hi = min(shifts), max(shifts)
        if lo < 0.3 or hi > 11.5:
            continue
        s = second_order_score(shifts, couplings, deg)
        if s <= 0:
            continue
        scored.append((s, rec, shifts, couplings, deg))

    scored.sort(key=lambda t: t[0], reverse=True)
    chosen = scored[:N_MOLECULES]
    print(f"{len(records)} records -> {len(scored)} eligible -> {len(chosen)} chosen")

    fields = geometric_fields(LOW_MHZ, HIGH_MHZ, N_FIELDS)
    molecules = []
    for rank, (score, rec, shifts, couplings, deg) in enumerate(chosen):
        win_lo, win_hi = signal_window(shifts, couplings, deg)
        frames = []
        for f in fields:
            # simulate at high native resolution OVER THE TIGHT WINDOW so every
            # stored point lands where the protons are (max resolution per byte).
            _, spec = simulate_spectrum_composite(
                shifts, couplings, deg, f,
                points=SIM_POINTS, ppm_from=win_lo, ppm_to=win_hi,
                linewidth_hz=LINEWIDTH_HZ,
            )
            ds = maxpool_downsample(spec, DISP_POINTS)
            # PER-FRAME normalization: each field's tallest peak -> full scale, so
            # peak HEIGHTS stay ~constant across the sweep and the eye tracks the
            # change in resolution / splitting rather than overall amplitude.
            m = float(ds.max())
            q = np.clip(np.round(ds / m * 65535.0), 0, 65535).astype("<u2") if m > 0 \
                else np.zeros(DISP_POINTS, dtype="<u2")
            frames.append(base64.b64encode(q.tobytes()).decode("ascii"))

        molecules.append({
            "id": molecule_id(rec),
            "chembl_id": rec.get("chembl_id"),
            "smiles": rec.get("smiles"),
            "n_groups": len(shifts),
            "degeneracy": [int(d) for d in deg],
            "shifts": [round(float(s), 3) for s in shifts],
            "couplings": [[round(float(couplings[i][j]), 1) for j in range(len(shifts))]
                          for i in range(len(shifts))],
            "win": [win_lo, win_hi],     # ppm display window (data-driven)
            "frames": frames,            # base64 uint16 LE, len DISP_POINTS each
        })
        print(f"  [{rank+1:2d}] {rec.get('chembl_id'):12s} score={score:6.1f} "
              f"win=[{win_lo:.2f},{win_hi:.2f}] smiles={rec.get('smiles')[:36]}")

    payload = {
        "meta": {
            "low_mhz": LOW_MHZ,
            "high_mhz": HIGH_MHZ,
            "fields_mhz": fields,
            "n_fields": N_FIELDS,
            "disp_points": DISP_POINTS,
            "ppm_from": PPM_FROM,
            "ppm_to": PPM_TO,
            "linewidth_hz": LINEWIDTH_HZ,
            "windowed": True,
            "encoding": "base64-uint16le-per-frame-normalized (points span molecule.win)",
            "source": "pyspin composite simulator over mol_to_matrix/data/spin_systems.json",
        },
        "molecules": molecules,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size = OUT.stat().st_size
    print(f"\nwrote {OUT}  ({size/1e6:.2f} MB, {len(molecules)} molecules x {N_FIELDS} fields)")


if __name__ == "__main__":
    main()
