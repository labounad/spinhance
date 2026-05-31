#!/usr/bin/env python3
"""
precompute_field_sweep.py
=========================
Generate the compact JSON dataset that powers the Spinhance website's
scroll-driven "field sweep" hero animation.

For a curated subset of molecules from mol_to_spin_system/data/spin_systems.json we
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
    system_transitions,
    _components,
)

# ── config ────────────────────────────────────────────────────────────────────
LOW_MHZ, HIGH_MHZ = 90.0, 600.0
N_FIELDS = 16            # geometric frames between low and high field
PPM_FROM, PPM_TO = 0.0, 12.0
SIM_POINTS = 32768       # 2**15 — simulated AND STORED over the TIGHT window
DISP_POINTS = SIM_POINTS # store every simulated point (no downsampling) -> smooth peaks
LINEWIDTH_HZ = 1.1       # crisp lines for high-field resolution
N_MOLECULES = 30         # how many molecules to ship
MAX_FRAGMENT_SPINS = 11  # skip very large coupled fragments (slow / huge)
SIGNAL_FRAC = 2e-3       # signal-extent threshold (fraction of peak) for windowing
SCAN_POINTS = 16384      # resolution of the full-range scan that finds the window

OUT = REPO / "docs" / "data" / "field_sweep.json"
# Spin-system source. Prefer the self-contained copy in docs/data so the site can
# be rebuilt from docs/ alone; fall back to the canonical mol_to_spin_system location.
SPIN = next((p for p in [REPO / "docs" / "data" / "spin_systems.json",
                         REPO / "mol_to_spin_system" / "data" / "spin_systems.json"] if p.exists()),
            REPO / "mol_to_spin_system" / "data" / "spin_systems.json")
# Precomputed 3D coordinates (force-field embedded) for the 3D structure view.
# Prefer the in-docs copy, then the working copy, then the tracked backup.
XYZ_CANDIDATES = [REPO / "docs" / "data" / "chembl_8spin.xyz",
                  REPO / "generate" / "data" / "chembl_8spin.xyz",
                  REPO / "data" / "chembl_8spin.xyz"]


def build_xyz_index():
    """Index chembl_8spin.xyz by chembl_id and smiles -> cleaned XYZ block (element+coords).

    The source file's H rows carry trailing spin-group labels ("A N"); we keep only
    the element symbol and x/y/z so 3Dmol.js parses it cleanly (it infers bonds).
    """
    path = next((p for p in XYZ_CANDIDATES if p.exists()), None)
    by_chembl, by_smiles = {}, {}
    if path is None:
        print("WARNING: chembl_8spin.xyz not found; 3D structures will be omitted")
        return by_chembl, by_smiles
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1; continue
        natoms = int(lines[i].strip())
        meta = json.loads(lines[i + 1])
        clean = []
        for ln in lines[i + 2: i + 2 + natoms]:
            p = ln.split()
            clean.append(f"{p[0]} {p[1]} {p[2]} {p[3]}")
        xyz = f"{natoms}\n{meta.get('chembl_id', '')}\n" + "\n".join(clean)
        if meta.get("chembl_id"):
            by_chembl[meta["chembl_id"]] = xyz
        if meta.get("smiles"):
            by_smiles[meta["smiles"]] = xyz
        i += 2 + natoms
    print(f"indexed {len(by_chembl)} XYZ structures from {path.name}")
    return by_chembl, by_smiles


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


def molecule_sticks(shifts, couplings, deg, field, win_lo, win_hi):
    """Merged STICK spectrum (centers_ppm, amps) for the whole molecule at `field`.

    Replicates simulate_spectrum_composite's per-component proton renormalization,
    but returns the discrete transitions instead of a broadened curve. The website
    broadens these into smooth Lorentzians live (resolution-independent, tiny data).
    """
    G = len(shifts)
    cfs, cas = [], []
    for comp in _components(couplings, G):
        sub_shifts = [shifts[g] for g in comp]
        sub_J = [[couplings[a][b] for b in comp] for a in comp]
        sub_deg = [deg[g] for g in comp]
        cf, ca = system_transitions(sub_shifts, sub_J, sub_deg, field)
        if not len(cf):
            continue
        comp_protons = sum(int(deg[g]) for g in comp)
        raw = ca.sum()
        if raw > 0:
            ca = ca * (comp_protons / raw)
        cfs.append(cf); cas.append(ca)
    if not cfs:
        return np.array([]), np.array([])
    centers = np.concatenate(cfs) / field          # Hz -> ppm
    amps = np.concatenate(cas)
    margin = 0.04 * (win_hi - win_lo)
    keep = (centers >= win_lo - margin) & (centers <= win_hi + margin) & (amps > 0)
    centers, amps = centers[keep], amps[keep]
    if not len(centers):
        return centers, amps
    order = np.argsort(centers)
    centers, amps = centers[order], amps[order]
    tol = 0.25 / field                              # merge lines within ~0.25 Hz
    mc, ma = [centers[0]], [amps[0]]
    for c, a in zip(centers[1:], amps[1:]):
        if c - mc[-1] < tol:
            tot = ma[-1] + a
            mc[-1] = (mc[-1] * ma[-1] + c * a) / tot
            ma[-1] = tot
        else:
            mc.append(c); ma.append(a)
    mc, ma = np.array(mc), np.array(ma)
    thr = ma.max() * 5e-4                            # drop negligible lines
    k = ma > thr
    return mc[k], ma[k]


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

    xyz_chembl, xyz_smiles = build_xyz_index()
    fields = geometric_fields(LOW_MHZ, HIGH_MHZ, N_FIELDS)
    molecules = []
    max_sticks = 0
    n_xyz = 0
    for rank, (score, rec, shifts, couplings, deg) in enumerate(chosen):
        win_lo, win_hi = signal_window(shifts, couplings, deg)
        xyz = xyz_chembl.get(rec.get("chembl_id")) or xyz_smiles.get(rec.get("smiles"))
        if xyz:
            n_xyz += 1
        frames = []
        for f in fields:
            mc, ma = molecule_sticks(shifts, couplings, deg, f, win_lo, win_hi)
            max_sticks = max(max_sticks, len(mc))
            # amplitudes -> uint16 normalized to this frame's largest line
            amax = float(ma.max()) if len(ma) else 1.0
            cen = np.asarray(mc, dtype="<f4")
            amp = np.clip(np.round(ma / amax * 65535.0), 0, 65535).astype("<u2")
            frames.append({
                "c": base64.b64encode(cen.tobytes()).decode("ascii"),   # ppm float32
                "a": base64.b64encode(amp.tobytes()).decode("ascii"),   # uint16 / 65535
            })

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
            "xyz": xyz,                  # precomputed 3D coords (XYZ block) or None
            "frames": frames,            # per field: {c: centers ppm, a: amps} (base64)
        })
        print(f"  [{rank+1:2d}] {rec.get('chembl_id'):12s} score={score:6.1f} "
              f"win=[{win_lo:.2f},{win_hi:.2f}] smiles={rec.get('smiles')[:34]}")
    print(f"max sticks/frame = {max_sticks}; molecules with 3D coords = {n_xyz}/{len(chosen)}")

    payload = {
        "meta": {
            "low_mhz": LOW_MHZ,
            "high_mhz": HIGH_MHZ,
            "fields_mhz": fields,
            "n_fields": N_FIELDS,
            "ppm_from": PPM_FROM,
            "ppm_to": PPM_TO,
            "linewidth_hz": LINEWIDTH_HZ,
            "format": "sticks",
            "encoding": "per frame {c: base64 float32 centers ppm, a: base64 uint16 amps/65535}; broaden client-side",
            "source": "pyspin composite simulator over mol_to_spin_system/data/spin_systems.json",
        },
        "molecules": molecules,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size = OUT.stat().st_size
    print(f"\nwrote {OUT}  ({size/1e6:.2f} MB, {len(molecules)} molecules x {N_FIELDS} fields)")


if __name__ == "__main__":
    main()
