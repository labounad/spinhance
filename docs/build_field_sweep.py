#!/usr/bin/env python3
"""
precompute_field_sweep.py
=========================
Generate the compact JSON dataset that powers the Spinhance website's
scroll-driven "field sweep" hero animation.

For a curated subset of molecules from docs/data/spin_systems_pubchem.json (a random
1000-molecule sample of the PubChem set; see sample_pubchem_subset.py) we
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
N_MOLECULES = 40         # how many molecules to ship
SAMPLE_SEED = 42         # RNG seed for reproducible random molecule selection
MAX_FRAGMENT_SPINS = 11  # skip very large coupled fragments (slow / huge)
SIGNAL_FRAC = 2e-3       # signal-extent threshold (fraction of peak) for windowing
SCAN_POINTS = 16384      # resolution of the full-range scan that finds the window

OUT = REPO / "docs" / "data" / "field_sweep.json"
# Spin-system source: randomized ChEMBL 8-spin dataset with Gaussian-jittered
# shifts and couplings (replaces the PubChem 1000-molecule pool).
SPIN = REPO / "mol_to_spin_system" / "data" / "spin_systems_chembl_8spin_randomized.json"
# Precomputed 3D coordinates (force-field embedded) for the 3D structure view —
# the full ChEMBL 8-spin set from molecular generation. We stream it and keep
# only the blocks for the chosen molecules (see build_xyz_index).
XYZ_CANDIDATES = [REPO / "generate" / "data" / "buckets" / "chembl_8spin.xyz.gz",
                  REPO / "generate" / "data" / "buckets" / "chembl_8spin.xyz"]


def _open_xyz(path):
    """Open an .xyz or .xyz.gz file as a text stream."""
    if str(path).endswith(".gz"):
        import gzip
        return gzip.open(path, "rt")
    return open(path, "r")


def build_xyz_index(wanted_chembl, wanted_smiles, n_target):
    """Stream the (huge, gzipped) XYZ file and index ONLY the chosen molecules.

    Each block is: a count line, a JSON-meta line, then `natoms` coordinate rows.
    H rows may carry trailing spin-group labels; we keep only element + x/y/z so
    3Dmol.js parses cleanly (it infers bonds by distance). The full PubChem XYZ is
    ~1.6 GB gzipped, so we never load it whole — we stream block-by-block, retain
    only matches, and stop early once every chosen molecule is found.
    """
    def _is_real_file(p):
        """Return False if p is a Git LFS pointer (not actual binary data)."""
        try:
            with open(p, "rb") as fh:
                return not fh.read(7).startswith(b"version")
        except OSError:
            return False

    path = next((p for p in XYZ_CANDIDATES if p.exists() and _is_real_file(p)), None)
    by_chembl, by_smiles = {}, {}
    if path is None:
        print("WARNING: XYZ file not found or is an LFS pointer; 3D structures will be omitted")
        return by_chembl, by_smiles
    found = 0
    with _open_xyz(path) as fh:
        while found < n_target:
            header = fh.readline()
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            natoms = int(header)
            meta_line = fh.readline()
            coord_lines = [fh.readline() for _ in range(natoms)]
            try:
                meta = json.loads(meta_line)
            except Exception:
                continue
            cid, smi = meta.get("chembl_id"), meta.get("smiles")
            if cid not in wanted_chembl and smi not in wanted_smiles:
                continue
            clean = []
            for ln in coord_lines:
                p = ln.split()
                if len(p) >= 4:
                    clean.append(f"{p[0]} {p[1]} {p[2]} {p[3]}")
            xyz = f"{natoms}\n{cid or ''}\n" + "\n".join(clean)
            if cid:
                by_chembl[cid] = xyz
            if smi:
                by_smiles[smi] = xyz
            found += 1
    print(f"indexed {found}/{n_target} chosen-molecule XYZ structures from {path.name}")
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
    import random
    rng = random.Random(SAMPLE_SEED)

    records = json.loads(SPIN.read_text())
    eligible = []
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
        eligible.append((rec, shifts, couplings, deg))

    chosen_raw = rng.sample(eligible, min(N_MOLECULES, len(eligible)))
    chosen = [(rec, shifts, couplings, deg) for rec, shifts, couplings, deg in chosen_raw]
    print(f"{len(records)} records -> {len(eligible)} eligible -> {len(chosen)} chosen (seed={SAMPLE_SEED})")

    wanted_chembl = {rec.get("chembl_id") for rec, *_ in chosen if rec.get("chembl_id")}
    wanted_smiles = {rec.get("smiles") for rec, *_ in chosen if rec.get("smiles")}
    xyz_chembl, xyz_smiles = build_xyz_index(wanted_chembl, wanted_smiles, len(chosen))
    fields = geometric_fields(LOW_MHZ, HIGH_MHZ, N_FIELDS)
    molecules = []
    max_sticks = 0
    n_xyz = 0
    for rank, (rec, shifts, couplings, deg) in enumerate(chosen):
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
        print(f"  [{rank+1:2d}] {rec.get('chembl_id'):12s} "
              f"win=[{win_lo:.2f},{win_hi:.2f}] smiles={rec.get('smiles')[:40]}")
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
            "source": "pyspin composite simulator; 40 random molecules from randomized ChEMBL 8-spin dataset",
        },
        "molecules": molecules,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    size = OUT.stat().st_size
    print(f"\nwrote {OUT}  ({size/1e6:.2f} MB, {len(molecules)} molecules x {N_FIELDS} fields)")


if __name__ == "__main__":
    main()
