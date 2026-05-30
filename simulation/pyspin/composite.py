"""
pyspin.composite
================
Exact 1H simulator with COMPOSITE-PARTICLE REDUCTION for magnetically
equivalent groups — the scalable engine.

Why
---
Expanding a CH3 to 3 spins or a tert-butyl to 9 spins blows up the 2^N Hilbert
space. But equivalent spins in a group all share one shift and couple
identically to everything else, so the group's *total spin* is what matters. A
group of d spin-½ decomposes into total-spin manifolds S with multiplicities

    w(d, S) = C(d, d/2 - S) - C(d, d/2 - S - 1)

(e.g. CH3: S=3/2 ×1 and S=1/2 ×2). The full spectrum is the multiplicity-
weighted sum over every combination of per-group total spins, where each group
is treated as a single particle of spin S_g. This collapses, e.g., a tert-butyl
from 2^9 = 512 down to a handful of small manifolds.

Each combination is simulated by the same exact method as pyspin.simulator
(Mz-block diagonalisation, F+ detection), generalised to arbitrary spin-S
particles. Correctness is checked against the spin-½ simulator in the tests.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

__all__ = ["spin_reps", "simulate_spectrum_composite"]


def spin_reps(d: int) -> list[tuple[float, int]]:
    """Total-spin manifolds (S, multiplicity) for ``d`` equivalent spin-½."""
    reps = []
    S = d / 2.0
    while S >= -1e-9:
        k = d / 2.0 - S
        ki = int(round(k))
        w = math.comb(d, ki) - (math.comb(d, ki - 1) if ki >= 1 else 0)
        if w > 0:
            reps.append((S, w))
        S -= 1.0
    return reps


def _ops(S: float):
    """Return (m_values, raise_amp) for a spin-S particle.

    Index i corresponds to m = S - i (i = 0 is the top state, m = +S).
    raise_amp[i] = <i-1|S+|i> = sqrt(S(S+1) - m(m+1)) with m = S - i.
    """
    n = int(round(2 * S + 1))
    m = np.array([S - i for i in range(n)])
    aplus = np.zeros(n)
    for i in range(1, n):
        mi = m[i]
        aplus[i] = math.sqrt(max(0.0, S * (S + 1) - mi * (mi + 1)))
    return m, aplus


def _simulate_combination(S_list, nu_hz, Jgg, intensity_threshold):
    """Exact sim of one mixed-spin combination. Returns (freqs, amps) arrays.

    S_list[g] = total spin of group g; nu_hz[g] = its offset (Hz);
    Jgg[g][h] = inter-group coupling (Hz). Groups with S=0 are inert.
    """
    G = len(S_list)
    mvals = []
    aplus = []
    dims = []
    for S in S_list:
        m, ap = _ops(S)
        mvals.append(m)
        aplus.append(ap)
        dims.append(len(m))

    # Enumerate product basis as tuples of per-group indices, grouped by total Mz.
    blocks: dict[float, list[tuple]] = {}
    for state in product(*[range(d) for d in dims]):
        Mz = sum(mvals[g][state[g]] for g in range(G))
        blocks.setdefault(round(Mz, 6), []).append(state)

    # Diagonalise each Mz block.
    E = {}
    V = {}
    idx = {}
    for Mz, states in blocks.items():
        index = {s: i for i, s in enumerate(states)}
        idx[Mz] = index
        dim = len(states)
        H = np.zeros((dim, dim))
        for a, s in enumerate(states):
            ms = [mvals[g][s[g]] for g in range(G)]
            diag = sum(nu_hz[g] * ms[g] for g in range(G))
            for g in range(G):
                for h in range(g + 1, G):
                    diag += Jgg[g][h] * ms[g] * ms[h]
            H[a, a] = diag
            # flip-flop: raise g (i_g -> i_g-1), lower h (i_h -> i_h+1)
            for g in range(G):
                if s[g] == 0:
                    continue
                for h in range(G):
                    if h == g or s[h] >= dims[h] - 1:
                        continue
                    s2 = list(s)
                    s2[g] -= 1
                    s2[h] += 1
                    s2 = tuple(s2)
                    b = index[s2]
                    if b > a:
                        val = 0.5 * Jgg[g][h] * aplus[g][s[g]] * aplus[h][s[h] + 1]
                        H[a, b] += val
                        H[b, a] += val
        Eb, Vb = np.linalg.eigh(H)
        E[Mz] = Eb
        V[Mz] = Vb

    # Single-quantum transitions: F+ connects block Mz -> Mz+1.
    freqs, amps = [], []
    for Mz in sorted(blocks):
        up = round(Mz + 1, 6)
        if up not in blocks:
            continue
        lo_states = blocks[Mz]
        up_index = idx[up]
        Fplus = np.zeros((len(blocks[up]), len(lo_states)))
        for q, s in enumerate(lo_states):
            for g in range(G):
                if s[g] == 0:
                    continue  # cannot raise further
                s2 = list(s); s2[g] -= 1; s2 = tuple(s2)
                Fplus[up_index[s2], q] += aplus[g][s[g]]
        M = V[up].T @ Fplus @ V[Mz]
        inten = M * M
        df = E[up][:, None] - E[Mz][None, :]
        keep = inten > intensity_threshold
        freqs.append(df[keep])
        amps.append(inten[keep])

    if freqs:
        return np.concatenate(freqs), np.concatenate(amps)
    return np.array([]), np.array([])


def simulate_spectrum_composite(
    shifts,
    couplings,
    degeneracy,
    field_mhz,
    points=16384,
    ppm_from=0.0,
    ppm_to=12.0,
    linewidth_hz=1.0,
    intensity_threshold=1e-6,
):
    """Simulate a 1H spectrum using composite-particle reduction.

    Same signature/return as pyspin.simulator.simulate_spectrum:
    returns (ppm_axis, intensity) normalised to unit integral.
    """
    G = len(shifts)
    nu_hz = [shifts[g] * field_mhz for g in range(G)]
    reps = [spin_reps(int(degeneracy[g])) for g in range(G)]

    all_freqs, all_amps = [], []
    # Iterate every combination of per-group total spins; weight by multiplicities.
    for combo in product(*reps):
        S_list = [c[0] for c in combo]
        weight = 1
        for c in combo:
            weight *= c[1]
        f, a = _simulate_combination(S_list, nu_hz, couplings, intensity_threshold)
        if len(f):
            all_freqs.append(f)
            all_amps.append(a * weight)

    freqs = np.concatenate(all_freqs) if all_freqs else np.array([])
    amps = np.concatenate(all_amps) if all_amps else np.array([])

    ppm = np.linspace(ppm_from, ppm_to, points)
    spec = np.zeros(points)
    centers = freqs / field_mhz
    hwhm = (linewidth_hz / 2.0) / field_mhz
    chunk = 2000
    for s in range(0, len(centers), chunk):
        c = centers[s:s + chunk][None, :]
        a = amps[s:s + chunk][None, :]
        spec += (a / (1.0 + ((ppm[:, None] - c) / hwhm) ** 2)).sum(axis=1)

    dppm = (ppm_to - ppm_from) / points
    total = spec.sum() * dppm
    if total > 0:
        spec = spec / total
    return ppm, spec
