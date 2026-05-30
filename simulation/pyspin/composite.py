"""
pyspin.composite
================
Exact 1H simulator with COMPOSITE-PARTICLE REDUCTION for magnetically
equivalent groups — the scalable engine (vectorised).

Why
---
Expanding a CH3 to 3 spins or a tert-butyl to 9 spins blows up the 2^N Hilbert
space. Equivalent spins in a group share one shift and couple identically to
everything else, so only the group's *total spin* matters. A group of d
spin-½ decomposes into total-spin manifolds S with multiplicities

    w(d, S) = C(d, d/2 - S) - C(d, d/2 - S - 1)

(CH3: S=3/2 ×1 and S=1/2 ×2). The full spectrum is the multiplicity-weighted
sum over every combination of per-group total spins, each group treated as one
spin-S particle. This collapses, e.g., a tert-butyl from 2^9=512 to a handful
of small manifolds.

Each combination is solved by Mz-block diagonalisation with F+ detection,
generalised to arbitrary spin-S. The Hamiltonian is assembled with numpy
(einsum diagonal + searchsorted flip-flop lookups), and diagonalised with
scipy's BLAS-backed eigh when available.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from simulation.pyspin.simulator import peaks_to_spectrum

try:                                  # BLAS-backed when available
    from scipy.linalg import eigh as _eigh_impl

    def _eigh(H):
        return _eigh_impl(H, overwrite_a=True, check_finite=False)
except Exception:                     # numpy fallback
    def _eigh(H):
        return np.linalg.eigh(H)

__all__ = ["spin_reps", "simulate_spectrum_composite", "largest_component_spins",
           "system_transitions", "composite_transitions"]


def system_transitions(shifts, couplings, degeneracy, field_mhz,
                       intensity_threshold=1e-6):
    """Raw (freqs_Hz, amps) for ONE spin system via composite reduction.

    No component decomposition, no normalisation, no broadening — just the
    multiplicity-weighted transitions summed over per-group total-spin
    combinations. Shared by the exact engine and the clustered approximation.
    """
    nu_hz = [shifts[g] * field_mhz for g in range(len(shifts))]
    reps = [spin_reps(int(degeneracy[g])) for g in range(len(shifts))]
    fs, ams = [], []
    for combo in product(*reps):
        S_list = [c[0] for c in combo]
        weight = 1
        for c in combo:
            weight *= c[1]
        f, a = _simulate_combination(S_list, nu_hz, couplings, intensity_threshold)
        if len(f):
            fs.append(f); ams.append(a * weight)
    if fs:
        return np.concatenate(fs), np.concatenate(ams)
    return np.array([]), np.array([])


def largest_component_spins(couplings, degeneracy) -> int:
    """Largest connected-component size in *spins* (degeneracy summed).

    This is what bounds the exact-simulation cost: pyspin's Hilbert space for a
    molecule is set by its biggest coupled fragment, not its total proton count.
    Used by the ``auto`` engine router to decide pyspin vs MNova per molecule.
    """
    comps = _components(couplings, len(degeneracy))
    return max(sum(int(degeneracy[g]) for g in comp) for comp in comps)


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
    """(m_values, raise_amp) for a spin-S particle; index i ↔ m = S - i."""
    n = int(round(2 * S + 1))
    m = np.array([S - i for i in range(n)])
    aplus = np.zeros(n)
    for i in range(1, n):
        mi = m[i]
        aplus[i] = math.sqrt(max(0.0, S * (S + 1) - mi * (mi + 1)))
    return m, aplus


def _components(couplings, G):
    """Connected components of the coupling graph (edge where J != 0).

    Each component is an independent spin system; their spectra add.
    """
    seen = [False] * G
    comps = []
    for start in range(G):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            g = stack.pop()
            comp.append(g)
            for h in range(G):
                if not seen[h] and couplings[g][h] != 0.0:
                    seen[h] = True
                    stack.append(h)
        comps.append(sorted(comp))
    return comps


def _simulate_combination(S_list, nu_hz, Jgg, intensity_threshold):
    """Exact sim of one mixed-spin combination → (freqs_Hz, amps)."""
    # Drop inert S=0 groups (no shift, no coupling, no signal).
    active = [g for g, S in enumerate(S_list) if S > 0]
    G = len(active)
    if G == 0:
        return np.array([]), np.array([])

    nu = np.array([nu_hz[g] for g in active])
    J = np.array([[Jgg[active[a]][active[b]] for b in range(G)] for a in range(G)])
    mlist, aplist, dims = [], [], []
    for g in active:
        m, ap = _ops(S_list[g])
        mlist.append(m); aplist.append(ap); dims.append(len(m))
    dims = np.array(dims, dtype=np.int64)

    # Mixed-radix weights so each state tuple ↔ a unique integer key.
    weight = np.ones(G, dtype=np.int64)
    for g in range(G - 2, -1, -1):
        weight[g] = weight[g + 1] * dims[g + 1]
    total = int(np.prod(dims))

    allidx = np.array(list(np.ndindex(*[int(d) for d in dims])), dtype=np.int64)
    keys = allidx @ weight
    Mmat = np.empty((total, G))
    for g in range(G):
        Mmat[:, g] = mlist[g][allidx[:, g]]
    Mz = np.round(Mmat.sum(axis=1), 6)
    diag_all = Mmat @ nu + 0.5 * np.einsum("ag,gh,ah->a", Mmat, J, Mmat)

    # Group state indices by Mz.
    order_by_mz: dict[float, np.ndarray] = {}
    for mz in np.unique(Mz):
        order_by_mz[mz] = np.nonzero(Mz == mz)[0]

    # Diagonalise each block.
    E, V, blk_keys, blk_sorted, blk_argsort = {}, {}, {}, {}, {}
    for mz, gstates in order_by_mz.items():
        dim = len(gstates)
        kb = keys[gstates]
        args = np.argsort(kb)
        blk_keys[mz] = kb
        blk_sorted[mz] = kb[args]
        blk_argsort[mz] = args
        local = {int(k): i for i, k in enumerate(kb)}  # only used implicitly

        H = np.zeros((dim, dim))
        np.fill_diagonal(H, diag_all[gstates])
        ig_all = allidx[gstates]  # (dim, G)
        for g in range(G):
            for h in range(G):
                if g == h or J[g, h] == 0.0:
                    continue
                ig = ig_all[:, g]; ih = ig_all[:, h]
                mask = (ig > 0) & (ih < dims[h] - 1)
                if not mask.any():
                    continue
                src = np.nonzero(mask)[0]
                amp = 0.5 * J[g, h] * aplist[g][ig[mask]] * aplist[h][ih[mask] + 1]
                tgt_key = kb[mask] - weight[g] + weight[h]
                pos = np.searchsorted(blk_sorted[mz], tgt_key)
                tgt = args[pos]
                H[tgt, src] += amp
        E[mz], V[mz] = _eigh(H)

    # Single-quantum transitions: F+ connects block mz → mz+1.
    freqs, amps = [], []
    for mz in sorted(order_by_mz):
        up = round(mz + 1.0, 6)
        if up not in order_by_mz:
            continue
        lo = order_by_mz[mz]; hi = order_by_mz[up]
        ig_all = allidx[lo]
        Fplus = np.zeros((len(hi), len(lo)))
        sorted_up, args_up = blk_sorted[up], blk_argsort[up]
        for g in range(G):
            ig = ig_all[:, g]
            mask = ig > 0
            if not mask.any():
                continue
            src = np.nonzero(mask)[0]
            tgt_key = keys[lo][mask] - weight[g]
            pos = np.searchsorted(sorted_up, tgt_key)
            tgt = args_up[pos]
            Fplus[tgt, src] += aplist[g][ig[mask]]
        M = V[up].T @ Fplus @ V[mz]
        inten = M * M
        df = E[up][:, None] - E[mz][None, :]
        keep = inten > intensity_threshold
        freqs.append(df[keep]); amps.append(inten[keep])

    if freqs:
        return np.concatenate(freqs), np.concatenate(amps)
    return np.array([]), np.array([])


def composite_transitions(shifts, couplings, degeneracy, field_mhz,
                          intensity_threshold=1e-6):
    """Molecule's exact line list as ``(centers_ppm, amps)`` via composite
    reduction + connected-component decomposition.

    Per-component intensities are renormalised to proton count so inter-component
    areas are correct. This is the raw peak list — broaden it (or store it) to
    get a spectrum.
    """
    G = len(shifts)
    all_freqs, all_amps = [], []
    for comp in _components(couplings, G):
        sub_shifts = [shifts[g] for g in comp]
        sub_J = [[couplings[a][b] for b in comp] for a in comp]
        sub_deg = [degeneracy[g] for g in comp]
        cf, ca = system_transitions(sub_shifts, sub_J, sub_deg, field_mhz,
                                    intensity_threshold)
        if not len(cf):
            continue
        comp_protons = sum(int(degeneracy[g]) for g in comp)
        raw = ca.sum()
        if raw > 0:
            ca = ca * (comp_protons / raw)
        all_freqs.append(cf); all_amps.append(ca)

    freqs = np.concatenate(all_freqs) if all_freqs else np.array([])
    amps = np.concatenate(all_amps) if all_amps else np.array([])
    return freqs / field_mhz, amps


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

    Returns (ppm_axis, intensity) normalised to unit integral.
    """
    centers, amps = composite_transitions(shifts, couplings, degeneracy,
                                          field_mhz, intensity_threshold)
    ppm = np.linspace(ppm_from, ppm_to, points)
    spec = peaks_to_spectrum(centers, amps, points=points, ppm_from=ppm_from,
                             ppm_to=ppm_to, linewidth_hz=linewidth_hz,
                             field_mhz=field_mhz, normalize=True)
    return ppm, spec
