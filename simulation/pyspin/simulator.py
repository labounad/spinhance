"""
pyspin.simulator
================
PROTOTYPE exact spin-1/2 NMR simulator (feasibility test for replacing MNova).

Physics
-------
Hamiltonian (Hz units), for N coupled spin-1/2:

    H = Σ_i ν_i Iz_i  +  Σ_{i<j} J_ij ( Iz_i Iz_j + ½(I+_i I-_j + I-_i I+_j) )

where ν_i = δ_i(ppm) · field(MHz) is the offset in Hz.

Key efficiency trick: total Iz is conserved, so H is block-diagonal in the
number of "up" spins k. We diagonalise blocks of size C(N,k) instead of the
full 2^N matrix. Single-quantum transitions connect adjacent k blocks; the
detection operator is F+ = Σ_i I+_i, intensity = |⟨p|F+|q⟩|² (equal high-T
populations assumed).

Equivalent spins within a group (e.g. CH3) are expanded to individual spin-1/2
with zero intra-group coupling. This is correct but costs Hilbert space — a
composite-particle reduction would be the next optimisation for large groups
(tert-butyl etc.).

This is a prototype: correctness-first, with the one essential speed trick. Not
yet optimised (no sparse blocks, no symmetry factoring, no parallelism).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = ["simulate_spectrum", "expand_groups"]


def expand_groups(shifts, couplings, degeneracy):
    """Expand G spin groups into N individual spins (one per proton).

    Returns (nu_ppm, Jspin) where nu_ppm[a] is spin a's shift in ppm and
    Jspin[a,b] is the coupling in Hz (0 within a group — equivalent spins).
    """
    groups = []
    for g, d in enumerate(degeneracy):
        groups.extend([g] * int(d))
    n = len(groups)
    nu_ppm = np.array([shifts[g] for g in groups], dtype=float)
    Jspin = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            ga, gb = groups[a], groups[b]
            if ga != gb:
                Jspin[a, b] = Jspin[b, a] = couplings[ga][gb]
    return nu_ppm, Jspin


def _block_states(n, k):
    """All bitmask integers with exactly k set bits among n bits."""
    masks = []
    for bits in combinations(range(n), k):
        m = 0
        for b in bits:
            m |= (1 << b)
        masks.append(m)
    return masks


def simulate_spectrum(
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
    """Simulate a 1H spectrum; return (ppm_axis, intensity) with unit integral.

    Parameters mirror xml_io.matrix_to_xml. Uses the Iz-block algorithm above.
    """
    nu_ppm, Jspin = expand_groups(shifts, couplings, degeneracy)
    n = len(nu_ppm)
    nu_hz = nu_ppm * field_mhz  # offset of each spin in Hz

    # m_i = +1/2 if bit set else -1/2
    def mvals(mask):
        return np.array([0.5 if (mask >> i) & 1 else -0.5 for i in range(n)])

    # Diagonalise each Iz block; keep eigenvalues/vectors and basis index maps.
    block_E = {}
    block_V = {}
    block_idx = {}   # k -> {mask: row index}
    block_masks = {}
    for k in range(n + 1):
        masks = _block_states(n, k)
        block_masks[k] = masks
        idx = {m: i for i, m in enumerate(masks)}
        block_idx[k] = idx
        dim = len(masks)
        H = np.zeros((dim, dim))
        for a, mask in enumerate(masks):
            m = mvals(mask)
            # diagonal: Zeeman + Iz Iz
            diag = float(np.dot(nu_hz, m))
            for i in range(n):
                for j in range(i + 1, n):
                    diag += Jspin[i, j] * m[i] * m[j]
            H[a, a] = diag
            # off-diagonal flip-flop within same k
            for i in range(n):
                for j in range(i + 1, n):
                    bi = (mask >> i) & 1
                    bj = (mask >> j) & 1
                    if bi != bj:
                        mask2 = mask ^ ((1 << i) | (1 << j))
                        b = idx[mask2]
                        if b > a:
                            H[a, b] = H[b, a] = 0.5 * Jspin[i, j]
        E, V = np.linalg.eigh(H)
        block_E[k] = E
        block_V[k] = V

    # Single-quantum transitions: F+ connects block (k-1) -> k.
    freqs = []
    amps = []
    for k in range(1, n + 1):
        masks_lo = block_masks[k - 1]
        idx_hi = block_idx[k]
        dim_lo = len(masks_lo)
        dim_hi = len(block_masks[k])
        Fplus = np.zeros((dim_hi, dim_lo))
        for q, mask in enumerate(masks_lo):
            for i in range(n):
                if not ((mask >> i) & 1):
                    mask_p = mask | (1 << i)
                    Fplus[idx_hi[mask_p], q] += 1.0
        # transform to eigenbasis: M[p,q] = <p_k| F+ |q_{k-1}>
        M = block_V[k].T @ Fplus @ block_V[k - 1]
        inten = M * M
        Ehi = block_E[k][:, None]
        Elo = block_E[k - 1][None, :]
        df = Ehi - Elo  # transition frequency in Hz
        mask_keep = inten > intensity_threshold
        freqs.append(df[mask_keep])
        amps.append(inten[mask_keep])

    freqs = np.concatenate(freqs) if freqs else np.array([])
    amps = np.concatenate(amps) if amps else np.array([])

    # Build spectrum: Lorentzian broadening onto the ppm grid.
    ppm = np.linspace(ppm_from, ppm_to, points)
    centers_ppm = freqs / field_mhz
    hwhm_ppm = (linewidth_hz / 2.0) / field_mhz
    spec = lorentzian_broaden(centers_ppm, amps, points, ppm_from, ppm_to, hwhm_ppm)

    # normalise to unit integral over ppm
    dppm = (ppm_to - ppm_from) / points
    total = spec.sum() * dppm
    if total > 0:
        spec = spec / total
    return ppm, spec


def lorentzian_broaden(centers, amps, points, ppm_from, ppm_to, hwhm):
    """Bin transitions to a stick spectrum and FFT-convolve with a Lorentzian.

    O(n_transitions) binning + O(points log points) convolution — independent
    of the (often huge) number of transitions. Shared by both pyspin engines.
    """
    spec = np.zeros(points)
    if len(centers) == 0:
        return spec
    dppm = (ppm_to - ppm_from) / points
    # Linear-interpolation binning: split each line between its two nearest bins.
    pos = (np.asarray(centers) - ppm_from) / dppm
    inrange = (pos >= 0) & (pos <= points - 1)
    pos = pos[inrange]
    w = np.asarray(amps)[inrange]
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    stick = (np.bincount(i0, weights=w * (1 - frac), minlength=points)
             + np.bincount(i0 + 1, weights=w * frac, minlength=points + 1)[:points])
    # Lorentzian kernel centred on the grid; circular FFT convolution.
    x = (np.arange(points) - points // 2) * dppm
    kern = 1.0 / (1.0 + (x / hwhm) ** 2)
    return np.fft.irfft(np.fft.rfft(stick) * np.fft.rfft(np.fft.ifftshift(kern)),
                        n=points)
