"""
model.diff_renderer_ref
==========================
NumPy REFERENCE differentiable 1H renderer + analytic gradients.

Purpose
-------
This is the verified *oracle* for the Stage-2 spectral loss. It mirrors the
physics of ``simulation.pyspin.simulator`` (explicit spin-1/2 expansion) but
adds an analytic reverse-mode gradient of a spectral loss w.r.t. the shift and
coupling parameters, using the SAME eigendecomposition VJP that
``torch.linalg.eigh`` implements -- with the Lorentzian regularization
(``eigh_eps``) that the de-risking spike showed is needed for the exact
degeneracies produced by equivalent-spin expansion.

It exists for two jobs:
  1. A correctness oracle: ``model.diff_renderer_torch`` must match this
     (forward AND gradient) to within float tolerance.
  2. A standalone, torch-free way to gradient-check the whole pipeline against
     finite differences (see tests/test_diff_renderer.py).

Explicit expansion blows up as 2^(total spins), so this oracle is meant for
SMALL systems (gradient checks, unit tests). The production engine
(diff_renderer_torch, built on composite reduction) is what trains at scale.

API mirrors pyspin: simulate(shifts, couplings, degeneracy, field_mhz, ...).
"""

from __future__ import annotations

import numpy as np

__all__ = ["expand_groups", "simulate", "loss_and_grad", "DEFAULT_EIGH_EPS"]

# Regularization for the eigh backward, in Hz. The spike showed any value from
# 1e-3 to a few Hz recovers FD-accurate gradients on degenerate systems while
# barely touching well-separated ones; tie it to the linewidth scale.
DEFAULT_EIGH_EPS = 1.0


# -----------------------------------------------------------------------------
# Group -> spin expansion and parameter-independent operator templates
# -----------------------------------------------------------------------------

def expand_groups(shifts, couplings, degeneracy):
    """Expand G groups into N individual spin-1/2 (one per proton).

    Returns (nu_ppm[N], Jspin[N,N], owner[N]) where owner[a] is the group index
    of spin a (needed to fold per-spin gradients back to per-group params).
    Intra-group coupling is 0 (equivalent spins).
    """
    owner = []
    for g, d in enumerate(degeneracy):
        owner.extend([g] * int(d))
    n = len(owner)
    nu_ppm = np.array([shifts[g] for g in owner], dtype=float)
    Jspin = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            ga, gb = owner[a], owner[b]
            if ga != gb:
                Jspin[a, b] = Jspin[b, a] = couplings[ga][gb]
    return nu_ppm, Jspin, np.array(owner)


def _basis(n):
    return [tuple((s >> i) & 1 for i in range(n)) for s in range(2 ** n)]


def _spin_templates(n):
    """Constant matrices so H = sum_a nu_a Tnu[a] + sum_{a<b} Jspin_ab TJ[a,b]."""
    states = _basis(n)
    dim = len(states)
    idx = {s: i for i, s in enumerate(states)}
    m = np.array([[0.5 if b else -0.5 for b in s] for s in states])
    Tnu = [np.diag(m[:, a]).astype(float) for a in range(n)]
    TJ = {}
    for a in range(n):
        for b in range(a + 1, n):
            T = np.zeros((dim, dim))
            for r, s in enumerate(states):
                T[r, r] += m[r, a] * m[r, b]            # Iz Iz
                if s[a] != s[b]:                        # flip-flop
                    s2 = list(s); s2[a], s2[b] = s[b], s[a]
                    T[r, idx[tuple(s2)]] += 0.5
            TJ[(a, b)] = T
    return states, m, Tnu, TJ


def _fplus(states, n):
    dim = len(states)
    idx = {s: i for i, s in enumerate(states)}
    A = np.zeros((dim, dim))
    for c, s in enumerate(states):
        for i in range(n):
            if s[i] == 0:
                s2 = list(s); s2[i] = 1
                A[idx[tuple(s2)], c] += 1.0
    return A


# -----------------------------------------------------------------------------
# Forward
# -----------------------------------------------------------------------------

def _grid(points, ppm_from, ppm_to):
    g = np.linspace(ppm_from, ppm_to, points)
    return g, (g[1] - g[0])


def _forward_core(nu_hz, Jspin, field_mhz, Tnu, TJ, A, grid, dx, hwhm_ppm):
    """Return (spec, cache) where cache holds intermediates for the backward."""
    n = len(nu_hz)
    H = np.zeros_like(Tnu[0])
    for a in range(n):
        H = H + nu_hz[a] * Tnu[a]
    for (a, b), T in TJ.items():
        H = H + Jspin[a, b] * T
    E, V = np.linalg.eigh(H)
    dim = len(E)

    M = V.T @ A @ V
    amps_full = (M * M).ravel()
    df = (E[:, None] - E[None, :])
    centers_full = (df / field_mhz).ravel()

    keep = (centers_full >= grid[0]) & (centers_full <= grid[-1]) & (amps_full > 0)
    centers = centers_full[keep]
    amps = amps_full[keep]

    xk = grid[None, :] - centers[:, None]
    Lk = (hwhm_ppm / (xk ** 2 + hwhm_ppm ** 2)) / np.pi      # per-unit-amp lineshape
    spec_un = (amps[:, None] * Lk).sum(axis=0)
    area = spec_un.sum() * dx
    spec = spec_un / area if area > 0 else spec_un

    cache = dict(E=E, V=V, M=M, A=A, dim=dim, keep=keep, centers=centers,
                 amps=amps, xk=xk, Lk=Lk, spec_un=spec_un, area=area,
                 field=field_mhz, hwhm_ppm=hwhm_ppm, dx=dx, Tnu=Tnu, TJ=TJ)
    return spec, cache


def simulate(shifts, couplings, degeneracy, field_mhz, points=16384,
             ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0):
    """Normalized 1H spectrum on a (points,) ppm grid (unit integral)."""
    nu_ppm, Jspin, _ = expand_groups(shifts, couplings, degeneracy)
    n = len(nu_ppm)
    states, m, Tnu, TJ = _spin_templates(n)
    A = _fplus(states, n)
    grid, dx = _grid(points, ppm_from, ppm_to)
    hwhm_ppm = (linewidth_hz / 2.0) / field_mhz
    spec, _ = _forward_core(nu_ppm * field_mhz, Jspin, field_mhz,
                            Tnu, TJ, A, grid, dx, hwhm_ppm)
    return grid, spec


# -----------------------------------------------------------------------------
# Backward: gradient of a spectral MSE loss w.r.t. per-GROUP shifts & couplings
# -----------------------------------------------------------------------------

def loss_and_grad(shifts, couplings, degeneracy, field_mhz, target,
                  points=16384, ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0,
                  eigh_eps=DEFAULT_EIGH_EPS):
    """L = sum((spec - target)^2) dx, plus dL/dshifts (ppm) and dL/dcouplings (Hz).

    ``eigh_eps`` regularizes the eigh VJP: F_ij = dE/(dE^2 + eps^2). Pass 0 for
    the naive (unstable on degeneracies) backward, matching torch's default.
    """
    nu_ppm, Jspin, owner = expand_groups(shifts, couplings, degeneracy)
    n = len(nu_ppm)
    G = len(shifts)
    states, m, Tnu, TJ = _spin_templates(n)
    A = _fplus(states, n)
    grid, dx = _grid(points, ppm_from, ppm_to)
    hwhm_ppm = (linewidth_hz / 2.0) / field_mhz

    spec, c = _forward_core(nu_ppm * field_mhz, Jspin, field_mhz,
                            Tnu, TJ, A, grid, dx, hwhm_ppm)
    loss = float(np.sum((spec - target) ** 2) * dx)

    E, V, dim, keep = c["E"], c["V"], c["dim"], c["keep"]
    xk, Lk, spec_un, area = c["xk"], c["Lk"], c["spec_un"], c["area"]
    amps_keep = c["amps"]

    # dL/dspec -> dL/dspec_un (through normalization)
    dL_dspec = 2.0 * (spec - target) * dx
    dL_dspec_un = dL_dspec / area - (np.dot(dL_dspec, spec_un) / area ** 2) * dx

    # spec_un = sum_k amp_k Lk  -> grads to amps and centers (kept transitions)
    dL_damp_keep = (Lk * dL_dspec_un[None, :]).sum(axis=1)
    dLk_dc = (hwhm_ppm * 2.0 * xk / (xk ** 2 + hwhm_ppm ** 2) ** 2) / np.pi
    dL_dcenter_keep = (amps_keep[:, None] * dLk_dc * dL_dspec_un[None, :]).sum(axis=1)

    # scatter to full (dim,dim) transition grids
    dL_damp = np.zeros(dim * dim); dL_damp[keep] = dL_damp_keep
    dL_dcenter = np.zeros(dim * dim); dL_dcenter[keep] = dL_dcenter_keep
    dL_damp = dL_damp.reshape(dim, dim)
    dL_dcenter = dL_dcenter.reshape(dim, dim)

    # centers = (E_p - E_q)/field ; amps = M^2 ; M = V^T A V
    dL_dE = dL_dcenter.sum(axis=1) / field_mhz - dL_dcenter.sum(axis=0) / field_mhz
    dL_dM = 2.0 * c["M"] * dL_damp
    dL_dV = A @ V @ dL_dM.T + A.T @ V @ dL_dM

    # regularized eigh VJP
    VtdV = V.T @ dL_dV
    deltaE = E[None, :] - E[:, None]
    if eigh_eps == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(np.abs(deltaE) > 0, 1.0 / deltaE, 0.0)
    else:
        F = deltaE / (deltaE ** 2 + eigh_eps ** 2)
    np.fill_diagonal(F, 0.0)
    dL_dH = V @ (np.diag(dL_dE) + F * VtdV) @ V.T
    dL_dH = 0.5 * (dL_dH + dL_dH.T)

    # dL/dH -> per-spin nu (Hz) and per-spin-pair Jspin (Hz)
    dL_dnu_hz = np.array([np.sum(dL_dH * Tnu[a]) for a in range(n)])
    dL_dJspin = {(a, b): np.sum(dL_dH * TJ[(a, b)]) for (a, b) in TJ}

    # fold spins back to GROUPS
    # nu_a = shift_group(a) * field  -> dL/dshift_g = field * sum_{a in g} dL/dnu_a
    dL_dshift = np.zeros(G)
    for a in range(n):
        dL_dshift[owner[a]] += field_mhz * dL_dnu_hz[a]
    # Jspin_ab = couplings[owner a][owner b] (a,b in different groups), each group
    # pair (g,h) is repeated deg_g*deg_h times across spin pairs -> grads add.
    dL_dcoupling = np.zeros((G, G))
    for (a, b), val in dL_dJspin.items():
        g, h = owner[a], owner[b]
        if g != h:
            dL_dcoupling[g, h] += val
            dL_dcoupling[h, g] += val
    return loss, dL_dshift, dL_dcoupling
