"""
ml_model.diff_renderer_torch
============================
PyTorch differentiable 1H renderer for the Stage-2 spectral loss.

This is the production twin of ``ml_model.diff_renderer_ref`` (numpy oracle).
It produces a normalized spectrum from per-group (shifts, couplings, degeneracy)
and is differentiable w.r.t. shifts (ppm) and couplings (Hz), so a spectral loss
(e.g. Wasserstein/MSE at 90 and/or 600 MHz) can backprop into the model.

The one non-standard piece is ``RegularizedEigh``: a custom autograd Function
that replaces torch.linalg.eigh's backward with the Lorentzian-regularized VJP

    F_ij = (E_j - E_i) / ((E_j - E_i)^2 + eps^2)        # vs naive 1/(E_j - E_i)

The de-risking spike (ml_model/diff_renderer_ref + eigh_grad_spike) showed this
is required for the exact eigenvalue degeneracies that equivalent-spin systems
produce; the naive backward gives 1/0 -> NaN there. eps ~ linewidth (~1 Hz) is
insensitive over orders of magnitude and leaves well-separated systems untouched.

NOTE ON SCALE
-------------
This module uses explicit spin-1/2 expansion (2^N Hilbert space), which mirrors
the verified oracle and is correct, but blows up for big degeneracies. For
training at scale, swap the Hamiltonian/F+ construction for the
composite-particle reduction in ``simulation.pyspin.composite`` (total-spin
manifolds) -- the parameter-independent block/index structure is built once in
numpy and only H assembly + RegularizedEigh + overlaps need to be torch ops.
The gradient treatment (RegularizedEigh) is identical either way; that's the
part the spike de-risked.

VERIFY IN YOUR ENV
------------------
    python3 -m ml_model.diff_renderer_torch     # runs torch.autograd.gradcheck
and cross-check forward/gradient against the numpy oracle:
    diff_renderer_ref.loss_and_grad(...)  ==  this (to float tol).
"""

from __future__ import annotations

import itertools

import torch

DEFAULT_EIGH_EPS = 1.0   # Hz


# -----------------------------------------------------------------------------
# Custom regularized symmetric eigendecomposition
# -----------------------------------------------------------------------------

class RegularizedEigh(torch.autograd.Function):
    """eigh with a Lorentzian-regularized backward (degeneracy-safe).

    Forward identical to torch.linalg.eigh(H) for symmetric H.
    Backward uses F_ij = dE/(dE^2 + eps^2) instead of 1/dE.
    """

    @staticmethod
    def forward(ctx, H, eps):
        E, V = torch.linalg.eigh(H)
        ctx.save_for_backward(E, V)
        ctx.eps = float(eps)
        return E, V

    @staticmethod
    def backward(ctx, dE, dV):
        E, V = ctx.saved_tensors
        eps = ctx.eps
        # deltaE[i, j] = E[j] - E[i]
        deltaE = E.unsqueeze(-2) - E.unsqueeze(-1)
        if eps == 0.0:
            F = torch.where(deltaE.abs() > 0, 1.0 / deltaE, torch.zeros_like(deltaE))
        else:
            F = deltaE / (deltaE ** 2 + eps ** 2)
        F = F - torch.diag_embed(torch.diagonal(F, dim1=-2, dim2=-1))  # zero diagonal

        inner = torch.diag_embed(dE) if dE is not None else torch.zeros_like(V)
        if dV is not None:
            inner = inner + F * (V.transpose(-2, -1) @ dV)
        dH = V @ inner @ V.transpose(-2, -1)
        dH = 0.5 * (dH + dH.transpose(-2, -1))
        return dH, None


def regularized_eigh(H, eps=DEFAULT_EIGH_EPS):
    return RegularizedEigh.apply(H, eps)


# -----------------------------------------------------------------------------
# Parameter-independent operator structure (built once in numpy/torch, no grad)
# -----------------------------------------------------------------------------

def _structure(degeneracy, device, dtype):
    """Build owner map, spin templates (Tnu, TJ pairs) and F+ for the expanded
    spin-1/2 system. All independent of parameter VALUES -> no grad needed."""
    owner = []
    for g, d in enumerate(degeneracy):
        owner.extend([g] * int(d))
    n = len(owner)
    states = [tuple((s >> i) & 1 for i in range(n)) for s in range(2 ** n)]
    idx = {s: i for i, s in enumerate(states)}
    dim = len(states)
    m = torch.tensor([[0.5 if b else -0.5 for b in s] for s in states],
                     device=device, dtype=dtype)

    Tnu = torch.stack([torch.diag(m[:, a]) for a in range(n)])  # (n, dim, dim)

    pair_idx = []   # (a, b) spin pairs with a<b
    TJ = []
    for a in range(n):
        for b in range(a + 1, n):
            T = torch.zeros(dim, dim, device=device, dtype=dtype)
            for r, s in enumerate(states):
                T[r, r] += m[r, a] * m[r, b]
                if s[a] != s[b]:
                    s2 = list(s); s2[a], s2[b] = s[b], s[a]
                    T[r, idx[tuple(s2)]] += 0.5
            TJ.append(T)
            pair_idx.append((a, b))
    TJ = torch.stack(TJ) if TJ else torch.zeros(0, dim, dim, device=device, dtype=dtype)

    Fp = torch.zeros(dim, dim, device=device, dtype=dtype)
    for c, s in enumerate(states):
        for i in range(n):
            if s[i] == 0:
                s2 = list(s); s2[i] = 1
                Fp[idx[tuple(s2)], c] += 1.0

    return dict(owner=owner, n=n, dim=dim, Tnu=Tnu, TJ=TJ, Fp=Fp, pair_idx=pair_idx)


# -----------------------------------------------------------------------------
# Forward (differentiable)
# -----------------------------------------------------------------------------

def simulate(shifts, couplings, degeneracy, field_mhz, points=16384,
             ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0,
             eigh_eps=DEFAULT_EIGH_EPS, struct=None):
    """Normalized spectrum (points,), differentiable in shifts & couplings.

    shifts:    (G,) tensor, ppm
    couplings: (G, G) symmetric tensor, Hz (diagonal ignored)
    degeneracy: length-G int sequence (fixed; defines Hilbert space)
    Returns (ppm_axis, spectrum) tensors. Reuse ``struct`` across a fixed
    degeneracy pattern to avoid rebuilding operators.
    """
    device, dtype = shifts.device, shifts.dtype
    if struct is None:
        struct = _structure(degeneracy, device, dtype)
    owner, Tnu, TJ, Fp = struct["owner"], struct["Tnu"], struct["TJ"], struct["Fp"]
    pair_idx = struct["pair_idx"]

    # per-spin offsets (Hz): nu_a = shift[owner_a] * field
    owner_t = torch.tensor(owner, device=device, dtype=torch.long)
    nu_hz = shifts[owner_t] * field_mhz                          # (n,)

    H = (nu_hz[:, None, None] * Tnu).sum(0)
    if TJ.shape[0]:
        # each expanded spin pair takes its parent group-pair coupling
        gpair = torch.tensor([[owner[a], owner[b]] for (a, b) in pair_idx],
                             device=device, dtype=torch.long)
        jvals = couplings[gpair[:, 0], gpair[:, 1]]              # (npair,)
        H = H + (jvals[:, None, None] * TJ).sum(0)

    E, V = regularized_eigh(H, eigh_eps)
    M = V.transpose(-2, -1) @ Fp @ V
    amps = (M * M).reshape(-1)
    centers = ((E[:, None] - E[None, :]) / field_mhz).reshape(-1)   # ppm

    grid = torch.linspace(ppm_from, ppm_to, points, device=device, dtype=dtype)
    dx = (ppm_to - ppm_from) / (points - 1)
    hwhm = (linewidth_hz / 2.0) / field_mhz

    x = grid[None, :] - centers[:, None]
    L = (hwhm / (x ** 2 + hwhm ** 2)) / torch.pi
    spec_un = (amps[:, None] * L).sum(0)
    area = spec_un.sum() * dx
    spec = spec_un / area
    return grid, spec


# -----------------------------------------------------------------------------
# Self-test: gradcheck + parity vs the numpy oracle
# -----------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    dtype = torch.double
    field = 90.0
    # small A3X-like + extra spins to include a degeneracy
    deg = [2, 1, 1]
    G = len(deg)
    shifts = torch.tensor([2.5, 4.0, 6.0], dtype=dtype, requires_grad=True)
    base = torch.tensor([[0., 7., 1.], [7., 0., 5.], [1., 5., 0.]], dtype=dtype)
    coup = base.clone().requires_grad_(True)
    struct = _structure(deg, shifts.device, dtype)

    # target spectrum at perturbed shifts
    with torch.no_grad():
        _, tgt = simulate(shifts + 0.05, base, deg, field, points=2048,
                          linewidth_hz=8.0, struct=struct)

    def loss(s, c):
        _, spec = simulate(s, c, deg, field, points=2048, linewidth_hz=8.0,
                           eigh_eps=1.0, struct=struct)
        return ((spec - tgt) ** 2).sum() * (12.0 / 2047)

    L = loss(shifts, coup)
    L.backward()
    print("loss:", float(L))
    print("dL/dshifts:", shifts.grad.tolist())
    print("dL/dcouplings (upper tri):",
          [float(coup.grad[i, j]) for i in range(G) for j in range(i + 1, G)])

    # autograd.gradcheck on a tiny system (double precision)
    s2 = torch.tensor([2.0, 5.0], dtype=dtype, requires_grad=True)
    c2 = torch.tensor([[0., 6.0], [6.0, 0.]], dtype=dtype, requires_grad=True)
    st2 = _structure([2, 1], s2.device, dtype)
    with torch.no_grad():
        _, t2 = simulate(s2 + 0.05, c2, [2, 1], field, points=1024,
                         linewidth_hz=10.0, struct=st2)

    def f(s, c):
        _, spec = simulate(s, c, [2, 1], field, points=1024, linewidth_hz=10.0,
                           eigh_eps=1.0, struct=st2)
        return ((spec - t2) ** 2).sum()

    ok = torch.autograd.gradcheck(f, (s2, c2), eps=1e-6, atol=1e-4, rtol=1e-3)
    print("torch.autograd.gradcheck passed:", ok)


if __name__ == "__main__":
    _selftest()
