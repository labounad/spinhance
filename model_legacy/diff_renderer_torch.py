"""
model.diff_renderer_torch
=========================
PyTorch differentiable 1H renderer for the Stage-2 spectral loss — composite
(manifold-reduced) engine. Production twin of ``model.composite_diff`` (numpy
oracle), which it must match (forward + gradient) to float tolerance.

Design (see model/composite_diff.py and DESIGN.md §5/§6):
  * Total-spin MANIFOLD REDUCTION + Mz BLOCK-DIAGONALISATION: each group of d
    equivalent spins reduces to its total-spin manifolds, and we diagonalise the
    small Mz blocks — never a dense 2^N Hamiltonian, no per-parameter templates.
  * NO connected-component split (couplings are continuous at train time).
  * ``RegularizedEigh``: custom autograd Function whose backward uses the
    Lorentzian-regularised VJP  F_ij = ΔE/(ΔE²+ε²)  (de-risked spike) so
    backprop is stable on the exact degeneracies equivalent spins produce.
  * Memory-flat broadening: bin sticks to the grid (linear interp) + FFT-convolve
    with a Lorentzian kernel — O(points·log points), independent of #transitions.

Only the block Hamiltonians depend on (shifts, couplings); the F⁺ operators and
all index structure are parameter-independent and built once per degeneracy
pattern by ``_structure`` (cached). Autograd handles the gradient; the sole
custom backward is RegularizedEigh.

VERIFY IN YOUR ENV (no torch in the prototyping sandbox):
    python3 -m model_legacy.diff_renderer_torch
runs (a) parity vs the numpy oracle and (b) torch.autograd.gradcheck.
"""

from __future__ import annotations

import numpy as np
import torch

from model_legacy.composite_diff import build_static_plan

DEFAULT_EIGH_EPS = 1.0   # Hz


# -----------------------------------------------------------------------------
# Regularized symmetric eigendecomposition (degeneracy-safe backward)
# -----------------------------------------------------------------------------

class RegularizedEigh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eps):
        in_dtype = H.dtype
        # linalg.eigh has no bf16/fp16 CUDA kernel — upcast to fp32 only for
        # half-precision inputs.  fp32/fp64 inputs are kept as-is so that
        # downstream matmuls with tensors of the same dtype remain type-consistent.
        _HALF = (torch.bfloat16, torch.float16)
        with torch.autocast(device_type="cuda", enabled=False):
            Hf = H.float() if in_dtype in _HALF else H
            E, V = torch.linalg.eigh(Hf)
        ctx.save_for_backward(E, V)
        ctx.eps = float(eps)
        ctx.in_dtype = in_dtype
        return E, V

    @staticmethod
    def backward(ctx, dE, dV):
        E, V = ctx.saved_tensors
        eps = ctx.eps
        deltaE = E.unsqueeze(-2) - E.unsqueeze(-1)          # E_j - E_i
        if eps == 0.0:
            F = torch.where(deltaE.abs() > 0, 1.0 / deltaE, torch.zeros_like(deltaE))
        else:
            F = deltaE / (deltaE ** 2 + eps ** 2)
        F = F - torch.diag_embed(torch.diagonal(F, dim1=-2, dim2=-1))
        inner = torch.diag_embed(dE) if dE is not None else torch.zeros_like(V)
        if dV is not None:
            inner = inner + F * (V.transpose(-2, -1) @ dV)
        dH = V @ inner @ V.transpose(-2, -1)
        return 0.5 * (dH + dH.transpose(-2, -1)).to(ctx.in_dtype), None


def regularized_eigh(H, eps=DEFAULT_EIGH_EPS):
    return RegularizedEigh.apply(H, eps)


# -----------------------------------------------------------------------------
# Parameter-independent plan (torch-free; built in composite_diff, cached here)
# -----------------------------------------------------------------------------

_PLAN_CACHE = {}


def _structure(degeneracy, device=None, dtype=torch.float64):
    """Manifold-combination plan for a degeneracy pattern (parameter-independent,
    torch-free; shared with the numpy oracle). Cached per pattern."""
    key = tuple(int(x) for x in degeneracy)
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = build_static_plan(key)
    return _PLAN_CACHE[key]


# -----------------------------------------------------------------------------
# Differentiable forward
# -----------------------------------------------------------------------------

def _to(t, arr, device, dtype):
    return torch.as_tensor(arr, device=device, dtype=dtype)


def _assemble_H(blk, nu_active, Jpairs, device, dtype):
    """Block Hamiltonian as a differentiable torch tensor. Single-sample."""
    n = blk["n"]
    Mb = _to(None, blk["Mb"], device, dtype)
    ising = _to(None, blk["ising"], device, dtype)
    diag = Mb @ nu_active
    if ising.shape[1]:
        diag = diag + ising @ Jpairs
    H = torch.diag_embed(diag)
    if len(blk["rows"]):
        rows = torch.as_tensor(blk["rows"], device=device)
        cols = torch.as_tensor(blk["cols"], device=device)
        pidx = torch.as_tensor(blk["pidx"], device=device)
        amps = _to(None, blk["amps"], device, dtype)
        vals = Jpairs[pidx] * amps
        H = H.index_put((rows, cols), vals, accumulate=True)
    return 0.5 * (H + H.transpose(-2, -1))


def _assemble_H_batch(blk, nu_active, Jpairs, device, dtype):
    """Batched block Hamiltonian: nu_active (B, A), Jpairs (B, P) -> (B, n, n)."""
    n = blk["n"]
    Mb    = _to(None, blk["Mb"],    device, dtype)   # (n, A)
    ising = _to(None, blk["ising"], device, dtype)   # (n, P)
    diag  = nu_active @ Mb.T                         # (B, n)
    if ising.shape[1] and Jpairs.shape[1] > 0:
        diag = diag + Jpairs @ ising.T               # (B, n)
    H = torch.diag_embed(diag)                       # (B, n, n)
    if len(blk["rows"]):
        rows = torch.as_tensor(blk["rows"], device=device)   # (Q,)
        cols = torch.as_tensor(blk["cols"], device=device)   # (Q,)
        pidx = torch.as_tensor(blk["pidx"], device=device)   # (Q,)
        amps = _to(None, blk["amps"], device, dtype)          # (Q,)
        vals = Jpairs[:, pidx] * amps                         # (B, Q)
        # Scatter into flattened (B, n*n) then reshape — differentiable via index_add
        flat  = H.reshape(H.shape[0], n * n)
        fidx  = (rows * n + cols).unsqueeze(0).expand(vals.shape[0], -1)  # (B, Q)
        flat  = flat.scatter_add(1, fidx, vals)
        H     = flat.reshape(H.shape[0], n, n)
    return 0.5 * (H + H.transpose(-2, -1))


def simulate(shifts, couplings, degeneracy, field_mhz, points=16384,
             ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0,
             eigh_eps=DEFAULT_EIGH_EPS, struct=None):
    device, dtype = shifts.device, shifts.dtype
    if struct is None:
        struct = _structure(degeneracy, device, dtype)
    nu = shifts * field_mhz                                  # (G,) Hz

    freqs, amps = [], []
    for (S_list, weight, sb) in struct["combos"]:
        active = sb["active"]
        nu_active = nu[torch.as_tensor(active, device=device)]
        if sb["pairs"]:
            gi = torch.as_tensor([g for (g, h) in sb["pair_groups"]], device=device)
            hi = torch.as_tensor([h for (g, h) in sb["pair_groups"]], device=device)
            Jpairs = couplings[gi, hi]
        else:
            Jpairs = torch.zeros(0, device=device, dtype=dtype)
        E, V = {}, {}
        for mz, blk in sb["blocks"].items():
            H = _assemble_H(blk, nu_active, Jpairs, device, dtype)
            e, v = regularized_eigh(H, eigh_eps)
            E[mz] = e; V[mz] = v
        for mz, (up, Fp) in sb["fplus"].items():
            Fpt = _to(None, Fp, device, dtype)
            M = V[up].transpose(-2, -1) @ Fpt @ V[mz]
            amps.append((M * M).reshape(-1) * weight)
            freqs.append((E[up][:, None] - E[mz][None, :]).reshape(-1))
    freqs = torch.cat(freqs) if freqs else torch.zeros(0, device=device, dtype=dtype)
    amps = torch.cat(amps) if amps else torch.zeros(0, device=device, dtype=dtype)

    centers = freqs / field_mhz
    hwhm = (linewidth_hz / 2.0) / field_mhz
    grid = torch.linspace(ppm_from, ppm_to, points, device=device, dtype=dtype)
    dx = (ppm_to - ppm_from) / points
    spec = _broaden_fft(centers, amps, points, ppm_from, ppm_to, dx, hwhm,
                        device, dtype)
    area = spec.sum() * dx
    return grid, spec / area


def _broaden_fft_batch(centers, amps, points, ppm_from, ppm_to, dx, hwhm, device, dtype):
    """Batched FFT broadening: centers (B, K), amps (B, K) -> (B, points).
    Fully vectorized — no Python loop over batch items."""
    B = centers.shape[0]
    pos   = (centers - ppm_from) / dx                             # (B, K)
    valid = (pos >= 0) & (pos <= points - 1) & (amps.detach() > 0)  # (B, K) bool mask

    # Flat indices into (B * (points+1)) for scatter_add
    b_idx  = torch.arange(B, device=device).unsqueeze(1).expand_as(pos)  # (B, K)
    pos_v  = pos[valid];   amp_v = amps[valid];   b_v = b_idx[valid]
    i0     = torch.floor(pos_v).long()
    frac   = pos_v - i0.to(dtype)
    gi0    = b_v * (points + 1) + i0
    gi1    = gi0 + 1

    stick_flat = torch.zeros(B * (points + 1), device=device, dtype=dtype)
    stick_flat = stick_flat.index_add(0, gi0, amp_v * (1 - frac))
    stick_flat = stick_flat.index_add(0, gi1, amp_v * frac)
    stick = stick_flat.reshape(B, points + 1)[:, :points]          # (B, points)

    x        = (torch.arange(points, device=device, dtype=dtype) - points // 2) * dx
    kern     = torch.fft.ifftshift(1.0 / (1.0 + (x / hwhm) ** 2))
    kern_fft = torch.fft.rfft(kern)                                # shared kernel
    return torch.fft.irfft(torch.fft.rfft(stick) * kern_fft, n=points)  # (B, points)


def simulate_batch(shifts, couplings, degeneracy, field_mhz, points=16384,
                   ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0,
                   eigh_eps=DEFAULT_EIGH_EPS, struct=None):
    """Batched simulate: shifts (B, G), couplings (B, G, G) -> (B, points).
    All samples must share the same degeneracy pattern (single-bucket assumption).
    Replaces B serial calls to simulate() with one batched eigh + FFT."""
    device, dtype = shifts.device, shifts.dtype
    B = shifts.shape[0]
    if struct is None:
        deg_list = [int(x) for x in degeneracy[0].tolist()]
        struct = _structure(deg_list, device, dtype)

    nu = shifts * field_mhz                                        # (B, G) Hz
    all_freqs, all_amps = [], []

    for (S_list, weight, sb) in struct["combos"]:
        active = torch.as_tensor(sb["active"], device=device)
        nu_active = nu[:, active]                                  # (B, A)
        if sb["pairs"]:
            gi = torch.as_tensor([g for (g, h) in sb["pair_groups"]], device=device)
            hi = torch.as_tensor([h for (g, h) in sb["pair_groups"]], device=device)
            Jpairs = couplings[:, gi, hi]                          # (B, P)
        else:
            Jpairs = torch.zeros(B, 0, device=device, dtype=dtype)

        E, V = {}, {}
        for mz, blk in sb["blocks"].items():
            H = _assemble_H_batch(blk, nu_active, Jpairs, device, dtype)  # (B, n, n)
            e, v = regularized_eigh(H, eigh_eps)                           # (B, n), (B, n, n)
            E[mz] = e;  V[mz] = v

        for mz, (up, Fp) in sb["fplus"].items():
            Fpt  = _to(None, Fp, device, dtype)                    # (n_up, n_lo)
            M    = V[up].transpose(-2, -1) @ Fpt @ V[mz]          # (B, n_up, n_lo)
            all_amps.append((M * M).reshape(B, -1) * weight)       # (B, T)
            all_freqs.append(
                (E[up][:, :, None] - E[mz][:, None, :]).reshape(B, -1))  # (B, T)

    if all_freqs:
        freqs = torch.cat(all_freqs, dim=1)                        # (B, K)
        amps  = torch.cat(all_amps,  dim=1)                        # (B, K)
    else:
        freqs = torch.zeros(B, 0, device=device, dtype=dtype)
        amps  = torch.zeros(B, 0, device=device, dtype=dtype)

    dx   = (ppm_to - ppm_from) / points
    hwhm = (linewidth_hz / 2.0) / field_mhz
    specs = _broaden_fft_batch(freqs / field_mhz, amps, points,
                               ppm_from, ppm_to, dx, hwhm, device, dtype)  # (B, points)
    area  = specs.sum(dim=-1, keepdim=True) * dx + 1e-12
    return specs / area                                            # (B, points)


def _broaden_fft(centers, amps, points, ppm_from, ppm_to, dx, hwhm, device, dtype):
    """Bin sticks (linear interp) then FFT-convolve with a Lorentzian kernel.
    Differentiable in centers and amps; O(points log points)."""
    pos = (centers - ppm_from) / dx
    valid = (pos >= 0) & (pos <= points - 1) & (amps > 0)
    pos = pos[valid]; w = amps[valid]
    if pos.numel() == 0:
        return torch.zeros(points, device=device, dtype=dtype)
    i0 = torch.floor(pos).long()
    frac = pos - i0.to(dtype)
    stick = torch.zeros(points + 1, device=device, dtype=dtype)
    stick = stick.index_add(0, i0, w * (1 - frac))
    stick = stick.index_add(0, i0 + 1, w * frac)
    stick = stick[:points]
    x = (torch.arange(points, device=device, dtype=dtype) - points // 2) * dx
    kern = 1.0 / (1.0 + (x / hwhm) ** 2)
    kern = torch.fft.ifftshift(kern)
    out = torch.fft.irfft(torch.fft.rfft(stick) * torch.fft.rfft(kern), n=points)
    return out


# -----------------------------------------------------------------------------
# Self-test: parity vs numpy oracle + gradcheck (run in your env)
# -----------------------------------------------------------------------------

def _selftest():
    from model_legacy import composite_diff as C
    F = 90.0
    cases = {
        "A3X[3,1]": ([3.0, 6.5], [[0.0, 6.8], [6.8, 0.0]], [3, 1]),
        "[2,1,1]": ([2.5, 4.0, 6.0], [[0, 7, 1], [7, 0, 5], [1, 5, 0]], [2, 1, 1]),
        "[9,3,1,1]": ([1.0, 1.3, 3.5, 7.4],
                      [[0, 0, 0, 0], [0, 0, 7.0, 0], [0, 7.0, 0, 1.5], [0, 0, 1.5, 0]],
                      [9, 3, 1, 1]),
    }
    print("parity vs numpy oracle (forward):")
    for name, (sh, cp, dg) in cases.items():
        s = torch.tensor(sh, dtype=torch.float64)
        c = torch.tensor(cp, dtype=torch.float64)
        _, spec_t = simulate(s, c, dg, F, points=4096, linewidth_hz=4.0)
        _, spec_n = C.simulate(sh, cp, dg, F, points=4096, linewidth_hz=4.0)
        corr = float(np.corrcoef(spec_t.detach().numpy(), spec_n)[0, 1])
        print(f"  {name:10s} corr(torch, numpy oracle) = {corr:.5f}")

    print("\ntorch.autograd.gradcheck on a spectral loss:")
    dg = [2, 1]
    s = torch.tensor([2.0, 5.0], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[0.0, 6.0], [6.0, 0.0]], dtype=torch.float64, requires_grad=True)
    st = _structure(dg, s.device, torch.float64)
    with torch.no_grad():
        _, tgt = simulate(s + 0.05, c, dg, 90.0, points=1024, linewidth_hz=10.0, struct=st)

    def f(s, c):
        _, spec = simulate(s, c, dg, 90.0, points=1024, linewidth_hz=10.0,
                           eigh_eps=1.0, struct=st)
        return ((spec - tgt) ** 2).sum()

    ok = torch.autograd.gradcheck(f, (s, c), eps=1e-6, atol=1e-4, rtol=1e-3)
    print("  gradcheck passed:", ok)


if __name__ == "__main__":
    _selftest()
