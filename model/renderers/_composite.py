"""
model.composite_diff
====================
NumPy ORACLE: manifold-reduced (composite) DIFFERENTIABLE 1H renderer.

Why this exists
---------------
The explicit renderer builds one dense 2^N Hamiltonian (N = total protons), which
blows up for high-degeneracy molecules (a t-Bu = 2^9). This engine mirrors
``simulation.pyspin.composite`` instead: each group of d equivalent spins is
reduced to its total-spin manifolds, and the Hamiltonian is block-diagonalised by
total magnetisation (Mz). We diagonalise the small Mz blocks, never a 2^N matrix.

Differences from pyspin.composite (deliberate, for a TRAINING loss):
  * NO connected-component splitting. Predicted couplings are continuous/soft-
    gated, so the coupling graph is effectively dense and component structure
    would be non-differentiable across J=0. Manifold reduction needs no zeros and
    is fully differentiable, so we keep only that. (Physics identical; we just
    forgo an optimisation that can't fire at train time.)
  * Analytic reverse-mode gradient of a spectral loss w.r.t. shifts (ppm) and
    couplings (Hz), via the Lorentzian-regularised eigh VJP (de-risked earlier),
    applied PER SMALL BLOCK -- so degeneracy handling and memory both stay benign.

This is the verified oracle; ``model.diff_renderer_torch`` mirrors it in torch.
Verification: forward vs pyspin (corr ~1), gradient vs finite differences.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from simulation.pyspin.composite import spin_reps, _ops

__all__ = ["simulate", "loss_and_grad", "DEFAULT_EIGH_EPS",
           "build_static_plan", "static_blocks", "max_block_dim"]

DEFAULT_EIGH_EPS = 1.0   # Hz


# -----------------------------------------------------------------------------
# One manifold combination: forward (+ cache) and backward
# -----------------------------------------------------------------------------

def _combo_plan(S_list):
    """Parameter-independent structure for active groups with spins ``S_list``.

    Returns enumeration of product states, their per-group m-indices, Mz value,
    and the raising amplitudes -- everything needed to assemble H and F+ from
    (nu, J) without rebuilding indices.
    """
    active = [g for g, S in enumerate(S_list) if S > 0]
    G = len(active)
    if G == 0:
        return None
    mlist, aplist, dims = [], [], []
    for g in active:
        m, ap = _ops(S_list[g])
        mlist.append(m); aplist.append(ap); dims.append(len(m))
    dims = np.array(dims, dtype=int)
    idx_all = np.array(list(np.ndindex(*[int(d) for d in dims])), dtype=int)  # (T,G)
    Mmat = np.stack([mlist[g][idx_all[:, g]] for g in range(G)], axis=1)      # (T,G)
    Mz = np.round(Mmat.sum(1), 6)
    blocks = {mz: np.nonzero(Mz == mz)[0] for mz in np.unique(Mz)}
    return dict(active=active, G=G, mlist=mlist, aplist=aplist, dims=dims,
                idx_all=idx_all, Mmat=Mmat, Mz=Mz, blocks=blocks)


def _assemble_block(plan, mz, nu, J):
    """Dense block Hamiltonian H (sub, sub) for magnetisation mz, plus templates
    dH/dnu_g and dH/dJ_gh restricted to this block (small)."""
    active, G = plan["active"], plan["G"]
    idx_all, Mmat, aplist, dims = (plan["idx_all"], plan["Mmat"],
                                   plan["aplist"], plan["dims"])
    states = plan["blocks"][mz]
    n = len(states)
    pos = {int(s): i for i, s in enumerate(states)}
    ig = idx_all[states]                       # (n, G) per-group m-index
    Mb = Mmat[states]                          # (n, G) m-values

    H = np.zeros((n, n))
    dnu = [np.zeros((n, n)) for _ in range(G)]
    dJ = {}
    # diagonal: Zeeman + Ising
    nu_act = np.array([nu[active[g]] for g in range(G)])
    for a in range(n):
        H[a, a] += float(np.dot(nu_act, Mb[a]))
        for g in range(G):
            dnu[g][a, a] += Mb[a, g]
        for g in range(G):
            for h in range(g + 1, G):
                H[a, a] += J[active[g]][active[h]] * Mb[a, g] * Mb[a, h]
                dJ.setdefault((g, h), np.zeros((n, n)))[a, a] += Mb[a, g] * Mb[a, h]
    # off-diagonal flip-flop within block: 0.5 J_gh I+_g I-_h
    for g in range(G):
        for h in range(G):
            if g == h:
                continue
            lo, hi = (g, h)
            for a, st in enumerate(states):
                i_g = ig[a, g]; i_h = ig[a, h]
                if i_g > 0 and i_h < dims[h] - 1:        # raise g, lower h
                    tgt = list(idx_all[st]); tgt[g] -= 1; tgt[h] += 1
                    # find target state index in this block
                    # mixed-radix key
                    key = 0
                    for gg in range(G):
                        key = key * dims[gg] + tgt[gg]
                    # map key -> state id
                    b = _key_to_state(plan, key)
                    if b in pos:
                        amp = 0.5 * aplist[g][i_g] * aplist[h][i_h + 1]
                        H[pos[b], a] += J[active[g]][active[h]] * amp
                        pair = (g, h) if g < h else (h, g)
                        dJ.setdefault(pair, np.zeros((n, n)))[pos[b], a] += amp
    return H, dnu, dJ, states, ig


def _key_to_state(plan, key):
    """Mixed-radix key -> global state id (cached on the plan)."""
    cache = plan.setdefault("_key2state", None)
    if cache is None:
        dims = plan["dims"]; idx_all = plan["idx_all"]
        keys = np.zeros(len(idx_all), dtype=np.int64)
        for gg in range(plan["G"]):
            keys = keys * dims[gg] + idx_all[:, gg]
        cache = {int(k): i for i, k in enumerate(keys)}
        plan["_key2state"] = cache
    return cache.get(int(key), -1)


# -----------------------------------------------------------------------------
# Parameter-INDEPENDENT static plan (shared with the torch renderer)
# -----------------------------------------------------------------------------

def static_blocks(plan):
    """Per Mz block, constant arrays to assemble H from (nu, J) without rebuilding
    indices: diagonal Ising coefficients + flip-flop (row, col, pair, amp); plus
    constant F+ matrices linking mz -> mz+1. Torch renderer mirrors this."""
    active, G = plan["active"], plan["G"]
    idx_all, Mmat, aplist, dims = (plan["idx_all"], plan["Mmat"],
                                   plan["aplist"], plan["dims"])
    pairs = [(g, h) for g in range(G) for h in range(g + 1, G)]
    pair_groups = [(active[g], active[h]) for (g, h) in pairs]

    def key(v):
        k = 0
        for gg in range(G):
            k = k * dims[gg] + int(v[gg])
        return int(k)

    keymap = {key(idx_all[s]): s for s in range(len(idx_all))}

    blocks = {}
    for mz, states in plan["blocks"].items():
        n = len(states)
        pos = {int(s): i for i, s in enumerate(states)}
        Mb = Mmat[states]
        ising = (np.stack([Mb[:, g] * Mb[:, h] for (g, h) in pairs], axis=1)
                 if pairs else np.zeros((n, 0)))
        rows, cols, pidx, amps = [], [], [], []
        for a, st in enumerate(states):
            iv = idx_all[st]
            for p, (g, h) in enumerate(pairs):
                if iv[g] > 0 and iv[h] < dims[h] - 1:           # I+_g I-_h
                    tgt = iv.copy(); tgt[g] -= 1; tgt[h] += 1
                    b = keymap.get(key(tgt), -1)
                    if b in pos:
                        rows.append(pos[b]); cols.append(a); pidx.append(p)
                        amps.append(0.5 * aplist[g][iv[g]] * aplist[h][iv[h] + 1])
                if iv[h] > 0 and iv[g] < dims[g] - 1:           # I-_g I+_h (symmetric)
                    tgt = iv.copy(); tgt[h] -= 1; tgt[g] += 1
                    b = keymap.get(key(tgt), -1)
                    if b in pos:
                        rows.append(pos[b]); cols.append(a); pidx.append(p)
                        amps.append(0.5 * aplist[h][iv[h]] * aplist[g][iv[g] + 1])
        blocks[mz] = dict(n=n, Mb=Mb.astype(np.float64), ising=ising.astype(np.float64),
                          rows=np.array(rows, int), cols=np.array(cols, int),
                          pidx=np.array(pidx, int), amps=np.array(amps, np.float64))
    fplus = {}
    for mz in plan["blocks"]:
        up = round(mz + 1.0, 6)
        if up not in plan["blocks"]:
            continue
        lo = plan["blocks"][mz]; hi = plan["blocks"][up]
        pos_hi = {int(s): i for i, s in enumerate(hi)}
        Fp = np.zeros((len(hi), len(lo)))
        for q, st in enumerate(lo):
            iv = idx_all[st]
            for g in range(G):
                if iv[g] > 0:
                    tgt = iv.copy(); tgt[g] -= 1
                    b = keymap.get(key(tgt), -1)
                    if b in pos_hi:
                        Fp[pos_hi[b], q] += aplist[g][iv[g]]
        fplus[mz] = (up, Fp)
    return dict(active=active, G=G, pairs=pairs, pair_groups=pair_groups,
                blocks=blocks, fplus=fplus)


def max_block_dim(degeneracy):
    """Largest Mz-block (the actual eigh size) over all manifold combinations,
    no component split — the real cost metric for the differentiable renderer.
    Far smaller than explicit 2^(total spins) for high-degeneracy groups."""
    reps = [spin_reps(int(d)) for d in degeneracy]
    best = 0
    for combo in product(*reps):
        plan = _combo_plan([c[0] for c in combo])
        if plan is None:
            continue
        best = max(best, max(len(v) for v in plan["blocks"].values()))
    return best


def build_static_plan(degeneracy):
    """List of manifold combinations (S_list, weight, static_blocks) for a
    degeneracy pattern. Parameter-independent; cache per pattern."""
    combos = []
    reps = [spin_reps(int(d)) for d in degeneracy]
    for combo in product(*reps):
        S_list = [c[0] for c in combo]
        weight = 1
        for c in combo:
            weight *= c[1]
        plan = _combo_plan(S_list)
        if plan is None:
            continue
        combos.append((S_list, float(weight), static_blocks(plan)))
    return dict(degeneracy=[int(x) for x in degeneracy], combos=combos)


def _regularized_F(E, eps):
    deltaE = E[None, :] - E[:, None]
    if eps == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(np.abs(deltaE) > 0, 1.0 / deltaE, 0.0)
    else:
        F = deltaE / (deltaE ** 2 + eps ** 2)
    np.fill_diagonal(F, 0.0)
    return F


def _fplus_block(plan, mz, up):
    """F+ matrix (hi states x lo states) connecting block mz -> mz+1."""
    active, G = plan["active"], plan["G"]
    idx_all, aplist, dims = plan["idx_all"], plan["aplist"], plan["dims"]
    lo = plan["blocks"][mz]; hi = plan["blocks"][up]
    pos_hi = {int(s): i for i, s in enumerate(hi)}
    Fp = np.zeros((len(hi), len(lo)))
    for q, st in enumerate(lo):
        for g in range(G):
            i_g = idx_all[st, g]
            if i_g > 0:                       # raise group g (m -> m+1)
                tgt = list(idx_all[st]); tgt[g] -= 1
                key = 0
                for gg in range(G):
                    key = key * dims[gg] + tgt[gg]
                b = _key_to_state(plan, key)
                if b in pos_hi:
                    Fp[pos_hi[b], q] += aplist[g][i_g]
    return Fp, lo, hi


# -----------------------------------------------------------------------------
# Transitions for the whole system (sum over manifold combinations)
# -----------------------------------------------------------------------------

def _system(shifts, couplings, degeneracy, field, eigh_eps, want_grad):
    nu = [shifts[g] * field for g in range(len(shifts))]
    J = [[couplings[a][b] for b in range(len(shifts))] for a in range(len(shifts))]
    reps = [spin_reps(int(degeneracy[g])) for g in range(len(shifts))]

    all_f, all_a = [], []
    grad_cache = []
    for combo in product(*reps):
        S_list = [c[0] for c in combo]
        weight = 1
        for c in combo:
            weight *= c[1]
        plan = _combo_plan(S_list)
        if plan is None:
            continue
        # diagonalise each block
        Eo, Vo, Ho_tpl = {}, {}, {}
        for mz in plan["blocks"]:
            H, dnu, dJ, states, ig = _assemble_block(plan, mz, nu, J)
            E, V = np.linalg.eigh(H)
            Eo[mz] = E; Vo[mz] = V
            Ho_tpl[mz] = (dnu, dJ)
        # transitions mz -> mz+1
        combo_trans = []
        for mz in sorted(plan["blocks"]):
            up = round(mz + 1.0, 6)
            if up not in plan["blocks"]:
                continue
            Fp, lo, hi = _fplus_block(plan, mz, up)
            M = Vo[up].T @ Fp @ Vo[mz]
            amps = (M * M) * weight
            freqs = Eo[up][:, None] - Eo[mz][None, :]
            all_f.append(freqs.ravel()); all_a.append(amps.ravel())
            combo_trans.append((mz, up, Fp, M))
        if want_grad:
            grad_cache.append((plan, S_list, weight, Eo, Vo, Ho_tpl, combo_trans))
    freqs = np.concatenate(all_f) if all_f else np.array([])
    amps = np.concatenate(all_a) if all_a else np.array([])
    return freqs, amps, grad_cache


# -----------------------------------------------------------------------------
# Broadening (dense, with backward) — oracle on small systems
# -----------------------------------------------------------------------------

def _grid(points, a, b):
    g = np.linspace(a, b, points)
    return g, (b - a) / points


def _broaden_dense(centers, amps, grid, hwhm):
    x = grid[None, :] - centers[:, None]
    L = (hwhm / (x ** 2 + hwhm ** 2)) / np.pi
    return (amps[:, None] * L), L, x


def simulate(shifts, couplings, degeneracy, field_mhz, points=16384,
             ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0):
    freqs, amps, _ = _system(shifts, couplings, degeneracy, field_mhz, 0.0, False)
    grid, dx = _grid(points, ppm_from, ppm_to)
    centers = freqs / field_mhz
    hwhm = (linewidth_hz / 2.0) / field_mhz
    keep = (centers >= ppm_from) & (centers <= ppm_to) & (amps > 0)
    raw, _, _ = _broaden_dense(centers[keep], amps[keep], grid, hwhm)
    spec = raw.sum(0)
    area = spec.sum() * dx
    return grid, (spec / area if area > 0 else spec)


def loss_and_grad(shifts, couplings, degeneracy, field_mhz, target,
                  points=16384, ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0,
                  eigh_eps=DEFAULT_EIGH_EPS):
    G = len(shifts)
    grid, dx = _grid(points, ppm_from, ppm_to)
    hwhm = (linewidth_hz / 2.0) / field_mhz
    freqs, amps, gcache = _system(shifts, couplings, degeneracy, field_mhz,
                                  eigh_eps, True)
    centers = freqs / field_mhz
    keep = (centers >= ppm_from) & (centers <= ppm_to) & (amps > 0)
    ck, ak = centers[keep], amps[keep]
    raw, L, xk = _broaden_dense(ck, ak, grid, hwhm)
    spec_un = raw.sum(0)
    area = spec_un.sum() * dx
    spec = spec_un / area
    loss = float(np.sum((spec - target) ** 2) * dx)

    # ---- backward to per-transition (centers, amps) ----
    dL_dspec = 2.0 * (spec - target) * dx
    dL_dspec_un = dL_dspec / area - (np.dot(dL_dspec, spec_un) / area ** 2) * dx
    dL_dak = (L * dL_dspec_un[None, :]).sum(1)
    dLdc = (hwhm * 2.0 * xk / (xk ** 2 + hwhm ** 2) ** 2) / np.pi
    dL_dck = (ak[:, None] * dLdc * dL_dspec_un[None, :]).sum(1)
    # scatter back to full transition order
    dL_damp = np.zeros(len(amps)); dL_damp[keep] = dL_dak
    dL_dcenter = np.zeros(len(freqs)); dL_dcenter[keep] = dL_dck
    dL_dfreq = dL_dcenter / field_mhz

    dL_dshift = np.zeros(G)
    dL_dcoup = np.zeros((G, G))

    # ---- walk combinations, accumulate per-block dL/dE, dL/dV, then eigh VJP ----
    off = 0
    for (plan, S_list, weight, Eo, Vo, Ho_tpl, combo_trans) in gcache:
        active = plan["active"]
        dE = {mz: np.zeros_like(Eo[mz]) for mz in Eo}
        dV = {mz: np.zeros_like(Vo[mz]) for mz in Vo}
        for (mz, up, Fp, M) in combo_trans:
            nlo = Vo[mz].shape[0]; nhi = Vo[up].shape[0]
            cnt = nhi * nlo
            damp = (dL_damp[off:off + cnt] * weight).reshape(nhi, nlo)
            dfreq = dL_dfreq[off:off + cnt].reshape(nhi, nlo)
            off += cnt
            dM = 2.0 * M * damp
            dV[up] += Fp @ Vo[mz] @ dM.T
            dV[mz] += Fp.T @ Vo[up] @ dM
            dE[up] += dfreq.sum(1)
            dE[mz] += -dfreq.sum(0)
        # eigh VJP per block -> dL/dH_block -> params via templates
        for mz in Eo:
            E, V = Eo[mz], Vo[mz]
            F = _regularized_F(E, eigh_eps)
            dH = V @ (np.diag(dE[mz]) + F * (V.T @ dV[mz])) @ V.T
            dH = 0.5 * (dH + dH.T)
            dnu, dJ = Ho_tpl[mz]
            Gp = plan["G"]
            for g in range(Gp):
                dL_dshift[active[g]] += field_mhz * np.sum(dH * dnu[g])
            for (g, h), T in dJ.items():
                gg, hh = active[g], active[h]
                val = np.sum(dH * T)
                dL_dcoup[gg, hh] += val
                dL_dcoup[hh, gg] += val
    return loss, dL_dshift, dL_dcoup
