"""Verify the numpy reference renderer: forward vs pyspin, gradient vs FD.

Run:  PYTHONPATH=<repo root> python3 model/test_diff_renderer.py
(torch-free; this is the oracle that the torch renderer must match.)
"""
import numpy as np
from model_legacy import diff_renderer_ref as R
from simulation.pyspin.composite import simulate_spectrum_composite

FIELD = 90.0
PTS, LW = 4096, 8.0   # smaller grid + broader lines (smoother -> cleaner FD check)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def fd_grad(shifts, couplings, deg, target, h=2e-4):
    G = len(shifts)
    gS = np.zeros(G)
    for g in range(G):
        sp = list(shifts); sp[g] += h
        sm = list(shifts); sm[g] -= h
        Lp, _, _ = R.loss_and_grad(sp, couplings, deg, FIELD, target, PTS, linewidth_hz=LW)
        Lm, _, _ = R.loss_and_grad(sm, couplings, deg, FIELD, target, PTS, linewidth_hz=LW)
        gS[g] = (Lp - Lm) / (2 * h)
    gJ = np.zeros((G, G))
    for g in range(G):
        for hh in range(g + 1, G):
            Jp = [row[:] for row in couplings]; Jp[g][hh] += h; Jp[hh][g] += h
            Jm = [row[:] for row in couplings]; Jm[g][hh] -= h; Jm[hh][g] -= h
            Lp, _, _ = R.loss_and_grad(shifts, Jp, deg, FIELD, target, PTS, linewidth_hz=LW)
            Lm, _, _ = R.loss_and_grad(shifts, Jm, deg, FIELD, target, PTS, linewidth_hz=LW)
            gJ[g, hh] = gJ[hh, g] = (Lp - Lm) / (2 * h)
    return gS, gJ


def check(name, shifts, couplings, deg, eps):
    # target = spectrum at slightly shifted params (meaningful nonzero grad)
    st = [s + 0.05 for s in shifts]
    _, tgt = R.simulate(st, couplings, deg, FIELD, PTS, linewidth_hz=LW)

    # forward agreement vs pyspin composite (SAME linewidth convention)
    _, mine = R.simulate(shifts, couplings, deg, FIELD, PTS, linewidth_hz=LW)
    _, ref = simulate_spectrum_composite(shifts, couplings, deg, FIELD,
                                         points=PTS, linewidth_hz=LW)
    c = corr(mine, ref)

    L, gS, gJ = R.loss_and_grad(shifts, couplings, deg, FIELD, tgt, PTS,
                                linewidth_hz=LW, eigh_eps=eps)
    fS, fJ = fd_grad(shifts, couplings, deg, tgt)
    an = np.concatenate([gS, gJ[np.triu_indices(len(shifts), 1)]])
    fd = np.concatenate([fS, fJ[np.triu_indices(len(shifts), 1)]])
    finite = np.isfinite(an).all()
    rel = np.linalg.norm(an - fd) / (np.linalg.norm(fd) + 1e-12) if finite else np.inf
    print(f"[{name:28s} eps={eps}] fwd corr vs pyspin={c:.5f} | "
          f"grad finite={finite} | rel L2 err vs FD={rel:.2e}")
    return c, rel, finite


if __name__ == "__main__":
    print("=== asymmetric, all distinct (deg all 1) ===")
    check("asym 4x deg1", [2.0, 3.5, 5.0, 7.0],
          [[0, 7.3, 1.1, 0.4], [7.3, 0, 5.5, 2.2],
           [1.1, 5.5, 0, 8.1], [0.4, 2.2, 8.1, 0]], [1, 1, 1, 1], eps=1.0)

    print("=== A3X: group of 3 equivalent (deg 3) + 1, the degenerate case ===")
    shifts = [3.0, 6.5]
    coup = [[0.0, 6.8], [6.8, 0.0]]
    deg = [3, 1]
    check("A3X deg[3,1]", shifts, coup, deg, eps=1.0)
    check("A3X deg[3,1]", shifts, coup, deg, eps=0.0)   # naive -> should degrade

    print("=== mixed degeneracy [2,1,1] ===")
    check("deg[2,1,1]", [2.5, 4.0, 6.0],
          [[0, 7.0, 1.0], [7.0, 0, 5.0], [1.0, 5.0, 0]], [2, 1, 1], eps=1.0)
