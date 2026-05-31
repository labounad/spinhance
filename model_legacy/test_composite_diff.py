"""Verify the composite differentiable renderer oracle (model.composite_diff).

Run from repo root:
    PYTHONPATH=. python3 -m model_legacy.test_composite_diff

Checks:
  1. forward vs pyspin.simulate_spectrum_composite (corr ~1, areas match),
     incl. high-degeneracy molecules the explicit renderer can't expand;
  2. analytic gradient vs central finite differences (incl. degenerate A3X);
  3. memory-flat FFT-binning broadening matches the dense Lorentzian sum.
"""
import numpy as np

from model_legacy import composite_diff as C
from simulation.pyspin.composite import simulate_spectrum_composite as PYSPIN
from simulation.pyspin.simulator import lorentzian_broaden

F = 90.0


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def test_forward():
    cases = {
        "A3X[3,1]": ([3.0, 6.5], [[0, 6.8], [6.8, 0]], [3, 1]),
        "[2,1,1,1]": ([2.5, 4.0, 6.0, 7.2],
                      [[0, 7, 1, 0], [7, 0, 5, 2], [1, 5, 0, 8], [0, 2, 8, 0]],
                      [2, 1, 1, 1]),
        "[6,1,1]": ([1.2, 3.6, 7.0], [[0, 6.9, 0], [6.9, 0, 0], [0, 0, 0]], [6, 1, 1]),
        # high-degeneracy: a tert-butyl (9H) + CH3 (3H) + 2 singlets, coupled
        "[9,3,1,1]": ([1.0, 1.3, 3.5, 7.4],
                      [[0, 0, 0, 0], [0, 0, 7.0, 0], [0, 7.0, 0, 1.5], [0, 0, 1.5, 0]],
                      [9, 3, 1, 1]),
    }
    for name, (sh, cp, dg) in cases.items():
        _, m = C.simulate(sh, cp, dg, F, points=8192, linewidth_hz=4.0)
        _, r = PYSPIN(sh, cp, dg, F, points=8192, linewidth_hz=4.0)
        c = corr(m, r); ratio = float(m.sum() / r.sum())
        print(f"  {name:11s} n_spins={sum(dg):2d}  corr={c:.5f}  area_ratio={ratio:.4f}")
        assert c > 0.999, f"{name} forward corr {c}"
        assert abs(ratio - 1.0) < 1e-3, f"{name} area ratio {ratio}"
    print("forward vs pyspin (incl. 14-spin t-Bu system): OK")


def _fd(sh, cp, dg, tgt, pts, lw, h=2e-4):
    G = len(sh); gS = np.zeros(G); gJ = np.zeros((G, G))
    for g in range(G):
        sp = list(sh); sp[g] += h; sm = list(sh); sm[g] -= h
        Lp, _, _ = C.loss_and_grad(sp, cp, dg, F, tgt, pts, linewidth_hz=lw)
        Lm, _, _ = C.loss_and_grad(sm, cp, dg, F, tgt, pts, linewidth_hz=lw)
        gS[g] = (Lp - Lm) / (2 * h)
    for g in range(G):
        for k in range(g + 1, G):
            Jp = [r[:] for r in cp]; Jp[g][k] += h; Jp[k][g] += h
            Jm = [r[:] for r in cp]; Jm[g][k] -= h; Jm[k][g] -= h
            Lp, _, _ = C.loss_and_grad(sh, Jp, dg, F, tgt, pts, linewidth_hz=lw)
            Lm, _, _ = C.loss_and_grad(sh, Jm, dg, F, tgt, pts, linewidth_hz=lw)
            gJ[g, k] = gJ[k, g] = (Lp - Lm) / (2 * h)
    return gS, gJ


def test_gradient():
    pts, lw = 2048, 8.0
    cases = {
        "A3X[3,1]": ([3.0, 6.5], [[0, 6.8], [6.8, 0]], [3, 1]),
        "[2,1,1]": ([2.5, 4.0, 6.0], [[0, 7, 1], [7, 0, 5], [1, 5, 0]], [2, 1, 1]),
        "[6,1,1]": ([1.2, 3.6, 7.0], [[0, 6.9, 0], [6.9, 0, 0], [0, 0, 0]], [6, 1, 1]),
    }
    for name, (sh, cp, dg) in cases.items():
        st = [s + 0.05 for s in sh]
        _, tgt = C.simulate(st, cp, dg, F, points=pts, linewidth_hz=lw)
        for eps in (1.0, 0.0) if name == "A3X[3,1]" else (1.0,):
            _, gS, gJ = C.loss_and_grad(sh, cp, dg, F, tgt, pts, linewidth_hz=lw,
                                        eigh_eps=eps)
            fS, fJ = _fd(sh, cp, dg, tgt, pts, lw)
            G = len(sh); iu = np.triu_indices(G, 1)
            an = np.concatenate([gS, gJ[iu]]); fd = np.concatenate([fS, fJ[iu]])
            rel = np.linalg.norm(an - fd) / (np.linalg.norm(fd) + 1e-12)
            print(f"  {name:9s} eps={eps}  rel L2 vs FD={rel:.2e}")
            assert np.isfinite(an).all() and rel < 1e-3, f"{name} grad rel {rel}"
    print("gradient vs finite differences: OK")


def test_broaden_fft_matches_dense():
    # same sticks through dense Lorentzian sum vs bin+FFT-convolve
    pts, a, b = 8192, 0.0, 12.0
    grid = np.linspace(a, b, pts); dx = (b - a) / pts
    hwhm = 4.0 / F / 2.0
    rng = np.random.default_rng(0)
    centers = rng.uniform(1, 10, 200); amps = rng.uniform(0.1, 1, 200)
    dense = (amps[:, None] * (hwhm / ((grid[None, :] - centers[:, None]) ** 2 + hwhm ** 2)) / np.pi).sum(0)
    fft = lorentzian_broaden(centers, amps, pts, a, b, hwhm)
    dense /= dense.sum() * dx; fft /= fft.sum() * dx
    c = corr(dense, fft)
    print(f"  FFT-binning vs dense broadening corr={c:.5f}")
    assert c > 0.999
    print("broadening equivalence: OK")


if __name__ == "__main__":
    test_forward()
    test_gradient()
    test_broaden_fft_matches_dense()
    print("\nALL COMPOSITE-DIFF CHECKS PASSED")
