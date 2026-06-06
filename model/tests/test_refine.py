"""
Test-time spectral refinement (model/inference/refine.py): analysis-by-synthesis
polish of predicted shifts against the input spectrum, with the non-regressing and
cost guards. Tiny 2-spin AX system so the exact simulator is cheap.
"""
import numpy as np
import torch

from model.inference.refine import refine_shifts, refine_system
from model.renderers._torch_exact import simulate

DEG = np.array([1, 1])
CPL = np.array([[0.0, 7.0], [7.0, 0.0]])      # one 7 Hz coupling
TRUE = np.array([2.0, 4.0])                    # ppm
POINTS = 2048


def _target(shifts=TRUE):
    s = torch.as_tensor(shifts, dtype=torch.float64)
    c = torch.as_tensor(CPL, dtype=torch.float64)
    d = torch.as_tensor(DEG)
    _, sp = simulate(s, c, d, 90.0, POINTS, 0.0, 12.0)
    return sp.numpy()


# ── refine_system: 3-spin system with one ABSENT coupling (to test topology) ──────
DEG3 = np.array([1, 1, 1])
TRUE_SH3 = np.array([6.40, 6.30, 6.45])                       # ppm
TRUE_J3 = np.array([[0.0, 8.0, 0.0],                          # J01=8, J12=3, J02 ABSENT (0)
                    [8.0, 0.0, 3.0],
                    [0.0, 3.0, 0.0]])
_IU3 = np.triu_indices(3, 1)


def _target3(sh=TRUE_SH3, J=TRUE_J3, points=2048):
    _, sp = simulate(torch.as_tensor(sh, dtype=torch.float64),
                     torch.as_tensor(J, dtype=torch.float64),
                     torch.as_tensor(DEG3), 90.0, points, 0.0, 12.0)
    return sp.numpy()


# A perturbed prediction: shifts off, present J's off, absent J stays 0. Shared by the
# refine_system effect tests so the speed-optimized refactor must reproduce these outcomes.
PRED_SH3 = np.array([6.44, 6.27, 6.49])
PRED_J3 = np.array([[0.0, 6.0, 0.0],
                    [6.0, 0.0, 4.2],
                    [0.0, 4.2, 0.0]])
RS_KW = dict(field_mhz=90.0, points=2048, steps_per_scale=25, scales_ppm=(0.10, 0.0))


def _smae(p, t): return float(np.abs(np.sort(p)[::-1] - np.sort(t)[::-1]).mean())
def _jmae(C, T): m = np.abs(T[_IU3]) > 0.5; return float(np.abs(C[_IU3][m] - T[_IU3][m]).mean())


def test_refine_reduces_loss_and_moves_toward_truth():
    tgt = _target()
    s0 = np.array([2.18, 3.82])               # perturbed, inside the default trust box
    refined, info = refine_shifts(s0, CPL, DEG, tgt, points=POINTS, n_steps=60)
    assert not info["skipped"] and not info["reverted"]
    assert info["loss1"] < info["loss0"]                       # the objective improved
    assert np.abs(refined - TRUE).sum() < np.abs(s0 - TRUE).sum()   # closer to truth


def test_refine_stays_in_trust_region():
    tgt = _target()
    s0 = np.array([2.30, 3.70])
    trust = 0.1
    refined, _ = refine_shifts(s0, CPL, DEG, tgt, points=POINTS, n_steps=60, trust=trust)
    assert np.all(np.abs(refined - s0) <= trust + 1e-6)


def test_refine_non_regressing_on_correct_start():
    tgt = _target()
    refined, info = refine_shifts(TRUE.copy(), CPL, DEG, tgt, points=POINTS, n_steps=60)
    # already optimal: must never make the spectral objective worse
    assert info["loss1"] <= info["loss0"] + 1e-9


def test_refine_cost_guard_skips_and_returns_input():
    tgt = _target()
    s0 = np.array([2.2, 3.8])
    refined, info = refine_shifts(s0, CPL, DEG, tgt, points=POINTS, max_cost=1)
    assert info["skipped"] and info["steps"] == 0
    assert np.allclose(refined, s0)


# ── refine_system effect tests (the speed refactor must keep all of these passing) ──

def test_refine_system_improves_both_shift_and_J():
    tgt = _target3()
    nsh, ncp, info = refine_system(PRED_SH3, PRED_J3, DEG3, tgt, **RS_KW)
    assert not info["skipped"]
    assert _smae(nsh, TRUE_SH3) < _smae(PRED_SH3, TRUE_SH3)       # shifts closer to truth
    assert _jmae(ncp, TRUE_J3) < _jmae(PRED_J3, TRUE_J3)          # J closer to truth (the new capability)


def test_refine_system_reduces_spectral_loss():
    tgt = _target3()
    _, _, info = refine_system(PRED_SH3, PRED_J3, DEG3, tgt, **RS_KW)
    assert info["loss1"] < info["loss0"]


def test_refine_system_never_activates_absent_coupling():
    """A coupling the model predicted 0 must stay exactly 0 — never turn one on."""
    tgt = _target3()
    _, ncp, _ = refine_system(PRED_SH3, PRED_J3, DEG3, tgt, **RS_KW)
    assert ncp[0, 2] == 0.0 and ncp[2, 0] == 0.0                  # the absent edge


def test_refine_system_respects_trust_region():
    tgt = _target3()
    nsh, ncp, _ = refine_system(PRED_SH3, PRED_J3, DEG3, tgt,
                                trust_shift=0.08, trust_j=2.0, **RS_KW)
    assert np.all(np.abs(nsh - PRED_SH3) <= 0.08 + 1e-6)
    pres = np.abs(PRED_J3[_IU3]) > 0.5
    assert np.all(np.abs(ncp[_IU3][pres] - PRED_J3[_IU3][pres]) <= 2.0 + 1e-6)


def test_refine_system_non_regressing_on_correct_start():
    tgt = _target3()
    _, _, info = refine_system(TRUE_SH3.copy(), TRUE_J3.copy(), DEG3, tgt, **RS_KW)
    assert info["loss1"] <= info["loss0"] + 1e-9


def test_refine_system_cost_guard_skips():
    tgt = _target3()
    nsh, ncp, info = refine_system(PRED_SH3, PRED_J3, DEG3, tgt, max_cost=1, **RS_KW)
    assert info["skipped"]
    assert np.allclose(nsh, PRED_SH3) and np.allclose(ncp, PRED_J3)
