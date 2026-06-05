"""
Test-time spectral refinement (model/inference/refine.py): analysis-by-synthesis
polish of predicted shifts against the input spectrum, with the non-regressing and
cost guards. Tiny 2-spin AX system so the exact simulator is cheap.
"""
import numpy as np
import torch

from model.inference.refine import refine_shifts
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
