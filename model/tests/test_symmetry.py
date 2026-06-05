"""
Label-invariant coupling comparison (model/evaluation/symmetry.py) and the matching
data-side relabeling generator (model/data/dataset._orbit_perms).

These guard the symmetry-aware J metric/loss that the whole rebuild fleet is scored
and trained on: among nodes with equal (shift, degeneracy) the canonical sort's
tie-break is arbitrary, so coupling comparison must be invariant to within-class
relabeling (the AA'XX' case) — but must NOT rescue genuinely-wrong couplings.
"""
import numpy as np

from model.data.dataset import SYM_PERMS_MAX, _orbit_perms
from model.evaluation.symmetry import align_pred_couplings, label_permutations


def _cm(G, edges):
    """symmetric (G,G) coupling matrix + presence mask from {(i,j): val}."""
    cm = np.zeros((G, G)); mask = np.zeros((G, G))
    for (i, j), v in edges.items():
        cm[i, j] = cm[j, i] = v
        mask[i, j] = mask[j, i] = 1.0
    return cm, mask


# ── label_permutations ───────────────────────────────────────────────────────

def test_no_symmetry_returns_identity_only():
    perms = label_permutations([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1])
    assert len(perms) == 1
    assert np.array_equal(perms[0], np.arange(4))


def test_equal_shift_and_deg_generates_swap():
    # groups 0,1 share (shift, deg) -> identity + the (0,1) swap
    perms = label_permutations([1.0, 1.0, 2.0, 3.0], [1, 1, 1, 1])
    as_tuples = {tuple(p) for p in perms}
    assert as_tuples == {(0, 1, 2, 3), (1, 0, 2, 3)}


def test_equal_shift_but_different_deg_not_grouped():
    # same shift, different degeneracy => not chemically equivalent => no swap
    perms = label_permutations([1.0, 1.0, 2.0], [1, 2, 1])
    assert len(perms) == 1


def test_max_perms_guard_falls_back_to_identity():
    # 7 equal nodes => 7! = 5040 > max_perms(720) => identity only (no blow-up)
    perms = label_permutations([1.0] * 7, [1] * 7)
    assert len(perms) == 1


# ── align_pred_couplings ─────────────────────────────────────────────────────

def test_align_recovers_equivalent_relabeling():
    # AA'XX'-style: groups 0,1 equal-shift; swapping them flips which ortho/para J
    # each carries. A prediction that broke the tie the OTHER way is the SAME spectrum
    # and must score 0 after alignment.
    shifts, deg = [1.0, 1.0, 2.0, 3.0], [1, 1, 1, 1]
    tgt, mask = _cm(4, {(0, 2): 5.0, (1, 3): 7.0})
    swap = np.array([1, 0, 2, 3])
    pred = tgt[np.ix_(swap, swap)]                 # relabeled (equivalent) prediction
    # un-aligned masked error is large...
    raw_err = float((np.abs(pred - tgt) * mask).sum())
    assert raw_err > 1.0
    # ...but alignment recovers it exactly.
    aligned = align_pred_couplings(pred, tgt, mask, shifts, deg)
    assert float((np.abs(aligned - tgt) * mask).sum()) < 1e-9


def test_align_does_not_rescue_wrong_couplings():
    # genuinely wrong VALUE (not a relabeling) cannot be permuted into a match
    shifts, deg = [1.0, 1.0, 2.0, 3.0], [1, 1, 1, 1]
    tgt, mask = _cm(4, {(0, 2): 5.0, (1, 3): 7.0})
    pred, _ = _cm(4, {(0, 2): 99.0, (1, 3): 7.0})
    aligned = align_pred_couplings(pred, tgt, mask, shifts, deg)
    assert float((np.abs(aligned - tgt) * mask).sum()) > 1.0


def test_align_never_worse_than_identity():
    rng = np.random.default_rng(0)
    shifts, deg = [1.0, 1.0, 2.0, 2.0], [1, 1, 1, 1]   # two swappable classes
    tgt, mask = _cm(4, {(0, 2): 3.0, (1, 3): 4.0, (0, 3): 1.0})
    pred = rng.normal(size=(4, 4)); pred = (pred + pred.T) / 2
    id_err = float((np.abs(pred - tgt) * mask).sum())
    aligned = align_pred_couplings(pred, tgt, mask, shifts, deg)
    assert float((np.abs(aligned - tgt) * mask).sum()) <= id_err + 1e-9


def test_align_noop_without_symmetry_returns_input():
    shifts, deg = [1.0, 2.0, 3.0], [1, 1, 1]
    pred, mask = _cm(3, {(0, 1): 5.0})
    tgt, _ = _cm(3, {(0, 1): 9.0})
    out = align_pred_couplings(pred, tgt, mask, shifts, deg)
    assert out is pred                                # identity-only -> returned untouched


# ── _orbit_perms (data side: feeds batch.sym_perms) ──────────────────────────

def test_orbit_perms_shape_and_identity_row():
    perms = _orbit_perms([0, 0, 1, 2])               # groups 0,1 share an orbit
    assert perms.shape == (SYM_PERMS_MAX, 4)
    assert np.array_equal(perms[0], np.arange(4))     # row 0 is always identity
    assert {tuple(p) for p in perms} == {(0, 1, 2, 3), (1, 0, 2, 3)}  # swap + identity pad


def test_orbit_perms_trivial_is_identity_padded():
    perms = _orbit_perms([0, 1, 2, 3])               # no shared orbit
    assert perms.shape == (SYM_PERMS_MAX, 4)
    assert all(np.array_equal(p, np.arange(4)) for p in perms)
