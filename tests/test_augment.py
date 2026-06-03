"""Tests for the variability bake (mol_to_spin_system.augment): over-dispersion,
class-aware shift sharing, and sign-preserving coupling clamps (#7 / #7b)."""
import numpy as np

from mol_to_spin_system.augment import (
    DEFAULT_FLOOR, shift_sigma, sample_record,
)

_REC = {
    "labels": ["A", "B", "C"],
    "spin_groups": [[7.50, 1], [7.50, 1], [2.30, 3]],   # A,B equivalent aromatic + C methyl
    "shift_range": [[7.50, 7.50], [7.50, 7.50], [2.30, 2.30]],  # Pretsch point estimate (degenerate)
    "couplings": [["A", "B", 7.5], ["A", "C", -0.7]],   # aromatic (+) and long-range (-)
    "coupling_types": ["aromatic", "long_range"],
}


def test_overdispersion_floor():
    # Pretsch shifts are point estimates -> degenerate range -> sigma == the
    # (deliberately over-dispersed) floor, not the natural ~0.1 ppm spread.
    assert shift_sigma(7.5, 7.5) == DEFAULT_FLOOR
    assert DEFAULT_FLOOR >= 0.12   # over-dispersed vs ~0.1 natural


def test_shift_overdispersion_spread():
    draws = [sample_record(_REC, rng=np.random.default_rng(k)) for k in range(500)]
    a = [d["spin_groups"][0][0] for d in draws]
    assert 0.10 < float(np.std(a)) < 0.22   # ~0.15


def test_class_aware_equivalent_groups_share_draw():
    # equivalent groups (same mean+range) must get ONE shared shift each draw,
    # else an AA'BB' system would degrade to ABCD.
    for k in range(50):
        d = sample_record(_REC, rng=np.random.default_rng(k))
        assert d["spin_groups"][0][0] == d["spin_groups"][1][0]


def test_coupling_sign_preserved_and_bounded():
    for k in range(300):
        d = sample_record(_REC, rng=np.random.default_rng(k))
        jab = d["couplings"][0][2]   # aromatic, base +7.5
        jac = d["couplings"][1][2]   # long-range, base -0.7
        assert jab > 0 and jac < 0           # signs never flip
        assert abs(jab) <= 25 and abs(jac) <= 25
