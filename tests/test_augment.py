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
    "equiv_orbit": [0, 0, 1],   # A,B are one symmetry orbit (AA'BB' siblings); C distinct
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


def test_orbit_equivalent_groups_share_draw():
    # groups in the SAME symmetry orbit (A,B) must get ONE shared shift each draw,
    # or an AA'BB' system would degrade to ABCD.
    for k in range(50):
        d = sample_record(_REC, rng=np.random.default_rng(k))
        assert d["spin_groups"][0][0] == d["spin_groups"][1][0]


def test_coincidental_collision_sampled_independently():
    # Audit-2 A1: two groups with the SAME base shift+range but DIFFERENT orbits
    # are NOT equivalent (their Pretsch values merely coincide) -> independent
    # draws. The old (mean,range) key wrongly locked them together; the orbit key
    # must not.
    rec = dict(_REC)
    rec["equiv_orbit"] = [0, 1, 2]   # all distinct orbits despite A,B sharing a base shift
    distinct = sum(
        sample_record(rec, rng=np.random.default_rng(k))["spin_groups"][0][0]
        != sample_record(rec, rng=np.random.default_rng(k))["spin_groups"][1][0]
        for k in range(50)
    )
    assert distinct >= 45   # independent Gaussians -> almost always differ


def test_missing_orbit_falls_back_to_independent():
    # legacy records without equiv_orbit -> every group sampled independently
    rec = {k: v for k, v in _REC.items() if k != "equiv_orbit"}
    distinct = sum(
        sample_record(rec, rng=np.random.default_rng(k))["spin_groups"][0][0]
        != sample_record(rec, rng=np.random.default_rng(k))["spin_groups"][1][0]
        for k in range(50)
    )
    assert distinct >= 45


def test_coupling_sign_preserved_and_bounded():
    for k in range(300):
        d = sample_record(_REC, rng=np.random.default_rng(k))
        jab = d["couplings"][0][2]   # aromatic, base +7.5
        jac = d["couplings"][1][2]   # long-range, base -0.7
        assert jab > 0 and jac < 0           # signs never flip
        assert abs(jab) <= 25 and abs(jac) <= 25
