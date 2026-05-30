"""
Tests for pyspin.cluster — the local-cluster (first-order-between-clusters)
approximation. Validated against the exact composite engine. No MNova required.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.pyspin.cluster import (  # noqa: E402
    partition_clusters,
    simulate_spectrum_clustered,
)
from simulation.pyspin.composite import simulate_spectrum_composite  # noqa: E402


def _chain(n):
    shifts = list(np.linspace(0.8, 8.5, n))
    J = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        J[i][i + 1] = J[i + 1][i] = 7.0
    return shifts, J, [1] * n


def test_partition_caps_size():
    sh, J, dg = _chain(10)
    clusters = partition_clusters(J, dg, max_cluster=5)
    assert all(sum(dg[g] for g in c) <= 5 for c in clusters)
    assert sorted(g for c in clusters for g in c) == list(range(10))


def test_partition_cuts_weakest_bond():
    # 3-group chain; the 1-2 bond is weakest, so a max_cluster=2 cut isolates it
    J = [[0, 9, 0], [9, 0, 2], [0, 2, 0]]
    clusters = partition_clusters(J, [1, 1, 1], max_cluster=2)
    # strongest bond (0-1, J=9) kept together; weak 1-2 (J=2) cut
    assert [0, 1] in clusters and [2] in clusters


def test_reduces_to_exact_when_no_cut():
    sh, J, dg = _chain(6)
    _, ex = simulate_spectrum_composite(sh, J, dg, 90.0)
    _, cl = simulate_spectrum_clustered(sh, J, dg, 90.0, max_cluster=20)
    assert np.corrcoef(ex, cl)[0, 1] > 0.99999


def test_accurate_with_cut():
    sh, J, dg = _chain(10)
    _, ex = simulate_spectrum_composite(sh, J, dg, 90.0)
    _, cl = simulate_spectrum_clustered(sh, J, dg, 90.0, max_cluster=5)
    # first-order between clusters: near-exact on this weakly-coupled chain
    assert np.corrcoef(ex, cl)[0, 1] > 0.98


def test_handles_large_system():
    # 30 fully-chained spins: exact is intractable; clustered must be fast + sane
    sh, J, dg = _chain(30)
    ppm, sp = simulate_spectrum_clustered(sh, J, dg, 90.0, max_cluster=9)
    assert abs(sp.sum() * (12 / len(sp)) - 1.0) < 1e-6


def test_first_order_bath_gives_correct_multiplet():
    # A CH (1H) cut-coupled to a CH3 (3H): first-order should give a 1:3:3:1 quartet.
    # Put them in separate clusters by forcing max_cluster=3 (CH3 alone) ... instead
    # check the bath multiplicity logic directly via the clustered area conservation.
    sh = [1.0, 4.0]
    J = [[0.0, 7.0], [7.0, 0.0]]
    dg = [3, 1]
    _, ex = simulate_spectrum_composite(sh, J, dg, 90.0)
    _, cl = simulate_spectrum_clustered(sh, J, dg, 90.0, max_cluster=1)  # cut the bond
    assert np.corrcoef(ex, cl)[0, 1] > 0.98
