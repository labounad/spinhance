"""
Tests for pyspin.composite — composite-particle reduction.

Validates the reduced engine against the explicit spin-½ simulator (they must
produce identical spectra) and checks the spin-multiplicity bookkeeping.
No MestReNova required.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.pyspin.composite import spin_reps, simulate_spectrum_composite  # noqa: E402
from simulation.pyspin.simulator import simulate_spectrum  # noqa: E402


def test_spin_reps_methyl():
    # CH3: one S=3/2 manifold, two S=1/2 manifolds
    assert spin_reps(3) == [(1.5, 1), (0.5, 2)]


def test_spin_reps_ch2_and_total_dim():
    assert spin_reps(2) == [(1.0, 1), (0.0, 1)]
    # multiplicities weighted by manifold dimension must recover 2^d
    for d in (1, 2, 3, 4, 9):
        total = sum(int(round(2 * S + 1)) * w for S, w in spin_reps(d))
        assert total == 2 ** d


def test_composite_matches_spin_half_AX3():
    # methyl (CH3) coupled to one CH: composite must equal explicit expansion
    shifts = [1.20, 3.80]
    couplings = [[0.0, 6.8], [6.8, 0.0]]
    degeneracy = [3, 1]
    _, ref = simulate_spectrum(shifts, couplings, degeneracy, 90.0)
    _, comp = simulate_spectrum_composite(shifts, couplings, degeneracy, 90.0)
    assert np.corrcoef(ref, comp)[0, 1] > 0.99999
    assert abs(ref.sum() - comp.sum()) / ref.sum() < 1e-6


def test_composite_matches_spin_half_multi_group():
    shifts = [0.9, 1.6, 2.4, 3.6]
    n = 4
    J = [[0.0] * n for _ in range(n)]
    def s(i, j, v): J[i][j] = J[j][i] = v
    s(0, 1, 7.0); s(1, 2, 7.2); s(2, 3, 6.5)
    degeneracy = [3, 2, 1, 2]
    _, ref = simulate_spectrum(shifts, J, degeneracy, 90.0)
    _, comp = simulate_spectrum_composite(shifts, J, degeneracy, 90.0)
    assert np.corrcoef(ref, comp)[0, 1] > 0.99999
