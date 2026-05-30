"""
Tests for simulation.benchmarks.benchmark_fields.geometric_frequencies.

These cover only the pure frequency-grid logic — no MestReNova required.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.benchmarks.benchmark_fields import geometric_frequencies  # noqa: E402


def test_endpoints_and_count():
    f = geometric_frequencies(40.0, 1200.0, 100)
    assert len(f) == 100
    assert f[0] == pytest.approx(40.0)
    assert f[-1] == pytest.approx(1200.0)


def test_strictly_increasing():
    f = geometric_frequencies(40.0, 1200.0, 100)
    assert all(b > a for a, b in zip(f, f[1:]))


def test_gaps_grow_monotonically():
    """Denser at low field, sparser at high field => gaps increase."""
    f = np.array(geometric_frequencies(40.0, 1200.0, 100))
    gaps = np.diff(f)
    assert np.all(np.diff(gaps) > 0)
    # And the high-field gap is much larger than the low-field gap.
    assert gaps[-1] > 10 * gaps[0]


def test_constant_ratio():
    """Geometric spacing => consecutive ratios are (nearly) constant."""
    f = np.array(geometric_frequencies(40.0, 1200.0, 100))
    ratios = f[1:] / f[:-1]
    assert np.allclose(ratios, ratios[0], rtol=1e-9)


@pytest.mark.parametrize("bad", [
    (40.0, 1200.0, 1),     # n < 2
    (0.0, 1200.0, 10),     # fmin not positive
    (1200.0, 40.0, 10),    # fmin >= fmax
])
def test_invalid_args_raise(bad):
    with pytest.raises(ValueError):
        geometric_frequencies(*bad)
