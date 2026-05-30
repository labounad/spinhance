"""
Tests for spectrum_io (dense / sparse / peaks representations) and export.
No MNova required.
"""

import sys
import tarfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from simulation.spectrum_io import (  # noqa: E402
    sparsify, save_dense, save_sparse, save_peaks, load_spectrum,
)
from simulation.export import export_spectra  # noqa: E402
from simulation.pyspin.composite import (  # noqa: E402
    simulate_spectrum_composite, composite_transitions,
)


# ── sparsify (thresholded broadened) ──────────────────────────────────────────

def test_sparsify_drops_below_cutoff_and_renormalizes():
    y = np.zeros(1000)
    y[100] = 1.0
    y[200] = 0.0005      # below 0.001 * max → dropped
    y[300] = 0.5
    idx, val = sparsify(y, cutoff=0.001, renormalize=True)
    assert set(idx.tolist()) == {100, 300}
    # renormalised so the kept sum equals the original sum (∫ preserved)
    assert abs(val.sum() - y.sum()) < 1e-5


def test_sparse_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    y = np.abs(rng.standard_normal(16384)) * (np.arange(16384) % 503 == 0)
    p = save_sparse(tmp_path / "s.npz", y, cutoff=0.001)
    z = load_spectrum(p)
    assert z.shape == y.shape
    # nonzero structure preserved (sparse keeps the peaks)
    assert np.corrcoef(z, y)[0, 1] > 0.999


# ── peaks (line list, lineshape applied on load) ──────────────────────────────

def test_peaks_roundtrip_matches_dense(tmp_path):
    # methyl + CH AX3-ish system
    shifts = [1.2, 3.6]
    couplings = [[0.0, 7.0], [7.0, 0.0]]
    deg = [3, 1]
    _, dense = simulate_spectrum_composite(shifts, couplings, deg, 90.0)
    centers, amps = composite_transitions(shifts, couplings, deg, 90.0)
    p = save_peaks(tmp_path / "pk.npz", centers, amps,
                   linewidth_hz=1.0, field_mhz=90.0)
    recon = load_spectrum(p)
    assert recon.shape == dense.shape
    assert np.corrcoef(recon, dense)[0, 1] > 0.9999
    assert abs(recon.sum() * (12 / len(recon)) - 1.0) < 1e-6  # ∫ = 1


def test_peaks_far_smaller_than_dense(tmp_path):
    shifts = [1.2, 3.6, 7.2]
    n = 3
    J = [[0.0] * n for _ in range(n)]
    J[0][1] = J[1][0] = 7.0
    J[1][2] = J[2][1] = 6.0
    deg = [3, 1, 1]
    _, dense = simulate_spectrum_composite(shifts, J, deg, 90.0)
    centers, amps = composite_transitions(shifts, J, deg, 90.0)
    dp = save_dense(tmp_path / "d.npy", dense)
    pp = save_peaks(tmp_path / "p.npz", centers, amps, linewidth_hz=1.0, field_mhz=90.0)
    assert pp.stat().st_size < dp.stat().st_size  # line list beats dense array


# ── end-to-end export → reload ────────────────────────────────────────────────

def test_export_then_reload(tmp_path):
    # build a tiny spectra dir with 3 dense spectra + ppm_axis + index.csv
    sd = tmp_path / "spectra" / "90MHz"
    sd.mkdir(parents=True)
    np.save(sd / "ppm_axis.npy", np.linspace(0, 12, 16384))
    (tmp_path / "spectra" / "index.csv").write_text("index,id\nmol_000000,A\n")
    originals = {}
    for i in range(3):
        _, y = simulate_spectrum_composite([1.0 + i, 3.5], [[0, 7], [7, 0]], [3, 1], 90.0)
        np.save(sd / f"mol_{i:06d}.npy", y.astype(np.float32))
        originals[i] = y

    out = tmp_path / "pack.tar.gz"
    res = export_spectra(tmp_path / "spectra", out, sparsify_data=True, cutoff=0.001)
    assert res["spectra"] == 3 and out.exists()

    # extract and reload one sparse spectrum; must match the original
    ex = tmp_path / "ex"
    with tarfile.open(out) as tar:
        tar.extractall(ex)
    recon = load_spectrum(ex / "90MHz" / "mol_000000.npz")
    assert np.corrcoef(recon, originals[0])[0, 1] > 0.999
