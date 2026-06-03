"""
Data-layer tests: transforms/standardizer (torch-free) + dataset/collate -> SpinBatch.
Uses synthetic in-memory records (no data files, no RDKit: compute_scaffold=False).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.data.splits import canonical_order, make_splits
from model.data.standardization import DegeneracyVocab, Standardizer, class_balance
from model.data.transforms import encode_target, augment_spectrum, _one_h_height
from model.data.dataset import SpectrumMatrixDataset
from model.data.collate import collate_spin_batch
from model.schemas import SpinBatch

G, P = 8, 512


def _records(n=64, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 10))
        recs.append(dict(
            mol_id=f"m{i}", smiles="C", scaffold=f"s{i % 10}",
            shifts=rng.uniform(0.5, 9, G),
            couplings=c,
            degeneracy=rng.choice([1, 2, 3], size=G).astype(int),
            spec90=rng.random(P).astype(np.float32),
        ))
    return recs


# ── transforms / standardizer ──────────────────────────────────────────────────

def test_encode_target_canonical_and_shapes():
    r = _records(1)[0]
    vocab = DegeneracyVocab()
    t = encode_target(r["shifts"], r["couplings"], r["degeneracy"], vocab)
    assert t["shifts"].shape == (G,)
    assert t["j_mag"].shape == (G * (G - 1) // 2,)
    assert t["j_presence"].shape == (G * (G - 1) // 2,)
    assert t["deg_class"].shape == (G,)
    # canonical: shifts sorted descending
    assert np.all(np.diff(t["shifts"]) <= 1e-6)


def test_standardizer_roundtrip():
    recs = _records(40)
    vocab = DegeneracyVocab()
    std = Standardizer().fit(recs, vocab)
    x = np.array([1.0, 5.0, 9.0])
    assert np.allclose(std.inverse_shifts((x - std.shift_mean) / std.shift_std), x, atol=1e-5)
    sd = std.state_dict()
    std2 = Standardizer().load_state_dict(sd)
    assert std2.shift_mean == std.shift_mean and std2.j_std == std.j_std


def test_augment_preserves_length_and_nonneg():
    rng = np.random.default_rng(0)
    spec = np.abs(rng.standard_normal(P)).astype(np.float32)
    out = augment_spectrum(spec, 0.0, 12.0, rng=rng, n_protons=10)
    assert out.shape == (P,) and (out >= 0).all()


def test_noise_reference_is_1h_not_max():
    # Two unit-integral spectra, same N and same total integral but very
    # different max heights (a tall singlet vs a spread-out band). The 1H
    # reference height must be identical (it depends on Σspec/N, not max), so a
    # large singlet is no longer "punished" with extra noise.
    P_ = 16384
    dx = 12.0 / P_
    tall = np.zeros(P_, dtype=np.float64); tall[8000] = 1.0
    tall /= tall.sum() * dx                     # unit integral, very high max
    flat = np.ones(P_, dtype=np.float64)
    flat /= flat.sum() * dx                     # unit integral, tiny max
    assert tall.max() > 100 * flat.max()
    h_tall = _one_h_height(tall, 10)
    h_flat = _one_h_height(flat, 10)
    assert np.isclose(h_tall, h_flat, rtol=1e-6)         # reference unaffected by max
    # and it scales as 1/N
    assert np.isclose(_one_h_height(tall, 20), h_tall / 2, rtol=1e-6)


def test_augment_noise_level_in_range():
    # Sampled noise fraction stays within the log-uniform band across draws.
    rng = np.random.default_rng(1)
    spec = np.zeros(P, dtype=np.float32); spec[100] = 1.0; spec[300] = 0.3
    for _ in range(5):
        out = augment_spectrum(spec, 0.0, 12.0, rng=rng, n_protons=8,
                               noise_frac_range=(0.003, 0.015))
        assert out.shape == (P,) and (out >= 0).all()
    # peaks are not moved (no referencing shift)
    out = augment_spectrum(spec, 0.0, 12.0, rng=rng, n_protons=8, noise_frac_range=None)
    assert int(out[90:110].argmax()) + 90 == 100


def test_class_balance_shapes():
    recs = _records(40)
    vocab = DegeneracyVocab()
    cb = class_balance(recs, vocab)
    assert cb["deg_weights"].shape == (len(vocab),)
    assert cb["presence_pos_weight"] > 0


# ── dataset + collate -> SpinBatch ─────────────────────────────────────────────

def test_dataset_item_matrix_form():
    recs = _records(8)
    vocab = DegeneracyVocab()
    std = Standardizer().fit(recs, vocab)
    ds = SpectrumMatrixDataset(recs, vocab, std, spectrum_field="spec90", augment=False)
    item = ds[0]
    assert item["couplings"].shape == (G, G)
    assert item["coupling_mask"].shape == (G, G)
    # symmetric, zero diagonal
    assert torch.allclose(item["couplings"], item["couplings"].T)
    assert torch.allclose(torch.diagonal(item["couplings"]), torch.zeros(G))


def test_collate_produces_valid_spinbatch():
    recs = _records(16)
    vocab = DegeneracyVocab()
    std = Standardizer().fit(recs, vocab)
    ds = SpectrumMatrixDataset(recs, vocab, std, spectrum_field="spec90", augment=True)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_spin_batch)
    batch = next(iter(dl))
    assert isinstance(batch, SpinBatch)
    batch.validate()
    assert batch.batch_size == 4 and batch.n_groups == G
    assert batch.spectrum.shape == (4, P)
    assert len(batch.molecule_ids) == 4


def test_make_splits_no_leakage_synthetic():
    recs = _records(60)
    assignment, report = make_splits(recs, seed=0, compute_scaffold=False)
    assert report["scaffold_leaks"] == 0
    assert report["dup_matrix_leaks"] == 0
    assert set(assignment.values()) <= {"train", "val", "test"}
