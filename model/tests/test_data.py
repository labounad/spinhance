"""
Data-layer tests: transforms/standardizer (torch-free) + dataset/collate -> SpinBatch.
Uses synthetic in-memory records (no data files, no RDKit: compute_scaffold=False).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from model.data.splits import canonical_order, make_splits
from model.data.standardization import DegeneracyVocab, Standardizer, class_balance
from model.data.transforms import encode_target, augment_spectrum
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
    out = augment_spectrum(spec, 0.0, 12.0, rng=rng)
    assert out.shape == (P,) and (out >= 0).all()


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
