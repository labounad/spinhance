"""
Support-region tokenizer (Phase 2, IDEAS D/E/H) tests: extraction on synthetic
spectra (correct region count / mask / integration), empty + truncation edge
cases, the collate path assembling a RegionTokenBatch, and that the extractor's
feature dim matches the spingraph_decoder's region branch (FEAT_DIM == 80).
"""
import numpy as np
import torch

from model.data.collate import collate_spin_batch
from model.data.regions import FEAT_DIM, extract_support_regions
from model.architectures import build_architecture
from model.schemas.batch import RegionTokenBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB, N_POINTS

C = len(DEFAULT_DEG_VOCAB)


def _peaks(centers_ppm, P=4096, ppm_to=12.0, width_pts=12, height=1.0):
    """Synthetic unit-integral-ish spectrum with Gaussian peaks at given ppm."""
    x = np.arange(P)
    y = np.zeros(P, dtype=np.float64)
    for c in centers_ppm:
        mu = c / ppm_to * P
        y += height * np.exp(-0.5 * ((x - mu) / width_pts) ** 2)
    return y


def test_extract_two_peaks():
    spec = _peaks([3.0, 7.0], P=4096)
    feats, mask = extract_support_regions(spec, 0.0, 12.0, max_regions=48)
    assert feats.shape == (48, FEAT_DIM)
    assert mask.sum() == 2                                  # exactly two regions
    # region centers (feature 0 = center_ppm/12) recover ~3 and ~7 ppm
    centers = sorted(feats[mask > 0, 0] * 12.0)
    assert abs(centers[0] - 3.0) < 0.3 and abs(centers[1] - 7.0) < 0.3
    # raw integral (feature 4) is positive for real regions
    assert (feats[mask > 0, 4] > 0).all()


def test_empty_spectrum_all_pad():
    feats, mask = extract_support_regions(np.zeros(4096), 0.0, 12.0)
    assert mask.sum() == 0 and np.isfinite(feats).all()


def test_truncates_to_max_regions():
    spec = _peaks(list(np.linspace(1.0, 11.0, 30)), P=8192, width_pts=6)
    feats, mask = extract_support_regions(spec, 0.0, 12.0, max_regions=8)
    assert mask.sum() == 8                                  # capped


def _sample(P=64, regions=True):
    G = 8
    s = {
        "spectrum": torch.rand(P), "spectrum_ref": torch.rand(P),
        "shifts": torch.randn(G), "couplings": torch.zeros(G, G),
        "coupling_mask": torch.zeros(G, G),
        "degeneracy_classes": torch.zeros(G, dtype=torch.long),
        "degeneracy_values": torch.ones(G, dtype=torch.long),
        "mol_id": "m", "smiles": None, "bucket_key": (1,),
    }
    if regions:
        s["region_features"] = torch.zeros(48, FEAT_DIM)
        s["region_mask"] = torch.ones(48)
    return s


def test_collate_builds_region_batch():
    batch = collate_spin_batch([_sample(), _sample()])
    assert isinstance(batch.region_tokens, RegionTokenBatch)
    assert batch.region_tokens.features.shape == (2, 48, FEAT_DIM)
    assert batch.region_tokens.mask.shape == (2, 48)
    # and None when region features are absent
    assert collate_spin_batch([_sample(regions=False), _sample(regions=False)]).region_tokens is None


def test_model_consumes_extracted_regions():
    """End-to-end: extracted features feed the spingraph_decoder region branch."""
    spec = _peaks([2.5, 5.0, 8.5], P=N_POINTS)
    feats, mask = extract_support_regions(spec, 0.0, 12.0, max_regions=48)
    m = build_architecture("spingraph_decoder", n_deg_classes=C, size="tiny",
                           dim=32, enc_layers=1, dec_layers=1, n_heads=2,
                           region_feat_dim=FEAT_DIM).eval()
    rt = RegionTokenBatch(features=torch.from_numpy(feats)[None],
                          mask=torch.from_numpy(mask)[None])
    from model.schemas import SpinBatch
    G = 8
    batch = SpinBatch(spectrum=torch.from_numpy(spec.astype("float32"))[None],
                      spectrum_ref=torch.from_numpy(spec.astype("float32"))[None],
                      shifts=torch.zeros(1, G), couplings=torch.zeros(1, G, G),
                      coupling_mask=torch.zeros(1, G, G),
                      degeneracy_classes=torch.zeros(1, G, dtype=torch.long),
                      degeneracy_values=torch.ones(1, G, dtype=torch.long),
                      molecule_ids=["m"], region_tokens=rt)
    out = m(batch).validate(n_groups=G)
    assert out.shifts.shape == (1, G)
