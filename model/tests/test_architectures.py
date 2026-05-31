"""
Architecture contract tests: construction from registry, forward on a synthetic
SpinBatch and a raw tensor, ModelOutput schema validation, finite + deterministic
shapes. CPU only; tiny spectra.
"""
import pytest
import torch

from model.architectures import ARCHITECTURES, build_architecture
from model.schemas import ModelOutput, SpinBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB

B, G, P, C = 3, 8, 1024, len(DEFAULT_DEG_VOCAB)
NAMES = ["resnet1d", "resnet1d_attention_pool"]


def _batch():
    return SpinBatch(
        spectrum=torch.rand(B, P),
        spectrum_ref=torch.rand(B, P),
        shifts=torch.rand(B, G), couplings=torch.zeros(B, G, G),
        coupling_mask=torch.zeros(B, G, G),
        degeneracy_classes=torch.zeros(B, G, dtype=torch.long),
        degeneracy_values=torch.ones(B, G, dtype=torch.long),
        molecule_ids=[f"m{i}" for i in range(B)],
    )


def test_both_registered():
    for n in NAMES:
        assert n in ARCHITECTURES


@pytest.mark.parametrize("name", NAMES)
def test_forward_from_spinbatch(name):
    model = build_architecture(name, size="tiny").eval()
    out = model(_batch())
    assert isinstance(out, ModelOutput)
    out.validate(G)
    assert out.shifts.shape == (B, G)
    assert out.coupling_matrix().shape == (B, G, G)
    assert out.degeneracy_logits.shape == (B, G, C)


@pytest.mark.parametrize("name", NAMES)
def test_forward_from_raw_tensor(name):
    model = build_architecture(name, size="tiny").eval()
    out = model(torch.rand(B, P))
    out.validate(G)


@pytest.mark.parametrize("name", NAMES)
def test_outputs_finite(name):
    model = build_architecture(name, size="tiny").eval()
    out = model(torch.rand(B, P))
    for t in (out.shifts, out.coupling_matrix(), out.presence_matrix(), out.degeneracy_logits):
        assert torch.isfinite(t).all()


def test_deterministic_shapes_across_batch_sizes():
    model = build_architecture("resnet1d", size="tiny").eval()
    for b in (1, 5):
        out = model(torch.rand(b, P))
        assert out.shifts.shape == (b, G)


def test_backward_flows():
    model = build_architecture("resnet1d_attention_pool", size="tiny")
    out = model(torch.rand(B, P))
    loss = out.shifts.pow(2).mean() + out.coupling_matrix().pow(2).mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_size_presets_param_counts_increase():
    counts = []
    for size in ("tiny", "small", "medium"):
        m = build_architecture("resnet1d", size=size)
        counts.append(m.n_params)
    assert counts[0] < counts[1] < counts[2]
