"""
Tests for the typed data contracts (model.schemas) and the registry.
Tiny synthetic tensors only — no training, no data files.
"""
import pytest
import torch

from model.schemas import (
    SpinBatch, ModelOutput, LossOutput, RendererOutput, RunStatus, RunSummary,
)
from model.schemas import constants as C
from model.registry import Registry


B, G, P, Cls = 4, 8, 256, len(C.DEFAULT_DEG_VOCAB)


def _batch(device="cpu") -> SpinBatch:
    return SpinBatch(
        spectrum=torch.rand(B, P),
        spectrum_ref=torch.rand(B, P),
        shifts=torch.rand(B, G) * 10,
        couplings=torch.zeros(B, G, G),
        coupling_mask=torch.zeros(B, G, G),
        degeneracy_classes=torch.zeros(B, G, dtype=torch.long),
        degeneracy_values=torch.ones(B, G, dtype=torch.long),
        molecule_ids=[f"m{i}" for i in range(B)],
        smiles=["C"] * B,
    ).to(device)


# ── SpinBatch ──────────────────────────────────────────────────────────────────

def test_batch_validates():
    b = _batch().validate()
    assert b.batch_size == B and b.n_groups == G and len(b) == B


def test_batch_to_device_is_copy():
    b = _batch()
    b2 = b.to("cpu")
    assert b2 is not b
    assert b2.spectrum.device.type == "cpu"
    assert b2.molecule_ids == b.molecule_ids


def test_batch_validate_catches_bad_shape():
    b = _batch()
    b.shifts = torch.rand(B, G + 1)         # wrong group count
    with pytest.raises(AssertionError):
        b.validate()


# ── ModelOutput ──────────────────────────────────────────────────────────────

def test_output_matrix_form():
    out = ModelOutput(
        shifts=torch.rand(B, G),
        coupling_values=torch.rand(B, G, G),
        coupling_presence_logits=torch.rand(B, G, G),
        degeneracy_logits=torch.rand(B, G, Cls),
    ).validate()
    assert out.coupling_matrix().shape == (B, G, G)
    assert out.presence_matrix().shape == (B, G, G)


def test_output_edge_list_normalizes_to_matrix():
    E = G * (G - 1) // 2
    out = ModelOutput(
        shifts=torch.rand(B, G),
        coupling_values=torch.rand(B, E),                # edge-list form
        coupling_presence_logits=torch.rand(B, E),
        degeneracy_logits=torch.rand(B, G, Cls),
    )
    cm = out.coupling_matrix()
    assert cm.shape == (B, G, G)
    # symmetric with zero diagonal
    assert torch.allclose(cm, cm.transpose(1, 2))
    assert torch.allclose(torch.diagonal(cm, dim1=1, dim2=2), torch.zeros(B, G))


def test_output_validate_rejects_nonfinite():
    out = ModelOutput(
        shifts=torch.full((B, G), float("nan")),
        coupling_values=torch.rand(B, G, G),
        coupling_presence_logits=torch.rand(B, G, G),
        degeneracy_logits=torch.rand(B, G, Cls),
    )
    with pytest.raises(AssertionError):
        out.validate()


# ── LossOutput ─────────────────────────────────────────────────────────────────

def test_loss_output_scalar_and_backward():
    x = torch.rand(B, G, requires_grad=True)
    total = (x ** 2).mean()
    lo = LossOutput(total=total, components={"sq": total},
                    metrics={"sq": float(total.detach())}).validate()
    lo.total.backward()
    assert x.grad is not None


def test_loss_output_rejects_nonscalar():
    with pytest.raises(AssertionError):
        LossOutput(total=torch.rand(3)).validate()


def test_loss_output_rejects_tensor_metric():
    with pytest.raises(AssertionError):
        LossOutput(total=torch.tensor(0.0), metrics={"bad": torch.tensor(1.0)}).validate()


# ── RendererOutput ─────────────────────────────────────────────────────────────

def test_renderer_output_rendered_flag():
    r = RendererOutput(spectrum=torch.rand(B, P)).validate()
    assert r.rendered and not r.skipped
    s = RendererOutput(spectrum=None,
                       diagnostics={"rendered": False, "skip_reason": "too_big"})
    assert s.skipped and s.diagnostics["skip_reason"] == "too_big"


# ── diagnostics payloads ─────────────────────────────────────────────────────

def test_run_status_to_dict_merges_extra():
    st = RunStatus(state="running", run_id="r", epoch=1, epochs=10, stage="1",
                   global_step=5, best_score=0.5, best_epoch=1, device="cpu",
                   last_update_time=0.0, extra={"custom": 7})
    d = st.to_dict()
    assert d["state"] == "running" and d["custom"] == 7 and "extra" not in d


# ── Registry ─────────────────────────────────────────────────────────────────

def test_registry_register_build_and_errors():
    reg = Registry("widget")
    @reg.register("a")
    class A:
        def __init__(self, k=1):
            self.k = k
    assert "a" in reg and reg.available() == ["a"] and len(reg) == 1
    assert reg.build("a", k=3).k == 3
    with pytest.raises(KeyError):
        reg.register("a")(A)            # duplicate
    with pytest.raises(KeyError):
        reg.build("missing")            # unknown


def test_layer_registries_exist():
    from model.architectures import ARCHITECTURES
    from model.losses import LOSSES
    from model.renderers import RENDERERS
    for reg in (ARCHITECTURES, LOSSES, RENDERERS):
        assert hasattr(reg, "build") and hasattr(reg, "available")
