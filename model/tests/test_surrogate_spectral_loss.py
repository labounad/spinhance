"""
SurrogateSpectralLoss (Branch 6) tests: the frozen surrogate renders the
predicted matrix and the W1+cosine consistency term flows gradients back to
every prediction head while leaving the surrogate frozen. Also checks the
composite ramp gates the term by epoch. Tiny surrogate + grid, CPU.
"""
import torch

from model.losses import build_composite
from model.losses.surrogate_spectral_loss import SurrogateSpectralLoss
from model.renderers.surrogate import SurrogateRenderer
from model.schemas import ModelOutput, SpinBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB

B, G, P = 4, 8, 1024
C = len(DEFAULT_DEG_VOCAB)


def _tiny_surrogate_ckpt(tmp_path):
    """Save a small surrogate checkpoint in the trainer's {model, cfg} format."""
    m = SurrogateRenderer(dim=32, depth=2, heads=2, sticks_per_group=16, points=P)
    cfg = {"model": {"name": "surrogate", "dim": 32, "depth": 2, "heads": 2,
                     "sticks_per_group": 16, "points": P}}
    ckpt = tmp_path / "surrogate.pt"
    torch.save({"model": m.state_dict(), "cfg": cfg, "epoch": 0, "metrics": {}}, ckpt)
    return str(ckpt)


def _output(requires_grad=True):
    g = requires_grad
    return ModelOutput(
        shifts=torch.randn(B, G, requires_grad=g),
        coupling_values=torch.randn(B, G, G, requires_grad=g),
        coupling_presence_logits=torch.randn(B, G, G, requires_grad=g),
        degeneracy_logits=torch.randn(B, G, C, requires_grad=g),
    )


def _batch():
    spec = torch.rand(B, P)
    spec = spec / (spec.sum(-1, keepdim=True) * (12.0 / P))   # unit integral
    return SpinBatch(spectrum=spec, spectrum_ref=spec,
                     shifts=torch.randn(B, G), couplings=torch.zeros(B, G, G),
                     coupling_mask=torch.zeros(B, G, G),
                     degeneracy_classes=torch.zeros(B, G, dtype=torch.long),
                     degeneracy_values=torch.ones(B, G),
                     molecule_ids=[f"m{i}" for i in range(B)])


def _loss(tmp_path):
    return SurrogateSpectralLoss(checkpoint=_tiny_surrogate_ckpt(tmp_path), field=90,
                                 shift_mean=5.0, shift_std=2.0, j_mean=7.0, j_std=4.0)


def test_frozen_and_finite(tmp_path):
    loss = _loss(tmp_path)
    assert all(not p.requires_grad for p in loss.surrogate.parameters())
    lo = loss(_output(requires_grad=False), _batch()).validate()
    assert torch.isfinite(lo.total)
    assert 0.0 <= lo.metrics["cosine"] <= 1.0 + 1e-5
    assert lo.metrics["field"] == 90.0


def test_grad_flows_to_all_heads(tmp_path):
    loss = _loss(tmp_path)
    out = _output(requires_grad=True)
    loss(out, _batch()).total.backward()
    for name, t in (("shifts", out.shifts), ("couplings", out.coupling_values),
                    ("presence", out.coupling_presence_logits),
                    ("degeneracy", out.degeneracy_logits)):
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
    assert float(out.coupling_presence_logits.grad.abs().sum()) > 0
    assert float(out.degeneracy_logits.grad.abs().sum()) > 0
    # the surrogate stays frozen — no grad on its params
    assert all(p.grad is None for p in loss.surrogate.parameters())


def test_runs_under_autocast(tmp_path):
    """Regression: the matrix trainer runs under bf16/fp16 autocast; the frozen
    surrogate's FFT broadening must still run (it crashed with an index_add
    BFloat16-vs-Float dtype mismatch before the float32/disable-autocast fix)."""
    loss = _loss(tmp_path)
    out = _output(requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        lo = loss(out, _batch())
    assert torch.isfinite(lo.total)
    lo.total.backward()
    assert out.shifts.grad is not None and torch.isfinite(out.shifts.grad).all()


def test_diagonal_zeroed_and_symmetric(tmp_path):
    loss = _loss(tmp_path)
    _, couplings, _ = loss._physical_matrix(_output(requires_grad=False))
    diag = torch.diagonal(couplings, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6)
    assert torch.allclose(couplings, couplings.transpose(-1, -2), atol=1e-6)


def test_composite_ramp_gates_by_epoch(tmp_path):
    """start_epoch/ramp_epochs ramp the spectral term in linearly; matrix anchor stays on."""
    terms = [{"name": "matrix", "weight": 1.0},
             {"name": "surrogate_spectral", "weight": 0.4, "start_epoch": 2,
              "ramp_epochs": 2, "checkpoint": _tiny_surrogate_ckpt(tmp_path), "field": 90}]
    comp = build_composite(terms, shift_mean=5.0, shift_std=2.0, j_mean=7.0, j_std=4.0)
    out, batch = _output(requires_grad=False), _batch()

    comp.set_epoch(0)
    m0 = comp(out, batch).metrics
    assert m0["weight/surrogate_spectral"] == 0.0          # before start_epoch
    assert m0["weight/matrix"] == 1.0

    comp.set_epoch(2)
    assert comp(out, batch).metrics["weight/surrogate_spectral"] == 0.2   # frac 1/2 * 0.4

    comp.set_epoch(5)
    assert comp(out, batch).metrics["weight/surrogate_spectral"] == 0.4   # fully ramped


def test_composite_trapezoid_ramps_up_then_down(tmp_path):
    """Trapezoid: ramp in, hold, ramp back out to end_weight."""
    terms = [{"name": "matrix", "weight": 1.0},
             {"name": "surrogate_spectral", "weight": 0.3, "start_epoch": 4,
              "ramp_epochs": 2, "decay_start_epoch": 8, "decay_epochs": 2,
              "end_weight": 0.0, "checkpoint": _tiny_surrogate_ckpt(tmp_path), "field": 90}]
    comp = build_composite(terms, shift_mean=5.0, shift_std=2.0, j_mean=7.0, j_std=4.0)
    t = comp.terms[1]
    sched = {}
    for e in range(12):
        comp.set_epoch(e)
        sched[e] = round(comp._weight(t), 4)
    assert sched[3] == 0.0                       # before start
    assert sched[5] == 0.3                        # ramped up (frac 2/2)
    assert sched[6] == 0.3 and sched[7] == 0.3    # hold
    assert sched[9] == 0.0                         # decayed to end_weight (frac 2/2)
    assert sched[11] == 0.0                         # stays at floor
    assert 0.0 < sched[8] < 0.3                     # mid-decay strictly between
