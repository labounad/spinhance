"""
Exact renderer (ported) + spectral metrics tests:
- torch engine matches the numpy composite oracle (forward correlation),
- exact_no_grad RendererOutput: rendered flag, no grad, cost diagnostics,
- cost guard skips oversized systems with a reason,
- spectral metrics: identical->0/1, shifted->>0.
Small point grids; CPU only.
"""
import numpy as np
import pytest
import torch

from model.renderers import RENDERERS, build_renderer
from model.renderers import _composite as C
from model.evaluation.spectral_metrics import wasserstein1, cosine_similarity

P = 4096
FIELD = 90.0


def _tiny_system():
    # 3 groups, sparse coupling, low degeneracy -> small Hilbert space
    shifts = np.array([7.2, 3.5, 1.2])
    cpl = np.zeros((3, 3))
    cpl[0, 1] = cpl[1, 0] = 7.0
    deg = [1, 2, 3]
    return shifts, cpl, deg


def test_both_registered():
    assert "exact_no_grad" in RENDERERS and "exact_autograd_experimental" in RENDERERS


def test_torch_matches_numpy_oracle():
    shifts, cpl, deg = _tiny_system()
    _, spec_np = C.simulate(shifts, cpl, deg, FIELD, points=P)
    r = build_renderer("exact_no_grad", points=P)
    out = r.render(shifts, cpl, deg, FIELD).validate()
    spec_t = out.spectrum[0].cpu().numpy()
    corr = np.corrcoef(spec_np, spec_t)[0, 1]
    # ~0.997 expected: same transitions, slightly different broadening kernels
    # (numpy oracle dense-convolve vs torch FFT). Proves spectral equivalence.
    assert corr > 0.99, f"torch vs numpy oracle corr={corr:.5f}"


def test_no_grad_render_flags_and_cost():
    shifts, cpl, deg = _tiny_system()
    out = build_renderer("exact_no_grad", points=P).render(shifts, cpl, deg, FIELD)
    assert out.rendered and not out.skipped
    assert out.diagnostics["grad"] is False
    assert out.diagnostics["max_block_dim"] == C.max_block_dim(deg)
    assert out.spectrum.shape == (1, P)
    assert torch.isfinite(out.spectrum).all()


def test_cost_guard_skips_oversized():
    shifts, cpl, deg = _tiny_system()
    r = build_renderer("exact_no_grad", points=P, max_block=4)  # absurdly low limit
    out = r.render(shifts, cpl, deg, FIELD)
    assert out.skipped and out.spectrum is None
    assert out.diagnostics["skip_reason"] == "max_block_exceeded"
    assert out.diagnostics["max_block_dim"] > 4


def test_autograd_mode_flows_gradient_tiny():
    # tiny 2-group system, differentiable mode
    shifts = torch.tensor([5.0, 2.0], dtype=torch.float64, requires_grad=True)
    cpl = torch.zeros(2, 2, dtype=torch.float64)
    r = build_renderer("exact_autograd_experimental", points=1024, max_block=64)
    out = r.render(shifts, cpl, [1, 1], FIELD)
    assert out.rendered
    out.spectrum.sum().backward()
    assert shifts.grad is not None and torch.isfinite(shifts.grad).all()


def test_spectral_metrics_behaviour():
    a = torch.zeros(1, 256); a[0, 100] = 1.0
    b = torch.zeros(1, 256); b[0, 120] = 1.0
    assert wasserstein1(a, a).item() < 1e-9
    assert wasserstein1(a, b).item() > 0
    assert cosine_similarity(a, a).item() == pytest.approx(1.0, abs=1e-6)
    assert cosine_similarity(a, b).item() < 1e-6
