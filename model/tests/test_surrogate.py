"""
SurrogateRenderer (Branch 5) ground-breaking smoke tests: shape contract,
finite + unit-integral output, gradient flow to shifts/couplings/params, and
field conditioning actually changing the spectrum. Tiny grid, CPU.
"""
import torch

from model.renderers import RENDERERS, build_renderer
from model.renderers.surrogate import SurrogateRenderer

B, G, P = 4, 8, 2048


def _inputs(requires_grad=False):
    shifts = torch.rand(B, G) * 8 + 1
    cpl = torch.zeros(B, G, G)
    iu = torch.triu_indices(G, G, 1)
    j = torch.rand(B, iu.shape[1]) * 8
    cpl[:, iu[0], iu[1]] = j; cpl[:, iu[1], iu[0]] = j
    deg = torch.randint(1, 4, (B, G)).float()
    if requires_grad:
        shifts = shifts.requires_grad_(True)
        cpl = cpl.requires_grad_(True)
    return shifts, cpl, deg


def _model():
    return SurrogateRenderer(dim=32, depth=2, heads=2, sticks_per_group=16, points=P)


def test_registered():
    assert "surrogate" in RENDERERS
    assert isinstance(build_renderer("surrogate", points=P), SurrogateRenderer)


def test_shape_finite_unit_integral():
    m = _model().eval()
    shifts, cpl, deg = _inputs()
    spec = m(shifts, cpl, deg, 90.0)
    assert spec.shape == (B, P)
    assert torch.isfinite(spec).all() and (spec >= 0).all()
    dx = 12.0 / P
    integral = spec.sum(-1) * dx
    assert torch.allclose(integral, torch.ones(B), atol=1e-3), integral


def test_gradient_flows_to_matrix_and_params():
    m = _model()
    shifts, cpl, deg = _inputs(requires_grad=True)
    spec = m(shifts, cpl, deg, 90.0)
    # a dummy spectral objective
    target = torch.rand(B, P)
    loss = (spec - target).pow(2).mean()
    loss.backward()
    assert shifts.grad is not None and torch.isfinite(shifts.grad).all()
    assert cpl.grad is not None
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_field_conditioning_changes_output():
    m = _model().eval()
    shifts, cpl, deg = _inputs()
    s90 = m(shifts, cpl, deg, 90.0)
    s600 = m(shifts, cpl, deg, 600.0)
    # different field -> different rendered spectrum (peaks tighten / second order weakens)
    assert (s90 - s600).abs().max() > 1e-4


def test_peaks_sit_near_group_shifts():
    """With couplings off, each group's sticks should broaden to mass near its shift."""
    m = _model().eval()
    shifts = torch.tensor([[7.0, 5.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8]])
    cpl = torch.zeros(1, G, G)
    deg = torch.ones(1, G)
    spec = m(shifts, cpl, deg, 90.0)[0]
    ppm = torch.linspace(0, 12, P)
    # spectral mass should be concentrated in [0.5, 7.5] ppm (where the shifts are)
    in_band = spec[(ppm >= 0.5) & (ppm <= 7.5)].sum() * (12.0 / P)
    assert in_band > 0.9, f"only {in_band:.2f} of integral near the group shifts"
