"""
Matrix-loss tests: perfect prediction ~ 0, bad prediction larger, gradient flow,
finite, class-weighting accepted, and the composite ramp schedule.
Tiny synthetic ModelOutput/SpinBatch only.
"""
import math

import pytest
import torch

from model.losses import build_loss, build_composite, LOSSES
from model.losses.matrix_loss import MatrixLoss
from model.schemas import LossOutput, ModelOutput, SpinBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB

B, G, C = 4, 8, len(DEFAULT_DEG_VOCAB)
E = G * (G - 1) // 2


def _targets(seed=0):
    rng = torch.Generator().manual_seed(seed)
    shifts = torch.randn(B, G, generator=rng)
    cmat = torch.zeros(B, G, G)
    mask = torch.zeros(B, G, G)
    iu = torch.triu_indices(G, G, 1)
    pres = (torch.rand(B, E, generator=rng) > 0.5).float()
    jval = torch.randn(B, E, generator=rng) * pres
    cmat[:, iu[0], iu[1]] = jval; cmat[:, iu[1], iu[0]] = jval
    mask[:, iu[0], iu[1]] = pres; mask[:, iu[1], iu[0]] = pres
    deg_cls = torch.randint(0, C, (B, G), generator=rng)
    batch = SpinBatch(
        spectrum=torch.zeros(B, 4), spectrum_ref=None,
        shifts=shifts, couplings=cmat, coupling_mask=mask,
        degeneracy_classes=deg_cls, degeneracy_values=torch.ones(B, G, dtype=torch.long),
        molecule_ids=[f"m{i}" for i in range(B)],
    )
    return batch, iu, pres, jval, deg_cls


def _perfect_output(batch, iu, pres, jval, deg_cls):
    # presence logits: large +/- to match the mask; degeneracy one-hot logits
    pres_logits = torch.where(pres > 0.5, 20.0, -20.0)
    deg_logits = torch.full((B, G, C), -20.0)
    deg_logits.scatter_(2, deg_cls.unsqueeze(-1), 20.0)
    return ModelOutput(
        shifts=batch.shifts.clone(),
        coupling_values=jval.clone(),                 # edge-list
        coupling_presence_logits=pres_logits,
        degeneracy_logits=deg_logits,
    )


def test_loss_registered():
    assert "matrix" in LOSSES and "composite" in LOSSES


def test_perfect_prediction_near_zero():
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    lo = MatrixLoss()(out, batch).validate()
    assert lo.total.item() < 1e-3, lo.metrics


def test_bad_prediction_larger_than_perfect():
    batch, iu, pres, jval, deg_cls = _targets()
    good = MatrixLoss()(_perfect_output(batch, iu, pres, jval, deg_cls), batch).total.item()
    bad_out = ModelOutput(
        shifts=batch.shifts + 5.0,
        coupling_values=jval + 3.0,
        coupling_presence_logits=torch.where(pres > 0.5, -20.0, 20.0),  # inverted
        degeneracy_logits=torch.randn(B, G, C),
    )
    bad = MatrixLoss()(bad_out, batch).total.item()
    assert bad > good + 1.0


def test_gradient_flows_to_output():
    batch, iu, pres, jval, deg_cls = _targets()
    shifts = batch.shifts.clone().requires_grad_(True)
    cvals = jval.clone().requires_grad_(True)
    out = ModelOutput(
        shifts=shifts, coupling_values=cvals,
        coupling_presence_logits=torch.zeros(B, E, requires_grad=True),
        degeneracy_logits=torch.zeros(B, G, C, requires_grad=True),
    )
    lo = MatrixLoss()(out, batch).validate()
    lo.total.backward()
    assert shifts.grad is not None and torch.isfinite(shifts.grad).all()


def test_finite_and_scalar():
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    lo = MatrixLoss()(out, batch)
    assert lo.total.dim() == 0 and math.isfinite(lo.total.item())


def test_focal_runs_finite_grad_and_scalar():
    batch, iu, pres, jval, deg_cls = _targets()
    out = ModelOutput(
        shifts=batch.shifts.clone().requires_grad_(True),
        coupling_values=jval.clone().requires_grad_(True),
        coupling_presence_logits=torch.zeros(B, E, requires_grad=True),
        degeneracy_logits=torch.randn(B, G, C, requires_grad=True),
    )
    lo = MatrixLoss(deg_focal_gamma=2.0, presence_focal_gamma=2.0)(out, batch).validate()
    assert lo.total.dim() == 0 and math.isfinite(lo.total.item())
    lo.total.backward()
    assert torch.isfinite(out.degeneracy_logits.grad).all()


def test_focal_gamma_zero_matches_plain():
    """gamma=0 must take the focal-free path and equal the default loss exactly."""
    batch, iu, pres, jval, deg_cls = _targets(seed=3)
    out = ModelOutput(
        shifts=batch.shifts + 0.3, coupling_values=jval + 0.2,
        coupling_presence_logits=torch.randn(B, E),
        degeneracy_logits=torch.randn(B, G, C),
    )
    plain = MatrixLoss()(out, batch)
    z = MatrixLoss(deg_focal_gamma=0.0, presence_focal_gamma=0.0)(out, batch)
    assert abs(plain.metrics["deg"] - z.metrics["deg"]) < 1e-6
    assert abs(plain.metrics["presence"] - z.metrics["presence"]) < 1e-6


def test_focal_downweights_easy_examples():
    """On a mostly-correct prediction, focal shrinks the deg+presence terms vs plain."""
    batch, iu, pres, jval, deg_cls = _targets(seed=7)
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    # soften the (otherwise saturated) logits so 'easy but not perfect' applies
    out.degeneracy_logits = out.degeneracy_logits.clamp(-3.0, 3.0)
    out.coupling_presence_logits = out.coupling_presence_logits.clamp(-3.0, 3.0)
    plain = MatrixLoss()(out, batch)
    foc = MatrixLoss(deg_focal_gamma=2.0, presence_focal_gamma=2.0)(out, batch)
    assert foc.metrics["deg"] < plain.metrics["deg"]
    assert foc.metrics["presence"] < plain.metrics["presence"]


def test_class_weights_accepted():
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    loss = MatrixLoss(deg_class_weight=[1.0] * C, presence_pos_weight=2.0)
    lo = loss(out, batch).validate()
    assert math.isfinite(lo.total.item())


# ── composite ──────────────────────────────────────────────────────────────────

def test_composite_single_term_matches_matrix():
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    comp = build_composite([{"name": "matrix", "weight": 1.0}])
    direct = MatrixLoss()(out, batch).total.item()
    assert abs(comp(out, batch).total.item() - direct) < 1e-6


def test_composite_ramp_gates_by_epoch():
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    comp = build_composite([
        {"name": "matrix", "weight": 1.0},
        {"name": "matrix", "weight": 1.0, "start_epoch": 5, "ramp_epochs": 5},
    ])
    comp.set_epoch(0)
    # term 0 always active (weight 1.0); term 1 gated off until start_epoch
    assert comp._weight(comp.terms[0]) == 1.0
    assert comp._weight(comp.terms[1]) == 0.0
    comp.set_epoch(5)
    assert comp._weight(comp.terms[1]) == pytest.approx(1.0 / 5)
    comp.set_epoch(9)
    assert comp._weight(comp.terms[1]) == pytest.approx(1.0)


def test_composite_shared_kwargs_filtered():
    # shared class-balance kwargs should reach MatrixLoss without error
    comp = build_composite([{"name": "matrix", "weight": 1.0}],
                           deg_class_weight=[1.0] * C, presence_pos_weight=2.0)
    assert isinstance(comp.terms[0].loss, MatrixLoss)


# ── symmetry-aware coupling term (sym_jmag) ─────────────────────────────────────

def _sym_batch(Gs=4):
    """Batch where groups 0,1 are chemically equivalent (swappable): couplings are
    asymmetric under the (0,1) swap, and sym_perms carries identity + that swap."""
    Bs = 2
    iu = torch.triu_indices(Gs, Gs, 1)
    cmat = torch.zeros(Bs, Gs, Gs)
    mask = torch.zeros(Bs, Gs, Gs)
    for (i, j), v in {(0, 2): 5.0, (1, 3): 7.0}.items():
        cmat[:, i, j] = cmat[:, j, i] = v
        mask[:, i, j] = mask[:, j, i] = 1.0
    sigma = torch.tensor([1, 0] + list(range(2, Gs)))          # swap 0<->1
    perms = torch.stack([torch.arange(Gs), sigma]).long()      # (2, G)
    sym_perms = perms[None].repeat(Bs, 1, 1)                    # (B, 2, G)
    deg_cls = torch.zeros(Bs, Gs, dtype=torch.long)
    batch = SpinBatch(
        spectrum=torch.zeros(Bs, 4), spectrum_ref=None,
        shifts=torch.zeros(Bs, Gs), couplings=cmat, coupling_mask=mask,
        degeneracy_classes=deg_cls, degeneracy_values=torch.ones(Bs, Gs, dtype=torch.long),
        molecule_ids=[f"m{i}" for i in range(Bs)], sym_perms=sym_perms,
    )
    return batch, cmat, mask, iu, sigma, deg_cls


def _output_for(batch, cmat, mask, iu, deg_cls, coupling_matrix):
    Bs, Gs, _ = cmat.shape
    pres = mask[:, iu[0], iu[1]]
    deg_logits = torch.full((Bs, Gs, len(DEFAULT_DEG_VOCAB)), -20.0)
    deg_logits.scatter_(2, deg_cls.unsqueeze(-1), 20.0)
    return ModelOutput(
        shifts=batch.shifts.clone(),
        coupling_values=coupling_matrix[:, iu[0], iu[1]],
        coupling_presence_logits=torch.where(pres > 0.5, 20.0, -20.0),
        degeneracy_logits=deg_logits,
    )


def test_sym_jmag_forgives_equivalent_relabeling():
    """A prediction = target with the equivalent groups swapped is the SAME spectrum:
    sym_jmag must score its J term ~0, while the plain element-wise term penalizes it."""
    batch, cmat, mask, iu, sigma, deg_cls = _sym_batch()
    swapped = cmat[:, sigma][:, :, sigma]                       # relabeled prediction
    out = _output_for(batch, cmat, mask, iu, deg_cls, swapped)
    j_sym = MatrixLoss(sym_jmag=True)(out, batch).metrics["jmag"]
    j_plain = MatrixLoss(sym_jmag=False)(out, batch).metrics["jmag"]
    assert j_sym < 1e-4, j_sym
    assert j_plain > 1.0, j_plain


def test_sym_jmag_perfect_prediction_near_zero():
    batch, cmat, mask, iu, sigma, deg_cls = _sym_batch()
    out = _output_for(batch, cmat, mask, iu, deg_cls, cmat)
    assert MatrixLoss(sym_jmag=True)(out, batch).metrics["jmag"] < 1e-4


def test_sym_jmag_no_perms_falls_back_to_plain():
    """sym_jmag=True but a batch without sym_perms must no-op to the plain J term."""
    batch, iu, pres, jval, deg_cls = _targets()
    assert getattr(batch, "sym_perms", None) is None
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    sym = MatrixLoss(sym_jmag=True)(out, batch).metrics["jmag"]
    plain = MatrixLoss(sym_jmag=False)(out, batch).metrics["jmag"]
    assert abs(sym - plain) < 1e-6


def test_sym_jmag_grad_flows_and_finite():
    batch, cmat, mask, iu, sigma, deg_cls = _sym_batch()
    swapped = cmat[:, sigma][:, :, sigma]
    cvals = swapped[:, iu[0], iu[1]].clone().requires_grad_(True)
    Bs, Gs, _ = cmat.shape
    deg_logits = torch.full((Bs, Gs, C), -20.0); deg_logits.scatter_(2, deg_cls.unsqueeze(-1), 20.0)
    out = ModelOutput(
        shifts=batch.shifts.clone(), coupling_values=cvals,
        coupling_presence_logits=torch.where(mask[:, iu[0], iu[1]] > 0.5, 20.0, -20.0),
        degeneracy_logits=deg_logits,
    )
    lo = MatrixLoss(sym_jmag=True)(out, batch).validate()
    lo.total.backward()
    assert math.isfinite(lo.total.item())
    assert cvals.grad is not None and torch.isfinite(cvals.grad).all()
