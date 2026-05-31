"""
Hungarian graph loss: registration, perfect-aligned == matrix loss, permutation
invariance (permuted-perfect ~ 0), gradient flow, and hungarian <= matrix on a
mis-ordered prediction. Tiny synthetic tensors only.
"""
import torch

from model.losses import build_loss, LOSSES
from model.losses.matrix_loss import MatrixLoss
from model.losses.hungarian_loss import HungarianGraphLoss
from model.schemas import ModelOutput, SpinBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB

B, G, C = 4, 8, len(DEFAULT_DEG_VOCAB)
E = G * (G - 1) // 2


def _targets(seed=0):
    g = torch.Generator().manual_seed(seed)
    # distinct shifts so the optimal assignment is unambiguous
    shifts = torch.sort(torch.rand(B, G, generator=g) * 8 + 1, dim=1, descending=True).values
    iu = torch.triu_indices(G, G, 1)
    pres = (torch.rand(B, E, generator=g) > 0.5).float()
    jval = torch.randn(B, E, generator=g) * pres
    cmat = torch.zeros(B, G, G); mask = torch.zeros(B, G, G)
    cmat[:, iu[0], iu[1]] = jval; cmat[:, iu[1], iu[0]] = jval
    mask[:, iu[0], iu[1]] = pres; mask[:, iu[1], iu[0]] = pres
    deg_cls = torch.randint(0, C, (B, G), generator=g)
    batch = SpinBatch(
        spectrum=torch.zeros(B, 4), spectrum_ref=None,
        shifts=shifts, couplings=cmat, coupling_mask=mask,
        degeneracy_classes=deg_cls, degeneracy_values=torch.ones(B, G, dtype=torch.long),
        molecule_ids=[f"m{i}" for i in range(B)],
    )
    return batch, iu, pres, jval, deg_cls


def _perfect_output(batch, iu, pres, jval, deg_cls):
    pres_logits = torch.where(pres > 0.5, 20.0, -20.0)
    deg_logits = torch.full((B, G, C), -20.0)
    deg_logits.scatter_(2, deg_cls.unsqueeze(-1), 20.0)
    return ModelOutput(shifts=batch.shifts.clone(), coupling_values=jval.clone(),
                       coupling_presence_logits=pres_logits, degeneracy_logits=deg_logits)


def _permute_batch(batch, perm):
    """Return a new batch with target groups permuted by perm (B,G)."""
    bi = torch.arange(B)
    b3 = bi[:, None, None]
    return SpinBatch(
        spectrum=batch.spectrum, spectrum_ref=None,
        shifts=batch.shifts[bi[:, None], perm],
        couplings=batch.couplings[b3, perm[:, :, None], perm[:, None, :]],
        coupling_mask=batch.coupling_mask[b3, perm[:, :, None], perm[:, None, :]],
        degeneracy_classes=batch.degeneracy_classes[bi[:, None], perm],
        degeneracy_values=batch.degeneracy_values[bi[:, None], perm],
        molecule_ids=batch.molecule_ids,
    )


def test_registered():
    assert "hungarian" in LOSSES


def test_perfect_aligned_matches_matrix_loss():
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    h = HungarianGraphLoss()(out, batch).validate().total.item()
    m = MatrixLoss()(out, batch).total.item()
    assert h < 1e-3 and abs(h - m) < 1e-4


def test_permutation_invariance():
    """A perfect prediction stays ~0 even when the TARGET groups are permuted."""
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    g = torch.Generator().manual_seed(3)
    perm = torch.stack([torch.randperm(G, generator=g) for _ in range(B)])
    permuted = _permute_batch(batch, perm)
    h = HungarianGraphLoss()(out, permuted).total.item()
    assert h < 1e-3, h


def test_hungarian_leq_matrix_on_misordered():
    """When the prediction is correct up to a permutation, Hungarian loss should be
    much lower than the canonical matrix loss (which sees it as all-wrong)."""
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    g = torch.Generator().manual_seed(5)
    perm = torch.stack([torch.randperm(G, generator=g) for _ in range(B)])
    permuted = _permute_batch(batch, perm)        # target reordered; pred not
    h = HungarianGraphLoss()(out, permuted).total.item()
    m = MatrixLoss()(out, permuted).total.item()
    assert h <= m + 1e-6
    assert h < 1e-3 < m            # hungarian recovers it, matrix doesn't


def test_gradient_flows():
    batch, iu, pres, jval, deg_cls = _targets()
    shifts = batch.shifts.clone().requires_grad_(True)
    cvals = jval.clone().requires_grad_(True)
    out = ModelOutput(shifts=shifts, coupling_values=cvals,
                      coupling_presence_logits=torch.zeros(B, E, requires_grad=True),
                      degeneracy_logits=torch.zeros(B, G, C, requires_grad=True))
    lo = HungarianGraphLoss()(out, batch).validate()
    lo.total.backward()
    assert shifts.grad is not None and torch.isfinite(shifts.grad).all()


def test_build_via_registry_with_class_weights():
    loss = build_loss("hungarian", deg_class_weight=[1.0] * C, presence_pos_weight=2.0)
    assert isinstance(loss, HungarianGraphLoss)


def test_composite_can_use_hungarian():
    from model.losses import build_composite
    batch, iu, pres, jval, deg_cls = _targets()
    out = _perfect_output(batch, iu, pres, jval, deg_cls)
    comp = build_composite([{"name": "hungarian", "weight": 1.0}])
    assert comp(out, batch).total.item() < 1e-3
