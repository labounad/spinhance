"""
Soft-equivalence (Idea 2) + peak-channel (Idea 1) tests.

Idea 2: the spingraph edge head emits a per-edge soft_equiv logit
(auxiliary["soft_equiv_logits"]); SoftEquivLoss supervises it (BCE) against the
ground-truth "same shift" label and pulls predicted shifts of those pairs
together; decode() averages soft-equivalent groups to a shared shift.

Idea 1: use_peak_channel feeds a second conv input channel (peak-emphasis map)
derived from the spectrum; the model contract is otherwise unchanged.

Tiny dims, CPU. P must equal N_POINTS (ppm_pos is sized to it).
"""
import numpy as np
import torch

from model.architectures import build_architecture
from model.evaluation.metrics import decode
from model.losses import build_loss, build_composite
from model.losses.soft_equiv_loss import SoftEquivLoss
from model.schemas import ModelOutput, SpinBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB, N_POINTS

B, G, P = 2, 8, N_POINTS
C = len(DEFAULT_DEG_VOCAB)
E = G * (G - 1) // 2


def _model(**kw):
    return build_architecture("spingraph_decoder", n_deg_classes=C, size="tiny",
                              dim=32, enc_layers=1, dec_layers=1, n_heads=2,
                              node_hidden=32, edge_hidden=32, **kw)


def _spec():
    s = torch.rand(B, P)
    return s / (s.sum(-1, keepdim=True) * (12.0 / P))     # unit integral


def _batch(shifts=None):
    if shifts is None:
        shifts = torch.randn(B, G)
    spec = _spec()
    return SpinBatch(spectrum=spec, spectrum_ref=spec,
                     shifts=shifts, couplings=torch.zeros(B, G, G),
                     coupling_mask=torch.zeros(B, G, G),
                     degeneracy_classes=torch.zeros(B, G, dtype=torch.long),
                     degeneracy_values=torch.ones(B, G),
                     molecule_ids=[f"m{i}" for i in range(B)])


def _output(requires_grad=True, with_aux=True):
    g = requires_grad
    aux = {"soft_equiv_logits": torch.randn(B, E, requires_grad=g)} if with_aux else {}
    return ModelOutput(
        shifts=torch.randn(B, G, requires_grad=g),
        coupling_values=torch.randn(B, E, requires_grad=g),
        coupling_presence_logits=torch.randn(B, E, requires_grad=g),
        degeneracy_logits=torch.randn(B, G, C, requires_grad=g),
        auxiliary=aux)


# ── model contract: aux logits + peak channel ──────────────────────────────────

def test_soft_equiv_logits_exposed_only_when_enabled():
    # default (e.g. the 025 recipe): the flag is UNtrained, so it must NOT be exposed —
    # decode keys off the key's presence and would otherwise average random groups.
    assert "soft_equiv_logits" not in _model().eval()(_batch()).auxiliary
    # use_soft_equiv=True (the 026 recipe): exposed for the loss + decode-time averaging.
    se = _model(use_soft_equiv=True).eval()(_batch()).auxiliary["soft_equiv_logits"]
    assert se.shape == (B, E) and torch.isfinite(se).all()


def test_peak_channel_forward_and_two_input_channels():
    m = _model(use_peak_channel=True).eval()
    assert m.encoder.in_channels == 2
    out = m(_batch()).validate(n_groups=G)
    assert out.shifts.shape == (B, G) and torch.isfinite(out.shifts).all()
    # the peak channel is a real, peaked signal (not all zeros) for a random spectrum
    pk = m._peak_channel(_spec())
    assert pk.shape == (B, P) and float(pk.sum()) > 0


def test_peak_channel_off_is_single_channel():
    assert _model().encoder.in_channels == 1


# ── soft-equiv loss ─────────────────────────────────────────────────────────────

def test_label_marks_equal_shift_pairs():
    # groups 0 and 1 share the same shift in sample 0 -> that edge is soft-equivalent
    shifts = torch.arange(G, dtype=torch.float32).repeat(B, 1)
    shifts[0, 1] = shifts[0, 0]
    loss = SoftEquivLoss(tol_ppm=0.03, shift_std=1.0)
    lo = loss(_output(with_aux=True), _batch(shifts=shifts)).validate()
    assert lo.metrics["se_frac"] > 0
    assert torch.isfinite(lo.total)


def test_grad_flows_to_shifts_and_flag():
    shifts = torch.zeros(B, G)            # all-equal -> many soft-equiv pairs
    out = _output(with_aux=True)
    SoftEquivLoss(shift_std=1.0)(out, _batch(shifts=shifts)).total.backward()
    assert out.shifts.grad is not None and torch.isfinite(out.shifts.grad).all()
    se = out.auxiliary["soft_equiv_logits"]
    assert se.grad is not None and float(se.grad.abs().sum()) > 0


def test_consistency_penalizes_split_shifts():
    """Two GT-equal groups with predicted shifts pulled apart cost more than together."""
    shifts = torch.zeros(B, G)
    base = _output(with_aux=True, requires_grad=False)
    loss = SoftEquivLoss(consistency_weight=1.0, shift_std=1.0)
    split = ModelOutput(shifts=base.shifts.clone(), coupling_values=base.coupling_values,
                        coupling_presence_logits=base.coupling_presence_logits,
                        degeneracy_logits=base.degeneracy_logits, auxiliary=base.auxiliary)
    split.shifts[:, 0] = 1.0; split.shifts[:, 1] = -1.0     # force a wide split on a GT pair
    together = ModelOutput(shifts=torch.zeros(B, G), coupling_values=base.coupling_values,
                           coupling_presence_logits=base.coupling_presence_logits,
                           degeneracy_logits=base.degeneracy_logits, auxiliary=base.auxiliary)
    assert loss(split, _batch(shifts=shifts)).metrics["consistency"] > \
           loss(together, _batch(shifts=shifts)).metrics["consistency"]


def test_no_aux_is_noop():
    lo = SoftEquivLoss(shift_std=1.0)(_output(with_aux=False), _batch())
    assert float(lo.total) == 0.0 and lo.diagnostics.get("skipped")


def test_composite_with_matrix_and_soft_equiv():
    terms = [{"name": "matrix", "weight": 1.0},
             {"name": "soft_equiv", "weight": 0.5, "tol_ppm": 0.03}]
    comp = build_composite(terms, shift_mean=5.0, shift_std=2.0, j_mean=7.0, j_std=4.0)
    lo = comp(_output(with_aux=True), _batch())
    assert torch.isfinite(lo.total)
    assert "soft_equiv/bce" in lo.metrics


# ── decode-time averaging ───────────────────────────────────────────────────────

class _IdStd:
    def inverse_shifts(self, x): return np.asarray(x, float)
    def inverse_j(self, x): return np.asarray(x, float)


class _Vocab:
    def from_index(self, idx): return np.ones_like(np.asarray(idx))


def test_decode_averages_soft_equiv_pair():
    G4, E4 = 4, 6
    pred = {"shifts": np.array([[3.59, 3.61, 7.0, 1.0]]),
            "j_mag": np.zeros((1, E4)), "j_presence": np.full((1, E4), -9.0),
            "deg_logits": np.zeros((1, G4, C))}
    # triu(4) edge order: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3); flag only edge (0,1)
    se = np.zeros((1, E4)); se[0, 0] = 0.99
    pred["soft_equiv"] = se
    dec = decode(pred, _IdStd(), _Vocab(), soft_equiv_thresh=0.5)
    assert abs(dec["shifts"][0, 0] - 3.60) < 1e-6
    assert abs(dec["shifts"][0, 1] - 3.60) < 1e-6      # both collapsed to the mean
    assert abs(dec["shifts"][0, 2] - 7.0) < 1e-6       # untouched


def test_decode_without_soft_equiv_unchanged():
    G4, E4 = 4, 6
    pred = {"shifts": np.array([[3.59, 3.61, 7.0, 1.0]]),
            "j_mag": np.zeros((1, E4)), "j_presence": np.full((1, E4), -9.0),
            "deg_logits": np.zeros((1, G4, C))}
    dec = decode(pred, _IdStd(), _Vocab())
    assert abs(dec["shifts"][0, 0] - 3.59) < 1e-6      # no averaging without the flag
