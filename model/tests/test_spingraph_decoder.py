"""
SpinGraphDecoderModel (IDEAS north-star, Phase 1) tests: registration + shape
contract, edge-list/triu order matching ModelOutput, permutation-equivariant
symmetric edge head, graceful region_tokens=None vs present, and gradient flow
through the existing Hungarian (Stage-1) and surrogate spectral (Stage-2) losses
under bf16 autocast. Tiny dims, CPU. P must equal N_POINTS (ppm_pos is sized to it).
"""
import torch

from model.architectures import ARCHITECTURES, build_architecture
from model.architectures.spingraph_decoder import SpinGraphDecoderModel
from model.heads.pairwise_edge_head import PairwiseEdgeHead
from model.losses import build_loss
from model.losses.surrogate_spectral_loss import SurrogateSpectralLoss
from model.renderers.surrogate import SurrogateRenderer
from model.schemas import ModelOutput, SpinBatch
from model.schemas.batch import RegionTokenBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB, N_POINTS

B, G, P = 2, 8, N_POINTS
C = len(DEFAULT_DEG_VOCAB)


def _model():
    return build_architecture("spingraph_decoder", n_deg_classes=C, size="tiny",
                              dim=32, enc_layers=1, dec_layers=1, n_heads=2,
                              node_hidden=32, edge_hidden=32, region_feat_dim=80)


def _batch(regions=False):
    spec = torch.rand(B, P)
    spec = spec / (spec.sum(-1, keepdim=True) * (12.0 / P))    # unit integral
    rt = None
    if regions:
        R = 5
        feats = torch.randn(B, R, 80)
        mask = torch.ones(B, R)
        mask[:, 3:] = 0                                        # last 2 padded
        rt = RegionTokenBatch(features=feats, mask=mask)
    return SpinBatch(spectrum=spec, spectrum_ref=spec,
                     shifts=torch.randn(B, G), couplings=torch.zeros(B, G, G),
                     coupling_mask=torch.zeros(B, G, G),
                     degeneracy_classes=torch.zeros(B, G, dtype=torch.long),
                     degeneracy_values=torch.ones(B, G),
                     molecule_ids=[f"m{i}" for i in range(B)],
                     region_tokens=rt)


def test_registered_and_builds():
    assert "spingraph_decoder" in ARCHITECTURES
    assert isinstance(_model(), SpinGraphDecoderModel)


def test_shape_contract_global_only():
    m = _model().eval()
    out = m(_batch(regions=False)).validate(n_groups=G)
    assert out.shifts.shape == (B, G)
    assert out.coupling_matrix().shape == (B, G, G)
    assert out.degeneracy_logits.shape == (B, G, C)
    assert torch.isfinite(out.shifts).all()
    # also accepts a raw spectrum tensor
    raw = m(torch.rand(B, P)).validate(n_groups=G)
    assert raw.shifts.shape == (B, G)


def test_region_tokens_optional():
    m = _model().eval()
    o0 = m(_batch(regions=False))
    o1 = m(_batch(regions=True))
    # identical output contract whether or not region tokens are present
    assert o0.shifts.shape == o1.shifts.shape == (B, G)
    assert o1.coupling_matrix().shape == (B, G, G)


def test_edge_head_symmetric_and_equivariant():
    """edge_ij depends only on the unordered {h_i, h_j}: swapping two nodes
    permutes the coupling matrix rows/cols accordingly, and the matrix is symmetric."""
    head = PairwiseEdgeHead(dim=16, n_groups=4).eval()
    h = torch.randn(1, 4, 16)
    jmag, _, _ = head(h)
    M = ModelOutput(shifts=torch.zeros(1, 4), coupling_values=jmag,
                    coupling_presence_logits=jmag, degeneracy_logits=torch.zeros(1, 4, C)
                    ).coupling_matrix()
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-6)        # symmetric
    perm = [1, 0, 2, 3]
    jp, _, _ = head(h[:, perm, :])
    Mp = ModelOutput(shifts=torch.zeros(1, 4), coupling_values=jp,
                     coupling_presence_logits=jp, degeneracy_logits=torch.zeros(1, 4, C)
                     ).coupling_matrix()
    assert torch.allclose(Mp, M[:, perm][:, :, perm], atol=1e-5)    # equivariant


def test_hungarian_grad_flow_under_autocast():
    m = _model()
    loss = build_loss("hungarian", match_degeneracy_weight=1.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m(_batch())
        lo = loss(out, _batch())
    assert torch.isfinite(lo.total)
    lo.total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_surrogate_spectral_grad_flow(tmp_path):
    # tiny frozen surrogate checkpoint in the trainer's {model, cfg} format
    sm = SurrogateRenderer(dim=32, depth=2, heads=2, sticks_per_group=16, points=P)
    ckpt = tmp_path / "surr.pt"
    torch.save({"model": sm.state_dict(),
                "cfg": {"model": {"dim": 32, "depth": 2, "heads": 2,
                                  "sticks_per_group": 16, "points": P}}}, ckpt)
    m = _model()
    loss = SurrogateSpectralLoss(checkpoint=str(ckpt), field=90,
                                 shift_mean=5.0, shift_std=2.0, j_mean=7.0, j_std=4.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        lo = loss(m(_batch()), _batch())
    assert torch.isfinite(lo.total)
    lo.total.backward()
    assert any(p.grad is not None for p in m.parameters())
