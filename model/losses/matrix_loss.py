"""
model.losses.matrix_loss
========================
Stage-1 supervised matrix loss (ported from legacy losses.py), operating on the
matrix-form contract. No renderer dependency.

  shifts      -> smooth-L1 (Huber)                       standardized ppm
  couplings   -> smooth-L1, MASKED by ground-truth presence (upper triangle)
  presence    -> BCE-with-logits vs the coupling mask
  degeneracy  -> cross-entropy over the degeneracy vocab

``deg_class_weight`` (C,) and ``presence_pos_weight`` (scalar) counter class
imbalance (degeneracy ~89% d=1; couplings sparse). They are optional and moved to
the prediction's device lazily.

Two optional **focal** modulators (Lin et al. 2017) down-weight easy, well-classified
examples so gradients concentrate on the hard tail — they target the two documented
imbalance bottlenecks without touching the converged shift/J-magnitude terms:

  * ``deg_focal_gamma``      focal cross-entropy on degeneracy (compounds with the
                             class weights) -> lifts rare-class balanced accuracy.
  * ``presence_focal_gamma`` focal BCE on coupling presence -> sharpens the sparse,
                             easy-negative-dominated presence boundary -> F1.

Both default to 0.0, which reproduces the plain CE / BCE exactly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.losses.base import Loss
from model.losses.registry import LOSSES
from model.schemas import LossOutput, ModelOutput, SpinBatch

_DEFAULT_WEIGHTS = {"shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5}


def _as_tensor(x):
    if x is None or torch.is_tensor(x):
        return x
    return torch.as_tensor(x, dtype=torch.float32)


@LOSSES.register("matrix")
class MatrixLoss(Loss):
    name = "matrix"

    def __init__(self, weights=None, huber_beta: float = 1.0,
                 deg_class_weight=None, presence_pos_weight=None,
                 deg_focal_gamma: float = 0.0, presence_focal_gamma: float = 0.0,
                 sym_jmag: bool = False):
        self.w = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.huber_beta = huber_beta
        self.deg_class_weight = _as_tensor(deg_class_weight)
        self.presence_pos_weight = _as_tensor(presence_pos_weight)
        self.deg_focal_gamma = float(deg_focal_gamma)
        self.presence_focal_gamma = float(presence_focal_gamma)
        # symmetry-aware coupling term: score J against the best within-orbit relabeling
        # (batch.sym_perms), so swapping chemically-equivalent groups isn't penalized.
        self.sym_jmag = bool(sym_jmag)

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:
        device = output.shifts.device
        G = output.n_groups
        iu = torch.triu_indices(G, G, 1, device=device)

        pred_j = output.coupling_matrix()[:, iu[0], iu[1]]            # (B, E)
        tgt_j = batch.couplings[:, iu[0], iu[1]]
        mask = batch.coupling_mask[:, iu[0], iu[1]]                   # {0,1}
        pred_pres = output.presence_matrix()[:, iu[0], iu[1]]        # logits

        shift = F.smooth_l1_loss(output.shifts, batch.shifts, beta=self.huber_beta)

        sym = getattr(batch, "sym_perms", None)
        if self.sym_jmag and sym is not None:
            # min the coupling term over the within-orbit label relabelings: permute the
            # predicted coupling matrix by each perm, score vs the (fixed canonical) target,
            # take the per-sample best. Identity is row 0, so it's never worse than plain J.
            predC = output.coupling_matrix()                              # (B,G,G)
            B, K, _ = sym.shape
            bi = torch.arange(B, device=device)
            per_k = []
            for k in range(K):
                pk = sym[:, k, :]                                          # (B,G)
                predC_k = predC[bi[:, None, None], pk[:, :, None], pk[:, None, :]]  # (B,G,G)
                el = F.smooth_l1_loss(predC_k[:, iu[0], iu[1]], tgt_j,
                                      beta=self.huber_beta, reduction="none")       # (B,E)
                per_k.append((el * mask).sum(dim=1))                       # (B,)
            best = torch.stack(per_k, dim=1).min(dim=1).values            # (B,)
            jmag = (best / mask.sum(dim=1).clamp_min(1.0)).mean()
        else:
            jmag_el = F.smooth_l1_loss(pred_j, tgt_j, beta=self.huber_beta, reduction="none")
            jmag = (jmag_el * mask).sum() / mask.sum().clamp_min(1.0)

        ppw = self.presence_pos_weight
        if ppw is not None:
            ppw = ppw.to(device)
        if self.presence_focal_gamma > 0:
            bce_el = F.binary_cross_entropy_with_logits(
                pred_pres, mask, pos_weight=ppw, reduction="none")
            p = torch.sigmoid(pred_pres)
            pt = p * mask + (1.0 - p) * (1.0 - mask)             # prob of the true class
            focal = (1.0 - pt).clamp_min(0.0).pow(self.presence_focal_gamma)
            presence = (focal * bce_el).mean()
        else:
            presence = F.binary_cross_entropy_with_logits(pred_pres, mask, pos_weight=ppw)

        B, Gd, C = output.degeneracy_logits.shape
        dcw = self.deg_class_weight
        if dcw is not None:
            dcw = dcw.to(device)
        deg_logits = output.degeneracy_logits.reshape(B * Gd, C)
        deg_tgt = batch.degeneracy_classes.reshape(B * Gd)
        if self.deg_focal_gamma > 0:
            logp = F.log_softmax(deg_logits, dim=-1)
            logp_t = logp.gather(1, deg_tgt[:, None]).squeeze(1)  # (N,) log prob of target
            focal = (1.0 - logp_t.exp()).clamp_min(0.0).pow(self.deg_focal_gamma)
            ce_t = -logp_t
            if dcw is not None:                                  # weighted-mean (matches CE semantics)
                w_t = dcw[deg_tgt]
                deg = (focal * w_t * ce_t).sum() / w_t.sum().clamp_min(1.0)
            else:
                deg = (focal * ce_t).mean()
        else:
            deg = F.cross_entropy(deg_logits, deg_tgt, weight=dcw)

        total = (self.w["shift"] * shift + self.w["jmag"] * jmag
                 + self.w["presence"] * presence + self.w["deg"] * deg)

        components = {"shift": shift, "jmag": jmag, "presence": presence, "deg": deg}
        metrics = {k: float(v.detach()) for k, v in components.items()}
        return LossOutput(total=total, components=components, metrics=metrics)
