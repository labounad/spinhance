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
                 deg_class_weight=None, presence_pos_weight=None):
        self.w = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.huber_beta = huber_beta
        self.deg_class_weight = _as_tensor(deg_class_weight)
        self.presence_pos_weight = _as_tensor(presence_pos_weight)

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:
        device = output.shifts.device
        G = output.n_groups
        iu = torch.triu_indices(G, G, 1, device=device)

        pred_j = output.coupling_matrix()[:, iu[0], iu[1]]            # (B, E)
        tgt_j = batch.couplings[:, iu[0], iu[1]]
        mask = batch.coupling_mask[:, iu[0], iu[1]]                   # {0,1}
        pred_pres = output.presence_matrix()[:, iu[0], iu[1]]        # logits

        shift = F.smooth_l1_loss(output.shifts, batch.shifts, beta=self.huber_beta)

        jmag_el = F.smooth_l1_loss(pred_j, tgt_j, beta=self.huber_beta, reduction="none")
        jmag = (jmag_el * mask).sum() / mask.sum().clamp_min(1.0)

        ppw = self.presence_pos_weight
        if ppw is not None:
            ppw = ppw.to(device)
        presence = F.binary_cross_entropy_with_logits(pred_pres, mask, pos_weight=ppw)

        B, Gd, C = output.degeneracy_logits.shape
        dcw = self.deg_class_weight
        if dcw is not None:
            dcw = dcw.to(device)
        deg = F.cross_entropy(output.degeneracy_logits.reshape(B * Gd, C),
                              batch.degeneracy_classes.reshape(B * Gd), weight=dcw)

        total = (self.w["shift"] * shift + self.w["jmag"] * jmag
                 + self.w["presence"] * presence + self.w["deg"] * deg)

        components = {"shift": shift, "jmag": jmag, "presence": presence, "deg": deg}
        metrics = {k: float(v.detach()) for k, v in components.items()}
        return LossOutput(total=total, components=components, metrics=metrics)
