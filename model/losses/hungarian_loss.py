"""
model.losses.hungarian_loss
===========================
Permutation-invariant graph loss (master plan §Loss / family K). The G spin-group
labels are arbitrary, so canonical-order element comparison injects label noise
(near-tied shifts swap order). This loss instead solves an optimal assignment
between predicted and true groups, then applies the standard matrix term losses
on the matched pairs.

Pipeline:
  1. cost[b,i,j] = |tgt_shift_i - pred_shift_j| + w_deg * 1[deg_pred_j != deg_tgt_i]
     (+ optional weak coupling-row cost), computed on DETACHED predictions.
  2. linear_sum_assignment per sample -> perm (B, G): pred index matched to target i.
  3. permute predicted nodes (shifts, degeneracy logits) and edges (coupling /
     presence matrices) by perm — differentiable gather, so grads flow.
  4. delegate the matched term losses to MatrixLoss (no duplicated math).

The matching step is non-differentiable (argmin assignment), as in DETR; gradients
flow through the matched loss terms only.
"""
from __future__ import annotations

import numpy as np
import torch

from model.losses.base import Loss
from model.losses.matrix_loss import MatrixLoss
from model.losses.registry import LOSSES
from model.schemas import LossOutput, ModelOutput, SpinBatch


def _assign(cost: np.ndarray) -> np.ndarray:
    """(B, G, G) cost (rows=target, cols=pred) -> (B, G) matched pred index per target."""
    from scipy.optimize import linear_sum_assignment
    B, G, _ = cost.shape
    perms = np.zeros((B, G), dtype=np.int64)
    for b in range(B):
        _, col = linear_sum_assignment(cost[b])
        perms[b] = col
    return perms


@LOSSES.register("hungarian")
class HungarianGraphLoss(Loss):
    name = "hungarian"

    def __init__(self, weights=None, huber_beta: float = 1.0,
                 deg_class_weight=None, presence_pos_weight=None,
                 match_degeneracy_weight: float = 1.0,
                 match_coupling_weight: float = 0.0):
        self.match_deg_w = match_degeneracy_weight
        self.match_coup_w = match_coupling_weight
        # All matched-term math is delegated to MatrixLoss (same weights/class balance).
        self._matrix = MatrixLoss(weights=weights, huber_beta=huber_beta,
                                  deg_class_weight=deg_class_weight,
                                  presence_pos_weight=presence_pos_weight)

    # ── matching ───────────────────────────────────────────────────────────────

    def _match(self, output: ModelOutput, batch: SpinBatch) -> torch.Tensor:
        pred_shifts = output.shifts.detach().float().cpu().numpy()           # (B,G)
        tgt_shifts = batch.shifts.detach().float().cpu().numpy()
        pred_deg = output.degeneracy_logits.detach().argmax(-1).cpu().numpy()  # (B,G)
        tgt_deg = batch.degeneracy_classes.detach().cpu().numpy()

        # rows = target i, cols = pred j
        cost = np.abs(tgt_shifts[:, :, None] - pred_shifts[:, None, :])
        if self.match_deg_w:
            cost = cost + self.match_deg_w * (tgt_deg[:, :, None] != pred_deg[:, None, :])
        if self.match_coup_w:
            pc = np.abs(output.coupling_matrix().detach().float().cpu().numpy()).sum(-1)  # (B,G) pred row sums
            tc = np.abs(batch.couplings.detach().float().cpu().numpy()).sum(-1)
            cost = cost + self.match_coup_w * np.abs(tc[:, :, None] - pc[:, None, :])

        perms = _assign(cost)
        return torch.as_tensor(perms, device=output.shifts.device)           # (B,G)

    # ── apply ──────────────────────────────────────────────────────────────────

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:
        perm = self._match(output, batch)                                    # (B,G)
        B, G = perm.shape
        bi = torch.arange(B, device=perm.device)

        aligned_shifts = output.shifts[bi[:, None], perm]                    # (B,G)
        aligned_deg = output.degeneracy_logits[bi[:, None], perm]            # (B,G,C)

        predC = output.coupling_matrix()                                     # (B,G,G)
        predP = output.presence_matrix()
        b3 = bi[:, None, None]
        rows = perm[:, :, None]
        cols = perm[:, None, :]
        aligned_C = predC[b3, rows, cols]                                    # (B,G,G)
        aligned_P = predP[b3, rows, cols]

        aligned = ModelOutput(
            shifts=aligned_shifts,
            coupling_values=aligned_C,
            coupling_presence_logits=aligned_P,
            degeneracy_logits=aligned_deg,
        )
        lo = self._matrix(aligned, batch)
        lo.diagnostics["matched"] = True
        return lo
