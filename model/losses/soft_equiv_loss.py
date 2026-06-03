"""
model.losses.soft_equiv_loss
============================
Soft-equivalence edge supervision (Idea 2). Two spin groups are *soft-equivalent*
iff they are **chemically equivalent (same canonical symmetry orbit) but not
hard-equivalent** — i.e. related by molecular symmetry yet kept as distinct nodes
(an AA'BB' pair, or two symmetry-equivalent OMe). They therefore have identical
chemical shifts but can have different coupling rows. This is determined PURELY by
symmetry (the orbit), NOT by shift proximity — genuine *accidental* degeneracies
(distinct protons that merely overlap in shift) are deliberately NOT labelled, so
the head learns the real, learnable symmetry signal rather than "shifts are close."

Left unconstrained the model regresses their shifts independently and lands on,
say, 3.59 / 3.61 — the rendered spectrum then shows a spurious split doublet
instead of the correct single peak, hurting demonstrative fidelity even though
the per-shift MAE is tiny.

This loss adds two complementary signals on the symmetric upper-triangle edges
(same triu order as the coupling edges):

  * **flag BCE** — supervise the edge head's ``soft_equiv_logits`` against the
    symmetry-orbit ground-truth label (``batch.soft_equiv_target``) so the decoder
    learns to *identify* soft-equivalent pairs (used at inference to average them).
  * **shift-consistency** — for ground-truth soft-equivalent pairs, penalize the
    squared difference of the *predicted* shifts, directly pulling them together
    so they collapse to one peak. This is the differentiable form of the
    "average their shifts" rule; the hard averaging happens at decode time
    (``model.evaluation.metrics.decode``) gated on the predicted flag.

The label comes from ``batch.soft_equiv_target`` (canonical symmetry orbit, built
in the dataset). Works only with architectures that emit
``auxiliary["soft_equiv_logits"]`` (the spingraph_decoder edge head) AND batches
that carry ``soft_equiv_target``; a no-op zero loss otherwise so it stays
composable. (``tol_ppm``/``shift_std`` are accepted for config compatibility but
no longer used — the label is symmetry-derived, not a shift tolerance.)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.losses.base import Loss
from model.losses.registry import LOSSES
from model.schemas import LossOutput, ModelOutput, SpinBatch


@LOSSES.register("soft_equiv")
class SoftEquivLoss(Loss):
    name = "soft_equiv"

    def __init__(self, tol_ppm: float = 0.03, consistency_weight: float = 1.0,
                 shift_std: float = 1.0, max_pos_weight: float = 50.0, **_ignore):
        self.tol_std = float(tol_ppm) / float(shift_std)     # tolerance in standardized space
        self.consistency_weight = consistency_weight
        self.max_pos_weight = max_pos_weight

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:
        se_logits = output.auxiliary.get("soft_equiv_logits")
        if se_logits is None:                                # arch emits no flag -> no-op
            z = output.shifts.new_zeros(())
            return LossOutput(total=z, components={}, metrics={},
                              diagnostics={"skipped": "no soft_equiv_logits"})
        se_target = getattr(batch, "soft_equiv_target", None)
        if se_target is None:                                # no symmetry labels -> no-op
            z = output.shifts.new_zeros(())
            return LossOutput(total=z, components={}, metrics={},
                              diagnostics={"skipped": "no soft_equiv_target"})

        device = output.shifts.device
        G = output.n_groups
        iu = torch.triu_indices(G, G, 1, device=device)

        # Symmetry-derived label: 1 iff the two groups share a canonical orbit
        # (soft-equivalent = chemically equivalent, not hard-merged). Extracted with
        # the SAME triu the edge logits use, so ordering is guaranteed aligned. This
        # is NOT shift proximity — accidental near-degeneracies are deliberately
        # left unlabelled so the head learns the real (learnable) symmetry signal.
        label = se_target[:, iu[0], iu[1]].float()           # (B, E)

        # rare-positive imbalance: weight present pairs by neg/pos (clamped)
        n_pos = label.sum()
        if n_pos > 0:
            pw = (label.numel() - n_pos) / n_pos
            pw = pw.clamp(1.0, self.max_pos_weight)
        else:
            pw = None
        bce = F.binary_cross_entropy_with_logits(se_logits, label, pos_weight=pw)

        pred_i = output.shifts[:, iu[0]]
        pred_j = output.shifts[:, iu[1]]
        diff2 = (pred_i - pred_j) ** 2 * label               # only GT soft-equiv pairs
        consistency = diff2.sum() / n_pos.clamp_min(1.0)

        total = bce + self.consistency_weight * consistency

        with torch.no_grad():
            pred = (se_logits > 0)
            lab = label.bool()
            tp = (pred & lab).sum().float()
            acc = float((pred == lab).float().mean())
            recall = float(tp / n_pos) if n_pos > 0 else 0.0
        metrics = {"bce": float(bce.detach()), "consistency": float(consistency.detach()),
                   "se_frac": float(label.mean()), "se_acc": acc, "se_recall": recall}
        components = {"bce": bce, "consistency": consistency}
        return LossOutput(total=total, components=components, metrics=metrics)
