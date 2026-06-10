"""Permutation-invariant matrix loss via a shift-gated soft assignment (the v3
"gauge-equivariant" output-representation effort — see model/PERMUTATION_INVARIANT_DESIGN.md).

Why this exists
---------------
The spin-system graph carries *arbitrary* node labels (a gauge with no physical
meaning). The canonical shift-sort picks one representative, and the element-wise
``MatrixLoss`` scores against it. That representative is **multivalued at exactly-equal
shifts** and **nearly discontinuous at near-equal shifts** (an infinitesimal shift change
flips the sort order and transposes whole rows/cols of the coupling matrix). So the
element-wise loss demands a hard *ranking* of near-degenerate groups that the 90 MHz
spectrum cannot resolve — harshness that grows exactly where the observable is least
informative.

Fix: do not score against a hard canonical order. Build a **soft assignment** ``P``
(doubly-stochastic, via Sinkhorn) between predicted and target nodes from *shift
proximity*, align the prediction into the target's frame, and apply the usual matrix
terms on the aligned prediction. Key properties:

* **Well-separated shifts → P ≈ identity → reduces to the element-wise MatrixLoss.**
  (Continuity guarantee; verified by the tau->0 smoke test.)
* **Near-degenerate shifts → P spreads mass** between the close nodes, so the loss stops
  penalizing which of two unresolvable nodes a coupling is attached to.
* The coupling matrix is aligned by the **bilinear form ``Pᵀ J P``** — couplings move
  *with* their nodes, so edge structure is preserved (this is the graph-matching-aware
  alignment that naive node-only Hungarian got wrong).

This is *not* a spectral loss: the chemistry (labeled target) remains the supervisor;
``P`` is gated by shift proximity so it can only ever relabel nodes the spectrum itself
leaves ambiguous. It therefore cannot over-credit (it never invents coupling values, only
reassigns them among nodes the observable can't tell apart).

v3.0 scope: the assignment cost uses shift proximity only. A degeneracy-aware cost
(forbid soft-swapping nodes of different multiplicity, which ARE spectrum-distinguishable)
is a planned v3.1 refinement — noted in the design doc.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.losses.base import Loss
from model.losses.registry import LOSSES
from model.schemas import LossOutput, ModelOutput, SpinBatch

_DEFAULT_WEIGHTS = {"shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5}


def sinkhorn_log(cost: torch.Tensor, tau: float, n_iters: int) -> torch.Tensor:
    """Doubly-stochastic soft assignment from a (B, G, G) cost, in log-space.

    ``P[b, i, j]`` is the soft mass assigning predicted node ``i`` to target node ``j``.
    Lower cost -> higher mass. ``tau`` is the entropic temperature (tau->0 -> hard
    permutation; large tau -> uniform). Returns ``P`` (B, G, G), rows & cols ~sum to 1.
    """
    logP = -cost / max(tau, 1e-6)
    for _ in range(n_iters):
        logP = logP - torch.logsumexp(logP, dim=2, keepdim=True)   # row-normalize
        logP = logP - torch.logsumexp(logP, dim=1, keepdim=True)   # col-normalize
    return logP.exp()


@LOSSES.register("sinkhorn_align")
class SinkhornAlignLoss(Loss):
    """Shift-gated soft-assignment matrix loss. Drop-in generalization of ``matrix``.

    Args:
      weights: per-term weights {"shift","jmag","presence","deg"} (defaults match matrix).
      tau: Sinkhorn temperature on the (standardized-shift)² cost. Small -> near-hard.
      n_iters: Sinkhorn normalization iterations.
      huber_beta: smooth-L1 beta for shift/jmag (matches matrix loss).
      hard_eval: if True, snap P to the argmax permutation when not training (report-time).
    """

    name = "sinkhorn_align"

    def __init__(self, weights=None, tau: float = 0.05, n_iters: int = 50,
                 huber_beta: float = 1.0, hard_eval: bool = False,
                 detach_assign_shifts: bool = False, **_ignore):
        self.w = dict(_DEFAULT_WEIGHTS)
        if weights:
            self.w.update(weights)
        self.tau = float(tau)
        self.n_iters = int(n_iters)
        self.huber_beta = float(huber_beta)
        self.hard_eval = bool(hard_eval)
        # If True, the soft assignment P is built from DETACHED predicted shifts: P still
        # uses current shifts to choose the assignment, but no gradient flows from the
        # coupling alignment back into the shifts — so the coupling term relaxes
        # near-degenerate label assignments WITHOUT perturbing shifts (those stay purely
        # matrix-supervised). Pair with weights {shift:0,deg:0,presence:0,jmag:1} for a
        # clean coupling-only permutation-invariant term.
        self.detach_assign_shifts = bool(detach_assign_shifts)

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:
        psh = output.shifts                       # (B, G) standardized ppm
        tsh = batch.shifts                         # (B, G)
        B, G = psh.shape
        device = psh.device
        iu = torch.triu_indices(G, G, 1, device=device)

        pJ = output.coupling_matrix()              # (B, G, G) standardized Hz
        pPres = output.presence_matrix()           # (B, G, G) logits
        pDeg = output.degeneracy_logits            # (B, G, C)
        tJ = batch.couplings                       # (B, G, G)
        tmask = batch.coupling_mask                # (B, G, G) {0,1}
        tdeg = batch.degeneracy_classes            # (B, G) long

        # --- soft assignment: predicted node i -> target node j, gated by shift proximity
        psh_cost = psh.detach() if self.detach_assign_shifts else psh
        cost = (psh_cost[:, :, None] - tsh[:, None, :]) ** 2       # (B, G, G)
        P = sinkhorn_log(cost, self.tau, self.n_iters)            # (B, G, G)
        if self.hard_eval and not torch.is_grad_enabled():
            # snap to a hard permutation (one-hot per column) for reporting parity
            idx = P.argmax(dim=1)                                  # (B, G) pred index per target
            P = F.one_hot(idx, num_classes=G).transpose(1, 2).to(P.dtype)

        # --- align prediction into the TARGET node frame
        al_sh = torch.einsum("bij,bi->bj", P, psh)                # (B, G)
        al_J = torch.einsum("bij,bik,bkl->bjl", P, pJ, P)         # (B, G, G) = Pᵀ J P
        al_Pres = torch.einsum("bij,bik,bkl->bjl", P, pPres, P)   # (B, G, G) logits
        al_Deg = torch.einsum("bij,bic->bjc", P, pDeg)            # (B, G, C)

        # --- standard matrix terms on the aligned prediction vs the (fixed) target
        shift = F.smooth_l1_loss(al_sh, tsh, beta=self.huber_beta)

        pj = al_J[:, iu[0], iu[1]]                                 # (B, E)
        tj = tJ[:, iu[0], iu[1]]
        m = tmask[:, iu[0], iu[1]]
        jmag_e = F.smooth_l1_loss(pj, tj, beta=self.huber_beta, reduction="none") * m
        jmag = jmag_e.sum() / m.sum().clamp_min(1.0)

        presence = F.binary_cross_entropy_with_logits(al_Pres[:, iu[0], iu[1]], m)

        deg = F.cross_entropy(al_Deg.reshape(B * G, -1), tdeg.reshape(B * G))

        total = (self.w["shift"] * shift + self.w["jmag"] * jmag
                 + self.w["presence"] * presence + self.w["deg"] * deg)

        # --- diagnostics: how soft is the assignment? (validates the hypothesis live)
        diag = torch.einsum("bii->b", P) / G                       # mean diagonal mass per sample
        off_diag_mass = float((1.0 - diag).mean().detach())        # 0 => identity; >0 => relabeling
        row_entropy = float((-(P.clamp_min(1e-9).log() * P).sum(dim=2).mean()).detach())

        components = {"shift": shift, "jmag": jmag, "presence": presence, "deg": deg}
        metrics = {k: float(v.detach()) for k, v in components.items()}
        metrics["assign_offdiag"] = off_diag_mass
        metrics["assign_entropy"] = row_entropy
        return LossOutput(total=total, components=components, metrics=metrics,
                          diagnostics={"tau": self.tau}).validate()
