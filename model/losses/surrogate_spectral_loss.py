"""
model.losses.surrogate_spectral_loss
====================================
Stage-2 spectral-consistency loss (Branch 6). Renders the model's PREDICTED spin
matrix through the frozen, pre-trained differentiable surrogate renderer
(Branch 5) and compares the resulting spectrum to the clean target spectrum with
Wasserstein-1 (+ a cosine term). This pulls the matrix prediction toward one
that actually *reproduces the observed spectrum*, complementing the supervised
matrix anchor.

Two impedance mismatches are handled here, so the loss stays a pure
``ModelOutput + SpinBatch -> LossOutput`` term:

  * **Standardized -> physical.** The model predicts in z-scored space; the
    surrogate is a physics model wanting ppm / Hz / proton counts. We invert the
    standardization (shift/J mean+std injected by the trainer) and turn the
    degeneracy class logits into a soft expected proton count
    (``softmax @ vocab``) so gradients flow.
  * **Presence gating.** Predicted J magnitudes are only meaningful where a
    coupling exists, so the physical coupling matrix is gated by the predicted
    presence probability (``sigmoid(presence_logits)``) and its diagonal zeroed.

The surrogate is frozen (``requires_grad_(False)``, ``eval()``) but NOT wrapped
in ``no_grad`` — gradients must flow *through* it to the matrix model. Use the
composite loss's ``start_epoch`` / ``ramp_epochs`` to ramp this in after the
matrix anchor has stabilized.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.evaluation.spectral_metrics import cosine_similarity, wasserstein1
from model.losses.base import Loss
from model.losses.registry import LOSSES
from model.renderers import build_renderer
from model.schemas import LossOutput, ModelOutput, SpinBatch
from model.schemas.constants import DEFAULT_DEG_VOCAB, PPM_FROM, PPM_TO


def _load_frozen_surrogate(checkpoint: str):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mcfg = {k: v for k, v in (ckpt.get("cfg", {}).get("model", {}) or {}).items()
            if k != "name"}
    model = build_renderer("surrogate", **mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@LOSSES.register("surrogate_spectral")
class SurrogateSpectralLoss(Loss):
    name = "surrogate_spectral"

    def __init__(self, checkpoint: str, field: int = 90,
                 w1_weight: float = 1.0, cosine_weight: float = 0.5,
                 use_clean_target: bool = True,
                 shift_mean: float = 0.0, shift_std: float = 1.0,
                 j_mean: float = 0.0, j_std: float = 1.0,
                 deg_vocab=DEFAULT_DEG_VOCAB):
        self.surrogate = _load_frozen_surrogate(checkpoint)
        self.field = float(field)
        self.w1_weight = float(w1_weight)
        self.cosine_weight = float(cosine_weight)
        self.use_clean_target = bool(use_clean_target)
        self.shift_mean, self.shift_std = float(shift_mean), float(shift_std)
        self.j_mean, self.j_std = float(j_mean), float(j_std)
        self._vocab = torch.tensor([float(v) for v in deg_vocab])
        self.points = int(getattr(self.surrogate, "points", 16384))
        self.dx = (PPM_TO - PPM_FROM) / self.points
        self._device = None

    def _to(self, device):
        if self._device != device:
            self.surrogate = self.surrogate.to(device)
            self._vocab = self._vocab.to(device)
            self._device = device

    def _physical_matrix(self, output: ModelOutput):
        """Invert standardization + presence-gate -> (shifts ppm, couplings Hz, deg protons).

        Everything is cast to float32: the frozen surrogate is a physics model
        with FFT-based broadening that must not run in the matrix model's bf16/
        fp16 autocast dtype (FFT/index_add reject half precision)."""
        shifts = output.shifts.float() * self.shift_std + self.shift_mean        # (B,G) ppm
        jmag = output.coupling_matrix().float() * self.j_std + self.j_mean       # (B,G,G) Hz
        pres = torch.sigmoid(output.presence_matrix().float())                  # (B,G,G) prob
        couplings = jmag * pres                                                 # soft physical Hz
        G = couplings.shape[-1]
        diag = torch.eye(G, device=couplings.device, dtype=couplings.dtype)
        couplings = couplings * (1.0 - diag)                                    # zero diagonal
        couplings = 0.5 * (couplings + couplings.transpose(-1, -2))             # enforce symmetry
        deg = F.softmax(output.degeneracy_logits.float(), dim=-1) @ self._vocab.float()  # (B,G)
        return shifts, couplings, deg

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:
        self._to(output.shifts.device)
        target = batch.spectrum_ref if (self.use_clean_target and batch.spectrum_ref is not None) \
            else batch.spectrum
        target = target.to(output.shifts.device)

        # Run the frozen surrogate (and the spectral metrics) in float32 with
        # autocast disabled — its FFT broadening can't run in bf16/fp16. Gradients
        # still flow back to the (autocast) matrix model through the float cast.
        dev_type = output.shifts.device.type
        with torch.autocast(device_type=dev_type, enabled=False):
            shifts, couplings, deg = self._physical_matrix(output)
            target = target.float()
            pred = self.surrogate(shifts, couplings, deg, self.field)        # (B,P) unit integral
            w1 = wasserstein1(pred, target, dx=self.dx).mean()
            cos = cosine_similarity(pred, target).mean()
            total = self.w1_weight * w1 + self.cosine_weight * (1.0 - cos)

        return LossOutput(
            total=total,
            components={"w1": w1.detach(), "cos_term": (1.0 - cos).detach()},
            metrics={"w1": float(w1.detach()), "cosine": float(cos.detach()),
                     "field": self.field},
        )
