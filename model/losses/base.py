"""
model.losses.base
=================
Loss interface: ``ModelOutput + SpinBatch -> LossOutput``. A loss never touches
the optimizer, checkpoints, or the model — it is pure, so it can be tested in
isolation on tiny synthetic tensors.
"""
from __future__ import annotations

from model.schemas import LossOutput, ModelOutput, SpinBatch


class Loss:
    name: str = "loss"

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:  # pragma: no cover
        raise NotImplementedError

    def set_epoch(self, epoch: int) -> None:
        """Hook for schedule-aware losses (e.g. composite ramps). Default no-op."""
        return None
