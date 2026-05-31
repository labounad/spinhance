"""
model.architectures.base
========================
Base class for architectures. The contract is narrow: consume a ``SpinBatch``
(or a raw spectrum tensor) and return a ``ModelOutput``. Architectures never
compute losses or know which loss is in use.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.schemas import ModelOutput, SpinBatch


class SpinArchitecture(nn.Module):
    """Spectrum -> ModelOutput. Subclasses implement ``forward``."""

    @staticmethod
    def spectrum_of(x) -> torch.Tensor:
        """Accept either a SpinBatch or a raw (B, P) spectrum tensor."""
        if isinstance(x, SpinBatch):
            return x.spectrum
        return x

    def forward(self, x) -> ModelOutput:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
