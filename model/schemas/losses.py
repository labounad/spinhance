"""
model.schemas.losses
====================
Typed loss output. Every loss returns a ``LossOutput``:

  total        scalar tensor suitable for ``.backward()``
  components   dict of differentiable per-term tensors (for inspection/weighting)
  metrics      dict of detached python floats (for logging — never in the graph)
  diagnostics  arbitrary debug data: skip reasons, counts, cost estimates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "LossOutput":
        assert torch.is_tensor(self.total), "total must be a tensor"
        assert self.total.dim() == 0, f"total must be scalar, got shape {tuple(self.total.shape)}"
        assert torch.isfinite(self.total).all(), "total is non-finite"
        for k, v in self.metrics.items():
            assert isinstance(v, float), f"metric '{k}' must be a python float, got {type(v)}"
        return self
