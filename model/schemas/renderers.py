"""
model.schemas.renderers
=======================
Typed renderer output. A renderer maps spin-system parameters to a spectrum (or a
spectral summary). It must report whether it actually rendered or skipped, plus
cost diagnostics — this is the contract that keeps the exact-autograd renderer
honest about memory.

  spectrum     (B, P) or None when skipped / summary-only
  lines        optional line-list representation (centers/amps), renderer-specific
  metrics      detached python floats (runtime, etc.)
  diagnostics  rendered/skipped, n_rendered, n_skipped, skip reasons, cost proxies
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RendererOutput:
    spectrum: torch.Tensor | None
    lines: Any | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def rendered(self) -> bool:
        return bool(self.diagnostics.get("rendered", self.spectrum is not None))

    @property
    def skipped(self) -> bool:
        return not self.rendered

    def validate(self) -> "RendererOutput":
        if self.spectrum is not None:
            assert torch.is_tensor(self.spectrum), "spectrum must be a tensor or None"
            assert self.spectrum.dim() == 2, f"spectrum must be (B, P), got {tuple(self.spectrum.shape)}"
        return self
