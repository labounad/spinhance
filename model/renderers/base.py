"""
model.renderers.base
====================
Renderer interface: spin-system parameters -> spectrum (or summary), returned as
a typed ``RendererOutput`` that always reports whether it rendered or skipped,
plus cost diagnostics. A renderer never computes the model loss — a loss class
wraps it if needed.
"""
from __future__ import annotations

from model.schemas import RendererOutput


class Renderer:
    name: str = "renderer"

    def render(self, shifts, couplings, degeneracy, field_mhz, **kwargs) -> RendererOutput:  # pragma: no cover
        raise NotImplementedError
