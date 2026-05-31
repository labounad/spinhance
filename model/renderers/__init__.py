"""
Spin-system parameters -> spectrum/summary. Importing this package registers the
built-in renderers (exact_no_grad, exact_autograd_experimental).
"""
from model.renderers.registry import RENDERERS, build_renderer
from model.renderers.base import Renderer

from model.renderers import exact as _exact          # noqa: F401  (registers)
from model.renderers import surrogate as _surrogate  # noqa: F401  (registers)

__all__ = ["RENDERERS", "build_renderer", "Renderer"]
