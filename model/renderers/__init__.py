"""Spin-system parameters -> spectrum/summary. Import a concrete module to register it."""
from model.renderers.registry import RENDERERS, build_renderer

__all__ = ["RENDERERS", "build_renderer"]
