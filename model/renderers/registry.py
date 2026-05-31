"""Renderer registry. Renderers register here; losses/eval select by name."""
from __future__ import annotations

from model.registry import Registry

RENDERERS = Registry("renderer")


def build_renderer(name: str, **kwargs):
    return RENDERERS.build(name, **kwargs)
