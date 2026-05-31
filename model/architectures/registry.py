"""Architecture registry. Models register here; configs select by name."""
from __future__ import annotations

from model.registry import Registry

ARCHITECTURES = Registry("architecture")


def build_architecture(name: str, **kwargs):
    return ARCHITECTURES.build(name, **kwargs)
