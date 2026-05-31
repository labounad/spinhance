"""Loss registry. Losses register here; composite/config select by name."""
from __future__ import annotations

from model.registry import Registry

LOSSES = Registry("loss")


def build_loss(name: str, **kwargs):
    return LOSSES.build(name, **kwargs)
