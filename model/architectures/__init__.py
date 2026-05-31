"""Spectrum -> ModelOutput architectures. Import a concrete module to register it."""
from model.architectures.registry import ARCHITECTURES, build_architecture

__all__ = ["ARCHITECTURES", "build_architecture"]
