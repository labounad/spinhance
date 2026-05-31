"""ModelOutput + SpinBatch -> LossOutput. Import a concrete module to register it."""
from model.losses.registry import LOSSES, build_loss

__all__ = ["LOSSES", "build_loss"]
