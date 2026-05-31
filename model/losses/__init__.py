"""
ModelOutput + SpinBatch -> LossOutput. Importing this package registers the
built-in losses (matrix, hungarian, surrogate_spectral, composite).
"""
from model.losses.registry import LOSSES, build_loss
from model.losses.base import Loss

from model.losses import matrix_loss as _matrix        # noqa: F401
from model.losses import hungarian_loss as _hungarian  # noqa: F401
from model.losses import surrogate_spectral_loss as _surrogate_spectral  # noqa: F401
from model.losses import composite as _composite       # noqa: F401
from model.losses.composite import build_composite

__all__ = ["LOSSES", "build_loss", "Loss", "build_composite"]
