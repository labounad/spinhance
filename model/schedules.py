"""
model.schedules
===================
Torch-free schedules (so they can be unit-tested without torch):
  * curriculum_weights  - Stage-1 -> Stage-2 loss blend (Decision 7)
  * lr_factor           - linear warmup then cosine decay (multiplier for LambdaLR)
"""

from __future__ import annotations

import math

__all__ = ["curriculum_weights", "lr_factor"]


def curriculum_weights(epoch, stage1_epochs=20, ramp_epochs=10,
                       spectral_max=1.0, matrix_anchor=0.3):
    """(w_matrix, w_spectral) for a given epoch (curriculum blend).

    epoch < stage1_epochs           : matrix only (w_spectral = 0, w_matrix = 1).
    ramp window (ramp_epochs long)  : w_spectral ramps 0 -> spectral_max, while
                                      w_matrix decays 1 -> matrix_anchor (kept as
                                      an anchor against identifiability drift).
    after the ramp                  : (matrix_anchor, spectral_max).
    """
    if epoch < stage1_epochs:
        return 1.0, 0.0
    if ramp_epochs <= 0:
        return matrix_anchor, spectral_max
    frac = min(1.0, (epoch - stage1_epochs) / ramp_epochs)
    w_spectral = spectral_max * frac
    w_matrix = 1.0 + (matrix_anchor - 1.0) * frac
    return w_matrix, w_spectral


def lr_factor(step, warmup_steps, total_steps, min_factor=0.05):
    """Multiplicative LR factor: linear warmup then cosine decay to min_factor."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_factor + (1.0 - min_factor) * cos
