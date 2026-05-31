"""
model.training.schedules
========================
Torch-free schedules (ported from legacy):
  curriculum_weights  Stage-1 -> Stage-2 loss blend (for composite ramps)
  lr_factor           linear warmup then cosine decay (LambdaLR multiplier)
"""
from __future__ import annotations

import math

__all__ = ["curriculum_weights", "lr_factor"]


def curriculum_weights(epoch, stage1_epochs=20, ramp_epochs=10,
                       spectral_max=1.0, matrix_anchor=0.3):
    if epoch < stage1_epochs:
        return 1.0, 0.0
    if ramp_epochs <= 0:
        return matrix_anchor, spectral_max
    frac = min(1.0, (epoch - stage1_epochs) / ramp_epochs)
    return 1.0 + (matrix_anchor - 1.0) * frac, spectral_max * frac


def lr_factor(step, warmup_steps, total_steps, min_factor=0.05):
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    prog = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return min_factor + (1.0 - min_factor) * cos
