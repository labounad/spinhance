"""
aws_trainer.ema
================
Exponential moving average of model weights.  Validated and checkpointed as the
production model; the live model is the training model.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Unwrap DDP if needed
        src = model.module if hasattr(model, "module") else model
        for s, m in zip(self.shadow.parameters(), src.parameters()):
            s.data.lerp_(m.data, 1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.shadow.load_state_dict(sd)

    def to(self, device) -> "EMA":
        self.shadow = self.shadow.to(device)
        return self
