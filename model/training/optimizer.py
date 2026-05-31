"""Optimizer + LR schedule construction."""
from __future__ import annotations

import torch

from model.training.schedules import lr_factor


def build_optimizer_and_scheduler(model, lr, weight_decay, warmup_frac,
                                  steps_per_epoch, epochs,
                                  min_factor=0.05, stable_frac=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup = int(warmup_frac * total_steps)
    stable = int(stable_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, warmup, total_steps,
                                 min_factor=min_factor, stable_steps=stable))
    return opt, sched


def amp_context(amp: str, device: str):
    """Return (context_factory, grad_scaler)."""
    import contextlib
    if amp == "none" or device == "cpu":
        return (lambda: contextlib.nullcontext()), None
    dt = torch.bfloat16 if amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if dt == torch.float16 else None
    return (lambda: torch.autocast(device_type="cuda", dtype=dt)), scaler
