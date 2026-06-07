"""
model.training.loops
====================
Train and evaluation epoch loops over ``SpinBatch`` loaders. No loss math or
renderer internals live here — the loop calls ``loss_fn(output, batch)`` and the
metrics helper, and emits step-level diagnostics.
"""
from __future__ import annotations

import math
import time

import torch

from model.evaluation.metrics import evaluate_output

_SPIKE_FLOOR = 50.0   # never treat a grad norm below this as a spike (healthy norms ~5-30)


def grad_step_accept(gnorm: float, gnorm_ema: float | None, spike_factor: float,
                     floor: float = _SPIKE_FLOOR) -> bool:
    """Whether to APPLY an optimizer step given its (post-clip) grad norm. Reject when the
    norm is non-finite OR a sharp spike above the running EMA. clip_grad_norm_ rescales the
    GLOBAL gradient vector, so it cannot damp a single runaway parameter (and cannot sanitize
    a NaN at all) — a finite blow-up then cascades and kills the run (observed: 3M at ep6,
    grad 9 -> 703 -> ... -> 6.9e18 at the correct LR). Skipping the offending batch rides
    through it. In DDP the grad is all-reduced in backward(), so gnorm — and this decision —
    is identical across ranks, keeping them in lockstep."""
    if not math.isfinite(gnorm):
        return False
    if spike_factor > 0 and gnorm_ema is not None and gnorm > max(spike_factor * gnorm_ema, floor):
        return False
    return True


def train_epoch(model, loader, loss_fn, opt, sched, scaler, amp_ctx, device,
                *, epoch, global_step, grad_clip, log_every_steps, stage,
                grad_spike_factor=0.0, diagnostics=None):
    model.train()
    running: dict[str, float] = {}
    n_batches = 0
    step = global_step
    gnorm_ema: float | None = None   # running EMA of accepted grad norms, for spike detection
    n_skipped = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            out = model(batch)
            lo = loss_fn(out, batch)
        total = lo.total
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            accept = grad_step_accept(gnorm, gnorm_ema, grad_spike_factor)
            if accept:
                scaler.step(opt)
            scaler.update()
        else:
            total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            accept = grad_step_accept(gnorm, gnorm_ema, grad_spike_factor)
            if accept:
                opt.step()
        if accept:
            gnorm_ema = gnorm if gnorm_ema is None else 0.98 * gnorm_ema + 0.02 * gnorm
        else:
            n_skipped += 1
            if diagnostics is not None:
                diagnostics.log_event("grad_skip", {"epoch": epoch, "step": step,
                                                    "grad_norm": gnorm, "ema": gnorm_ema})
        sched.step()

        running["total"] = running.get("total", 0.0) + float(total.detach())
        for k, v in lo.metrics.items():
            running[k] = running.get(k, 0.0) + v
        n_batches += 1

        if diagnostics is not None and step % log_every_steps == 0:
            sm = {"loss_total": float(total.detach()), "lr": float(sched.get_last_lr()[0]),
                  "grad_norm": gnorm, "seconds_per_step": time.time() - t0, **lo.metrics}
            if torch.cuda.is_available():
                sm["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                sm["cuda_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            diagnostics.log_metrics(split="train_step", epoch=epoch, step=step,
                                    metrics=sm, extra={"stage": stage})
        step += 1

    n = max(1, n_batches)
    return {k: v / n for k, v in running.items()}, step


@torch.no_grad()
def evaluate(model, loader, loss_fn, standardizer, vocab, amp_ctx, device):
    model.eval()
    agg: dict[str, float] = {}
    nb = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with amp_ctx():
            out = model(batch)
            lo = loss_fn(out, batch)
        met = evaluate_output(out, batch, standardizer, vocab)
        met["loss_total"] = float(lo.total.detach())
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {k: v / max(1, nb) for k, v in agg.items()}
