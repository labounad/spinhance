"""
model.training.loops
====================
Train and evaluation epoch loops over ``SpinBatch`` loaders. No loss math or
renderer internals live here — the loop calls ``loss_fn(output, batch)`` and the
metrics helper, and emits step-level diagnostics.
"""
from __future__ import annotations

import time

import torch

from model.evaluation.metrics import evaluate_output


def train_epoch(model, loader, loss_fn, opt, sched, scaler, amp_ctx, device,
                *, epoch, global_step, grad_clip, log_every_steps, stage,
                diagnostics=None):
    model.train()
    running: dict[str, float] = {}
    n_batches = 0
    step = global_step

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
            scaler.step(opt); scaler.update()
        else:
            total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            opt.step()
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
