"""
model.stage1
============
Stage-1 (matrix-only) training and validation logic.

train_epoch  One pass over training data using matrix loss only.
evaluate     One pass over validation data; returns matrix metrics.

Both functions accept any iterable as ``batches``.  The caller
(train.fit) is responsible for wrapping DataLoaders in a _Prefetcher
so that tensors arrive on the correct device.
"""

from __future__ import annotations

import time

import torch

from model_legacy.losses import matrix_loss
from model_legacy.metrics import compute_metrics
from model_legacy.schedules import curriculum_weights


def train_epoch(
    model,
    batches,
    opt,
    sched,
    scaler,
    cfg,
    std,
    epoch: int,
    device,
    amp_ctx,
    balance=None,
    diagnostics=None,
    global_step_start: int = 0,
) -> tuple[dict, int]:
    """One stage-1 training epoch.  Returns (epoch_metrics, global_step)."""
    model.train()
    bal = balance or {}
    w_mat, _ = curriculum_weights(
        epoch, cfg.stage1_epochs, cfg.ramp_epochs, cfg.spectral_max, cfg.matrix_anchor
    )
    running: dict = {}
    step = global_step_start

    for batch_idx, batch in enumerate(batches):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        with amp_ctx():
            pred = model(batch["spectrum"])
            mloss, comps = matrix_loss(
                pred, batch,
                weights=cfg.loss_weights,
                deg_class_weight=bal.get("deg_weights"),
                presence_pos_weight=bal.get("presence_pos_weight"),
            )
            total = w_mat * mloss

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            scaler.step(opt)
            scaler.update()
            amp_scale = float(scaler.get_scale())
        else:
            total.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            opt.step()
            amp_scale = 1.0

        sched.step()
        step_secs = time.time() - t0

        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + float(v)
        running["total"] = running.get("total", 0.0) + float(total.detach())

        if diagnostics is not None and step % cfg.log_every_steps == 0:
            step_metrics: dict = {
                "loss_total":       float(total.detach()),
                "loss_shift":       float(comps.get("shift",    0.0)),
                "loss_jmag":        float(comps.get("jmag",     0.0)),
                "loss_presence":    float(comps.get("presence", 0.0)),
                "loss_deg":         float(comps.get("deg",      0.0)),
                "spectral_w1":      0.0,
                "lr":               float(sched.get_last_lr()[0]),
                "w_mat":            float(w_mat),
                "w_spec":           0.0,
                "grad_norm":        gnorm,
                "seconds_per_step": step_secs,
                "amp_scale":        amp_scale,
            }
            if torch.cuda.is_available():
                step_metrics["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                step_metrics["cuda_reserved_gb"]  = torch.cuda.memory_reserved(device)  / 1e9
            diagnostics.log_metrics(
                split="train_step", epoch=epoch, step=step,
                metrics=step_metrics, extra={"stage": 1, "batch_idx": batch_idx},
            )

        step += 1

    n = max(1, len(batches))
    epoch_metrics = {k: v / n for k, v in running.items()} | {"w_mat": w_mat, "w_spec": 0.0}
    return epoch_metrics, step


@torch.no_grad()
def evaluate(model, batches, cfg, std, vocab, device, amp_ctx, balance=None) -> dict:
    """Validation pass.  Returns matrix metrics averaged over all batches."""
    model.eval()
    bal = balance or {}
    agg: dict = {}
    nb = 0

    for batch in batches:
        with amp_ctx():
            pred = model(batch["spectrum"])
            mloss, _ = matrix_loss(
                pred, batch,
                weights=cfg.loss_weights,
                deg_class_weight=bal.get("deg_weights"),
                presence_pos_weight=bal.get("presence_pos_weight"),
            )
        pred_np = {k: pred[k].float().cpu().numpy() for k in pred}
        tgt_np  = {k: batch[k].cpu().numpy()
                   for k in ("shifts", "j_mag", "j_presence", "deg_class")}
        met = compute_metrics(pred_np, tgt_np, std, vocab)
        met["matrix_loss"] = float(mloss)
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1

    return {k: v / max(1, nb) for k, v in agg.items()}
