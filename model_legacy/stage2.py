"""
model.stage2
============
Stage-2 (matrix + spectral) training logic.

train_epoch  One pass over training data using the blended matrix + spectral loss.

Spectral pipeline:
  decoded predictions -> diff_renderer_torch.simulate_batch -> Wasserstein-1 vs reference.

OOM guards
----------
* K guard   total spectral lines are counted from the numpy struct (CPU, no GPU cost)
            before any GPU tensor is allocated.  Batches where K > _MAX_SPEC_K skip
            the spectral term; matrix loss still supervises those samples.

* Chunking  Samples are rendered _SPEC_CHUNK at a time so peak broadening tensors
            are (1, K) not (B, K).  At K=1M and chunk=1 that is ~28 MB peak vs
            ~1.4 GB for chunk=B=51.

The caller (train.fit) is responsible for wrapping DataLoaders in a _Prefetcher
so that tensors arrive on the correct device.
"""

from __future__ import annotations

import time

import torch

from model_legacy import diff_renderer_torch as renderer
from model_legacy.losses import matrix_loss, spectral_loss
from model_legacy.schedules import curriculum_weights


# K = total spectral lines per molecule across all spin-manifold combos.
# With no connected-component splitting, K grows exponentially with group count and
# degeneracy.  For K > 1M, keeping k samples alive in the autograd graph requires
# ~28 * K * k bytes for broadening intermediates alone, which quickly exceeds VRAM.
# Skip the spectral loss for those batches; matrix loss still supervises them.
_MAX_SPEC_K = 1_000_000
_SPEC_CHUNK = 1   # one sample at a time to keep peak (1, K) tensors bounded


# ── Differentiable decode ────────────────────────────────────────────────────────

def decode_physical(pred, std):
    """Differentiably decode standardized model output to physical units (ppm / Hz)."""
    shifts = pred["shifts"] * std.shift_std + std.shift_mean
    jmag   = pred["j_mag"]  * std.j_std    + std.j_mean
    gate   = torch.sigmoid(pred["j_presence"])
    jmag   = jmag * gate
    B, G   = shifts.shape
    iu     = torch.triu_indices(G, G, 1, device=shifts.device)
    C      = torch.zeros(B, G, G, device=shifts.device, dtype=shifts.dtype)
    C[:, iu[0], iu[1]] = jmag
    C[:, iu[1], iu[0]] = jmag
    return {"shifts": shifts, "couplings": C}


# ── Spectral consistency term ────────────────────────────────────────────────────

def _spectral_term(pred_phys, batch, cfg, device):
    deg = batch["shared_degeneracy"]
    if deg is None:
        z = torch.zeros((), device=device)
        return z, z

    B        = batch["spectrum"].shape[0]
    k        = max(1, int(round(cfg.render_subset_frac * B)))
    sel      = torch.randperm(B, device=device)[:k]
    deg_list = [int(x) for x in deg.tolist()]
    struct   = renderer._structure(deg_list, device, pred_phys["shifts"].dtype)

    # Count spectral lines from the numpy struct (CPU, free) before any GPU allocation.
    total_k = sum(
        Fp.shape[0] * Fp.shape[1]
        for _, _, sb in struct["combos"]
        for _, (_, Fp) in sb["fplus"].items()
    )
    if total_k > _MAX_SPEC_K:
        z = torch.zeros((), device=device)
        return z, z

    chunk_losses, chunk_w1s = [], []
    for start in range(0, k, _SPEC_CHUNK):
        idx = sel[start : start + _SPEC_CHUNK]
        sub = {
            "shifts":    pred_phys["shifts"][idx].float(),
            "couplings": pred_phys["couplings"][idx].float(),
        }
        ref = batch["spectrum_ref"][idx]
        with torch.autocast(device_type="cuda", enabled=False):
            loss, w1 = spectral_loss(
                sub, ref, batch["degeneracy"][idx], cfg.field_low, renderer,
                struct=struct, points=cfg.points,
                ppm_from=cfg.ppm_from, ppm_to=cfg.ppm_to,
                linewidth_hz=cfg.linewidth_hz, eigh_eps=cfg.eigh_eps,
            )
        chunk_losses.append(loss)
        chunk_w1s.append(w1)

    return torch.stack(chunk_losses).mean(), torch.cat(chunk_w1s).mean()


# ── Training epoch ───────────────────────────────────────────────────────────────

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
    """One stage-2 training epoch.  Returns (epoch_metrics, global_step)."""
    model.train()
    bal = balance or {}
    w_mat, w_spec = curriculum_weights(
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
            if w_spec > 0:
                pred_phys        = decode_physical(pred, std)
                sloss, w1        = _spectral_term(pred_phys, batch, cfg, device)
                total            = total + w_spec * sloss
                comps["spectral_w1"] = w1.detach()

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
                "loss_shift":       float(comps.get("shift",       0.0)),
                "loss_jmag":        float(comps.get("jmag",        0.0)),
                "loss_presence":    float(comps.get("presence",    0.0)),
                "loss_deg":         float(comps.get("deg",         0.0)),
                "spectral_w1":      float(comps.get("spectral_w1", 0.0)),
                "lr":               float(sched.get_last_lr()[0]),
                "w_mat":            float(w_mat),
                "w_spec":           float(w_spec),
                "grad_norm":        gnorm,
                "seconds_per_step": step_secs,
                "amp_scale":        amp_scale,
            }
            if torch.cuda.is_available():
                step_metrics["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                step_metrics["cuda_reserved_gb"]  = torch.cuda.memory_reserved(device)  / 1e9
            diagnostics.log_metrics(
                split="train_step", epoch=epoch, step=step,
                metrics=step_metrics, extra={"stage": 2, "batch_idx": batch_idx},
            )

        step += 1

    n = max(1, len(batches))
    epoch_metrics = {k: v / n for k, v in running.items()} | {"w_mat": w_mat, "w_spec": w_spec}
    return epoch_metrics, step
