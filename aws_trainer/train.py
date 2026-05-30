"""
aws_trainer.train
==================
Production training loop.  Key improvements over model/train.py:

  * DDP via torchrun (LOCAL_RANK / WORLD_SIZE env vars)
  * EMA: shadow model updated every step; validated/checkpointed from EMA
  * torch.amp.autocast + GradScaler (non-deprecated PyTorch 2.x API)
  * Gradient accumulation (Stage 1 only; Stage 2 keeps accum_steps=1)
  * torch.compile support
  * Hungarian-matched eval metrics (h_shift_mae, h_j_mae, h_deg_acc)
  * Richer checkpoints: best + periodic + last

Run:
    Single GPU:  PYTHONPATH=. python -m aws_trainer.run ...
    Multi-GPU:   PYTHONPATH=. torchrun --nproc_per_node=4 -m aws_trainer.run ...
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import diff_renderer_torch as renderer
from model.dataset import collate_fn
from model.losses import matrix_loss, spectral_loss
from model.metrics import compute_metrics, decode
from model.schedules import curriculum_weights, lr_factor
from model.targets import (DegeneracyVocab, Standardizer, class_balance)
from aws_trainer.config import VAWSConfig
from aws_trainer.dataset import (CachedSpectrumDataset, DistributedBucketSampler,
                                  SpectraCache)
from aws_trainer.ema import EMA
from aws_trainer.logging_utils import RunLogger
from aws_trainer.model import build_model, param_count


# ── DDP helpers ───────────────────────────────────────────────────────────────

def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _is_ddp() -> bool:
    return _world_size() > 1


def _is_main() -> bool:
    return _local_rank() == 0


def _setup_ddp() -> None:
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(_local_rank())


def _all_reduce_mean(metrics: dict, device) -> dict:
    import torch.distributed as dist
    if not dist.is_initialized():
        return metrics
    ws = dist.get_world_size()
    out = {}
    for k, v in metrics.items():
        t = torch.tensor(float(v), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = float(t) / ws
    return out


# ── AMP helpers ───────────────────────────────────────────────────────────────

def _make_amp(cfg: VAWSConfig, device_type: str):
    if cfg.amp_dtype == "none" or device_type == "cpu":
        return contextlib.nullcontext, None
    dt = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    ctx = lambda: torch.amp.autocast(device_type=device_type, dtype=dt)
    scaler = (torch.amp.GradScaler(device_type)
               if dt == torch.float16 else None)
    return ctx, scaler


# ── Physical decode (differentiable; used for Stage-2 spectral loss) ──────────

def _decode_physical(pred: dict, std: Standardizer) -> dict:
    shifts = pred["shifts"] * std.shift_std + std.shift_mean
    jmag = pred["j_mag"] * std.j_std + std.j_mean
    gate = torch.sigmoid(pred["j_presence"])
    jmag = jmag * gate
    B, G = shifts.shape
    iu = torch.triu_indices(G, G, 1, device=shifts.device)
    C = torch.zeros(B, G, G, device=shifts.device, dtype=shifts.dtype)
    C[:, iu[0], iu[1]] = jmag
    C[:, iu[1], iu[0]] = jmag
    return {"shifts": shifts, "couplings": C}


# ── Hungarian-matched eval metrics ────────────────────────────────────────────

def _hungarian_metrics(pred_np: dict, tgt_np: dict, std: Standardizer,
                       vocab: DegeneracyVocab) -> dict:
    """Compute shift/J/deg metrics under the best S_8 permutation (shift cost)."""
    from scipy.optimize import linear_sum_assignment
    dec = decode(pred_np, std, vocab)
    G = dec["shifts"].shape[1]
    tgt_shifts = std.inverse_shifts(tgt_np["shifts"])
    tgt_present = tgt_np["j_presence"] > 0.5
    tgt_jmag = std.inverse_j(tgt_np["j_mag"]) * tgt_present
    tgt_deg = np.stack([vocab.from_index(tgt_np["deg_class"][b])
                        for b in range(tgt_np["deg_class"].shape[0])])
    iu = np.triu_indices(G, 1)
    B = dec["shifts"].shape[0]
    h_shift, h_j_vals, h_deg = 0.0, [], 0.0
    for b in range(B):
        cost = np.abs(dec["shifts"][b, :, None] - tgt_shifts[b, None, :])
        _, col = linear_sum_assignment(cost)
        inv_col = np.argsort(col)
        a_shifts = dec["shifts"][b][inv_col]
        a_coup   = dec["couplings"][b][np.ix_(inv_col, inv_col)]
        a_deg    = dec["degeneracy"][b][inv_col]
        h_shift += float(np.abs(a_shifts - tgt_shifts[b]).mean())
        pred_j  = a_coup[iu]; gt_j = tgt_jmag[b]
        gt_pres = tgt_present[b]
        if gt_pres.any():
            h_j_vals.append(float(np.abs(pred_j[gt_pres] - gt_j[gt_pres]).mean()))
        h_deg += float((a_deg == tgt_deg[b]).mean())
    return {
        "h_shift_mae_ppm": h_shift / B,
        "h_j_mae_hz":      float(np.mean(h_j_vals)) if h_j_vals else float("nan"),
        "h_deg_acc":       h_deg / B,
    }


# ── Dataset / loader builders ─────────────────────────────────────────────────

def _build_loaders(records: list[dict], assignment: dict, cfg: VAWSConfig,
                   std: Standardizer, vocab: DegeneracyVocab, device: torch.device):
    by_fold: dict = {"train": [], "val": [], "test": []}
    for r in records:
        f = assignment.get(r["mol_id"])
        if f:
            by_fold[f].append(r)

    rank, ws = _local_rank(), _world_size()
    aug_kw = dict(noise_sigma_frac=cfg.aug_noise_frac,
                  max_ref_shift_ppm=cfg.aug_ref_shift_ppm,
                  baseline_amp_frac=cfg.aug_baseline_amp_frac)

    cache90 = None
    if cfg.preload_spectra and _is_main():
        cache90 = SpectraCache(by_fold["train"] + by_fold["val"] + by_fold["test"],
                               field=int(cfg.field_low), verbose=True)
    if cache90 is not None:
        n_train = len(by_fold["train"])
        n_val   = len(by_fold["val"])
        tr_cache  = SpectraCache.slice(cache90, 0)
        val_cache = SpectraCache.slice(cache90, n_train)
    else:
        tr_cache = val_cache = None

    def _ds(recs, aug, cache):
        return CachedSpectrumDataset(
            recs, vocab, std, cache=cache,
            spectrum_field=cfg.spectrum_field, augment=aug,
            ppm_from=cfg.ppm_from, ppm_to=cfg.ppm_to,
            aug_kwargs=aug_kw, seed=cfg.seed)

    tr_ds  = _ds(by_fold["train"], True, tr_cache)
    val_ds = _ds(by_fold["val"], False, val_cache)

    pin = (device.type == "cuda")
    # When the cache is in RAM there is no I/O to parallelize; workers would
    # only add forkserver pickle overhead (Python 3.14+ default on Linux).
    nw = 0 if cache90 is not None else cfg.num_workers

    # Stage 1 — plain shuffled loader (DDP via DistributedSampler)
    if _is_ddp():
        from torch.utils.data import DistributedSampler
        tr_samp1 = DistributedSampler(tr_ds, shuffle=True, seed=cfg.seed)
    else:
        tr_samp1 = None
    stage1_loader = DataLoader(
        tr_ds, batch_size=cfg.batch_size, sampler=tr_samp1,
        shuffle=(tr_samp1 is None), collate_fn=collate_fn,
        num_workers=nw, pin_memory=pin, drop_last=True, persistent_workers=(nw > 0))

    # Stage 2 — bucketed loader (each batch single degeneracy-pattern)
    stage2_sampler = DistributedBucketSampler(
        tr_ds.bucket_keys, cfg.batch_size, rank=rank, world_size=ws,
        shuffle=True, seed=cfg.seed)
    stage2_loader = DataLoader(
        tr_ds, batch_sampler=stage2_sampler, collate_fn=collate_fn,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=nw, pin_memory=pin)

    return stage1_loader, stage2_loader, val_loader, tr_samp1, stage2_sampler


# ── Spectral loss helper ───────────────────────────────────────────────────────

def _spectral_term(pred_phys: dict, batch: dict, cfg: VAWSConfig,
                   device: torch.device) -> tuple:
    deg = batch["shared_degeneracy"]
    if deg is None:
        z = torch.zeros((), device=device)
        return z, z
    B = batch["spectrum"].shape[0]
    k = max(1, int(round(cfg.render_subset_frac * B)))
    sel = torch.randperm(B, device=device)[:k]
    deg_list = [int(x) for x in deg.tolist()]
    struct = renderer._structure(deg_list, device, pred_phys["shifts"].dtype)
    sub = {"shifts": pred_phys["shifts"][sel],
           "couplings": pred_phys["couplings"][sel]}
    ref = batch["spectrum_ref"][sel]
    loss, w1 = spectral_loss(
        sub, ref, batch["degeneracy"][sel], cfg.field_low, renderer,
        struct=struct, points=cfg.points, ppm_from=cfg.ppm_from, ppm_to=cfg.ppm_to,
        linewidth_hz=cfg.linewidth_hz, eigh_eps=cfg.eigh_eps)
    return loss, w1.mean()


# ── Train epoch ───────────────────────────────────────────────────────────────

def _train_epoch(model: nn.Module, loader: DataLoader, opt, sched, scaler,
                 cfg: VAWSConfig, std: Standardizer, epoch: int,
                 device: torch.device, amp_ctx, balance: dict,
                 accum: int) -> dict:
    model.train()
    w_mat, w_spec = curriculum_weights(
        epoch, cfg.stage1_epochs, cfg.ramp_epochs, cfg.spectral_max, cfg.matrix_anchor)
    running: dict = {}
    stage = 1 if epoch < cfg.stage1_epochs else 2

    steps = 0
    opt.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        is_accum_step = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == len(loader))

        # Stage 1 uses gradient accumulation; Stage 2 does not
        use_accum = (stage == 1) and (accum > 1)
        no_sync_ctx = (model.no_sync()
                       if (_is_ddp() and use_accum and not is_accum_step)
                       else contextlib.nullcontext())

        with no_sync_ctx:
            with amp_ctx():
                pred = model(batch["spectrum"])
                mloss, comps = matrix_loss(
                    pred, batch, weights=cfg.loss_weights,
                    deg_class_weight=balance.get("deg_weights"),
                    presence_pos_weight=balance.get("presence_pos_weight"))
                total = w_mat * mloss
                if w_spec > 0:
                    pred_phys = _decode_physical(pred, std)
                    sloss, w1 = _spectral_term(pred_phys, batch, cfg, device)
                    total = total + w_spec * sloss
                    comps["spectral_w1"] = w1.detach()
                if use_accum:
                    total = total / accum

            if scaler is not None:
                scaler.scale(total).backward()
            else:
                total.backward()

        if is_accum_step:
            if scaler is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            steps += 1

        scale = accum if use_accum else 1
        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + float(v) * scale
        running["total"] = running.get("total", 0.0) + float(total.detach()) * scale

    n = max(1, steps)
    return {k: v / n for k, v in running.items()} | {"w_mat": w_mat, "w_spec": w_spec}


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, cfg: VAWSConfig,
              std: Standardizer, vocab: DegeneracyVocab,
              device: torch.device, amp_ctx, balance: dict) -> dict:
    model.eval()
    agg: dict = {}
    nb = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with amp_ctx():
            pred = model(batch["spectrum"])
            mloss, _ = matrix_loss(
                pred, batch, weights=cfg.loss_weights,
                deg_class_weight=balance.get("deg_weights"),
                presence_pos_weight=balance.get("presence_pos_weight"))
        pred_np = {k: pred[k].float().cpu().numpy() for k in pred}
        tgt_np  = {k: batch[k].cpu().numpy()
                   for k in ("shifts", "j_mag", "j_presence", "deg_class")}
        met = compute_metrics(pred_np, tgt_np, std, vocab)
        met.update(_hungarian_metrics(pred_np, tgt_np, std, vocab))
        met["matrix_loss"] = float(mloss)
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    result = {k: v / max(1, nb) for k, v in agg.items()}
    return _all_reduce_mean(result, device)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _save(path: str | Path, model: nn.Module, ema: EMA, std: Standardizer,
          cfg: VAWSConfig, epoch: int, best_score: float, metrics: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    src = model.module if hasattr(model, "module") else model
    torch.save({
        "model":       src.state_dict(),           # compatible with model/gui.py
        "ema_model":   ema.state_dict(),
        "standardizer": vars(std),
        "cfg":         asdict(cfg),
        "epoch":       epoch,
        "best_score":  best_score,
        "metrics":     metrics,
    }, path)


def _load_ema_weights(path: str | Path, model: nn.Module, std: Standardizer,
                      device: torch.device) -> None:
    """Load EMA weights from checkpoint into model for inference."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    key = "ema_model" if "ema_model" in ckpt else "model"
    src = model.module if hasattr(model, "module") else model
    src.load_state_dict(ckpt[key])
    sd = ckpt["standardizer"]
    std.shift_mean = sd["shift_mean"]; std.shift_std = sd["shift_std"]
    std.j_mean     = sd["j_mean"];     std.j_std     = sd["j_std"]


# ── Main fit function ─────────────────────────────────────────────────────────

def fit(records: list[dict], assignment: dict, cfg: VAWSConfig) -> tuple:
    """Full training run.  Call from run.py or directly.
    Returns (model, ema, std, vocab) on rank 0; (None, None, None, None) on other ranks."""

    if _is_ddp():
        _setup_ddp()

    device_id = _local_rank() if (_is_ddp() and torch.cuda.is_available()) else None
    if device_id is not None:
        device = torch.device("cuda", device_id)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device_type = device.type

    torch.manual_seed(cfg.seed + _local_rank())

    # Build vocab + standardizer from TRAIN records only
    vocab = DegeneracyVocab()
    train_recs = [r for r in records if assignment.get(r["mol_id"]) == "train"]
    std = Standardizer().fit(train_recs, vocab)
    if _is_main():
        print(f"standardizer: shift {std.shift_mean:.2f}±{std.shift_std:.2f} ppm | "
              f"J {std.j_mean:.2f}±{std.j_std:.2f} Hz", flush=True)

    # Class balancing
    cb = class_balance(train_recs, vocab)
    balance = {
        "deg_weights":          torch.tensor(cb["deg_weights"], device=device),
        "presence_pos_weight":  torch.tensor(cb["presence_pos_weight"], device=device),
    }
    if _is_main():
        print(f"class balance: presence_pos_weight={cb['presence_pos_weight']:.2f} "
              f"deg_counts={cb['deg_counts'].tolist()}", flush=True)

    # Build model
    model = build_model(cfg).to(device)
    if _is_main():
        print(f"model: {cfg.model_size} {param_count(model)} | device {device} | "
              f"stage2 {'ON' if cfg.stage2_enabled else 'OFF'}", flush=True)

    if cfg.compile_model:
        model = torch.compile(model)

    if _is_ddp():
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])

    ema = EMA(model.module if hasattr(model, "module") else model,
               decay=cfg.ema_decay)
    ema.to(device)

    # Data
    stage1_loader, stage2_loader, val_loader, samp1, samp2 = _build_loaders(
        records, assignment, cfg, std, vocab, device)

    # Optimizer + schedule
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(stage1_loader))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, warmup, total_steps, cfg.min_lr_frac))

    amp_ctx, scaler = _make_amp(cfg, device_type)
    logger = RunLogger(cfg, is_main=_is_main())

    if _is_main():
        ckpt_dir = Path(cfg.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best.pt"
        last_path = ckpt_dir / "last.pt"

    best_score = float("inf")
    bad_epochs = 0
    will_stage2 = cfg.stage2_enabled and (cfg.stage1_epochs < cfg.epochs)

    for epoch in range(cfg.epochs):
        # Stage 2 handoff: reset patience window (objective changes)
        if epoch == cfg.stage1_epochs and will_stage2:
            bad_epochs = 0

        # Set epoch for DDP / bucket samplers
        stage = 1 if epoch < cfg.stage1_epochs else 2
        loader = stage1_loader if stage == 1 else stage2_loader
        if stage == 1 and samp1 is not None:
            samp1.set_epoch(epoch)
        elif stage == 2:
            samp2.set_epoch(epoch)

        accum = cfg.accum_steps if stage == 1 else 1
        tr = _train_epoch(model, loader, opt, sched, scaler, cfg, std, epoch,
                          device, amp_ctx, balance, accum)
        ema.update(model)

        va = _evaluate(ema.shadow, val_loader, cfg, std, vocab, device, amp_ctx, balance)

        if _is_main():
            logger.print_epoch(epoch, stage, tr["w_mat"], tr["w_spec"], tr, va)
            logger.log(epoch, {"stage": stage, **{f"tr_{k}": v for k, v in tr.items()},
                                **{f"va_{k}": v for k, v in va.items()}})

        score = va.get("shift_mae_ppm", 1e9) + va.get("j_mae_hz", 10.0) / 10.0
        earlystop_active = (epoch >= cfg.stage1_epochs) or (not will_stage2)

        if _is_main():
            if score < best_score:
                best_score = score
                bad_epochs = 0
                _save(best_path, model, ema, std, cfg, epoch, best_score, va)
            elif earlystop_active:
                bad_epochs += 1
            _save(last_path, model, ema, std, cfg, epoch, best_score, va)
            if cfg.save_every > 0 and (epoch + 1) % cfg.save_every == 0:
                _save(ckpt_dir / f"epoch_{epoch:04d}.pt",
                      model, ema, std, cfg, epoch, best_score, va)

        # Early stop check (broadcast from rank 0)
        if _is_ddp():
            import torch.distributed as dist
            stop_flag = torch.tensor(
                int(earlystop_active and bad_epochs >= cfg.patience), device=device)
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(stop_flag.item())
        else:
            should_stop = earlystop_active and (bad_epochs >= cfg.patience)

        if should_stop:
            if _is_main():
                print(f"early stop at epoch {epoch}", flush=True)
            break

    if _is_main():
        logger.finish()
        print(f"DONE  best checkpoint → {best_path}", flush=True)
        return model, ema, std, vocab
    return None, None, None, None
