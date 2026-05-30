"""
model.train
==============
Training loop for Task 4, wiring together the verified components:

  splits.make_splits  -> folds (Decision 8)
  targets/dataset     -> standardized batches + bucketed sampler (Decisions 3,4,7)
  model.SpinHanceModel-> ResNet-1D + 4 heads (Decisions 1,2)
  losses.matrix_loss  -> Stage 1 (Decision 4)
  losses.spectral_loss + diff_renderer_torch -> Stage 2 (Decisions 5,6)
  schedules.curriculum_weights / lr_factor -> handoff + LR (Decision 7)
  metrics.compute_metrics -> physical-unit eval

Stage logic (curriculum blend):
  * epoch < stage1_epochs : matrix loss only, plain shuffled loader.
  * after that            : matrix (decayed anchor) + ramped spectral loss, using
                            the bucketed loader so the renderer builds one operator
                            ``struct`` per batch and renders only a random subset
                            (stochastic subset, Decision 7).

This module is torch and runs in your env (no torch in the prototyping sandbox);
all numeric cores it calls are unit-tested in the test_*.py files. A tiny
synthetic end-to-end smoke test is under ``python3 -m model.train --smoke``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import diff_renderer_torch as renderer
from model.dataset import (SpectrumMatrixDataset, BucketByDegeneracySampler,
                              collate_fn)
from model.losses import matrix_loss, spectral_loss
from model.metrics import compute_metrics
from model.model import SpinHanceModel
from model.schedules import curriculum_weights, lr_factor
from model.targets import DegeneracyVocab, Standardizer


@dataclass
class TrainConfig:
    points: int = 16384
    ppm_from: float = 0.0
    ppm_to: float = 12.0
    field_low: float = 90.0          # input + self-consistency field
    field_high: float = 600.0        # optional reproducibility field (Stage 2)
    spectrum_field: str = "spec90"
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    epochs: int = 60
    stage1_epochs: int = 20
    ramp_epochs: int = 10
    spectral_max: float = 1.0
    matrix_anchor: float = 0.3
    warmup_frac: float = 0.03
    render_subset_frac: float = 0.2  # stochastic subset for the spectral loss
    linewidth_hz: float = 1.0
    eigh_eps: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {
        "shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5})
    patience: int = 10
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype: str = "bf16"          # bf16|fp16|none
    ckpt_path: str = "model/checkpoints/spinhance.pt"


# -----------------------------------------------------------------------------
# Differentiable decode: standardized model outputs -> physical units
# -----------------------------------------------------------------------------

def decode_physical(pred, std: Standardizer):
    """shifts (B,G) ppm, couplings (B,G,G) Hz. Couplings are SOFT-gated by the
    presence probability so the spectral loss stays differentiable (no hard
    threshold) and absent couplings render near zero."""
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


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def build_datasets(records, assignment, cfg: TrainConfig):
    by_fold = {"train": [], "val": [], "test": []}
    for r in records:
        f = assignment.get(r["mol_id"])
        if f:
            by_fold[f].append(r)
    vocab = DegeneracyVocab()
    std = Standardizer().fit(by_fold["train"], vocab)
    mk = lambda recs, aug: SpectrumMatrixDataset(
        recs, vocab, std, spectrum_field=cfg.spectrum_field, augment=aug,
        ppm_from=cfg.ppm_from, ppm_to=cfg.ppm_to, seed=cfg.seed)
    return dict(train=mk(by_fold["train"], True), val=mk(by_fold["val"], False),
                test=mk(by_fold["test"], False)), std, vocab


# -----------------------------------------------------------------------------
# Spectral term for one (single-bucket) batch
# -----------------------------------------------------------------------------

def _spectral_term(pred_phys, batch, cfg, device):
    """Render a random subset of the batch at the low field and compare to the
    input spectrum (self-consistency). Returns (loss, mean_w1) or (0, 0)."""
    deg = batch["shared_degeneracy"]
    if deg is None:                       # not single-bucket -> skip (shouldn't happen w/ bucket sampler)
        z = torch.zeros((), device=device)
        return z, z
    B = batch["spectrum"].shape[0]
    k = max(1, int(round(cfg.render_subset_frac * B)))
    sel = torch.randperm(B, device=device)[:k]
    deg_list = [int(x) for x in deg.tolist()]
    struct = renderer._structure(deg_list, device, pred_phys["shifts"].dtype)
    sub = {"shifts": pred_phys["shifts"][sel], "couplings": pred_phys["couplings"][sel]}
    ref = batch["spectrum_ref"][sel]      # CLEAN spectrum (renderer is noise-free)
    loss, w1 = spectral_loss(
        sub, ref, batch["degeneracy"][sel], cfg.field_low, renderer, struct=struct,
        points=cfg.points, ppm_from=cfg.ppm_from, ppm_to=cfg.ppm_to,
        linewidth_hz=cfg.linewidth_hz, eigh_eps=cfg.eigh_eps)
    return loss, w1.mean()


# -----------------------------------------------------------------------------
# Epoch loops
# -----------------------------------------------------------------------------

def _to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch


def train_epoch(model, loader, opt, sched, scaler, cfg, std, epoch, device,
                amp_ctx):
    model.train()
    w_mat, w_spec = curriculum_weights(epoch, cfg.stage1_epochs, cfg.ramp_epochs,
                                       cfg.spectral_max, cfg.matrix_anchor)
    running = {}
    for batch in loader:
        batch = _to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            pred = model(batch["spectrum"])
            mloss, comps = matrix_loss(pred, batch, weights=cfg.loss_weights)
            total = w_mat * mloss
            if w_spec > 0:
                pred_phys = decode_physical(pred, std)
                sloss, w1 = _spectral_term(pred_phys, batch, cfg, device)
                total = total + w_spec * sloss
                comps["spectral_w1"] = w1.detach()
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()
        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + float(v)
        running["total"] = running.get("total", 0.0) + float(total.detach())
    n = max(1, len(loader))
    return {k: v / n for k, v in running.items()} | {"w_mat": w_mat, "w_spec": w_spec}


@torch.no_grad()
def evaluate(model, loader, cfg, std, vocab, device, amp_ctx):
    model.eval()
    agg, nb = {}, 0
    for batch in loader:
        batch = _to_device(batch, device)
        with amp_ctx():
            pred = model(batch["spectrum"])
            mloss, _ = matrix_loss(pred, batch, weights=cfg.loss_weights)
        pred_np = {k: pred[k].float().cpu().numpy() for k in pred}
        tgt_np = {k: batch[k].cpu().numpy()
                  for k in ("shifts", "j_mag", "j_presence", "deg_class")}
        met = compute_metrics(pred_np, tgt_np, std, vocab)
        met["matrix_loss"] = float(mloss)
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {k: v / max(1, nb) for k, v in agg.items()}


# -----------------------------------------------------------------------------
# Fit
# -----------------------------------------------------------------------------

def _amp_context(cfg, device):
    if cfg.amp_dtype == "none" or device == "cpu":
        import contextlib
        return (lambda: contextlib.nullcontext()), None
    dt = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if dt == torch.float16 else None
    return (lambda: torch.autocast(device_type="cuda", dtype=dt)), scaler


def fit(records, assignment, cfg: TrainConfig, model=None):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    ds, std, vocab = build_datasets(records, assignment, cfg)

    plain = DataLoader(ds["train"], batch_size=cfg.batch_size, shuffle=True,
                       collate_fn=collate_fn, drop_last=True)
    bucket_samp = BucketByDegeneracySampler(ds["train"].bucket_keys,
                                            cfg.batch_size, seed=cfg.seed)
    bucketed = DataLoader(ds["train"], batch_sampler=bucket_samp,
                          collate_fn=collate_fn)
    val_dl = DataLoader(ds["val"], batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn)

    model = (model or SpinHanceModel(n_groups=8, n_deg_classes=len(vocab))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(plain))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, warmup, total_steps))
    amp_ctx, scaler = _amp_context(cfg, device)

    best, bad = float("inf"), 0
    for epoch in range(cfg.epochs):
        loader = plain if epoch < cfg.stage1_epochs else bucketed
        tr = train_epoch(model, loader, opt, sched, scaler, cfg, std, epoch,
                         device, amp_ctx)
        va = evaluate(model, val_dl, cfg, std, vocab, device, amp_ctx)
        print(f"epoch {epoch:3d} | stage {'1' if epoch < cfg.stage1_epochs else '2'} "
              f"| w_spec {tr['w_spec']:.2f} | train_total {tr['total']:.4f} "
              f"| val shift_mae {va['shift_mae_ppm']:.3f}ppm "
              f"J_mae {va['j_mae_hz']:.2f}Hz pres_f1 {va['presence_f1']:.3f} "
              f"deg_acc {va['deg_acc']:.3f}")
        score = va["shift_mae_ppm"] + va["j_mae_hz"] / 10.0   # simple early-stop score
        if score < best:
            best, bad = score, 0
            import os
            os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "standardizer": vars(std), "cfg": vars(cfg)}, cfg.ckpt_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"early stop at epoch {epoch}")
                break
    return model, std, vocab


# -----------------------------------------------------------------------------
# Synthetic smoke test
# -----------------------------------------------------------------------------

def _smoke():
    from model.splits import make_splits
    rng = np.random.default_rng(0)
    G, P = 8, 2048
    recs = []
    for i in range(96):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 10))
        recs.append(dict(mol_id=f"m{i}", shifts=rng.uniform(0.5, 9, G), couplings=c,
                         degeneracy=rng.choice([1, 2, 3], size=G).astype(int),
                         scaffold=f"s{i % 12}", spec90=rng.random(P).astype(np.float32)))
    assignment, _ = make_splits(recs, seed=0)
    cfg = TrainConfig(points=P, batch_size=8, epochs=3, stage1_epochs=1,
                      ramp_epochs=1, warmup_frac=0.1, device="cpu",
                      amp_dtype="none", patience=5,
                      ckpt_path="/tmp/spinhance_smoke.pt")
    fit(recs, assignment, cfg)
    print("SMOKE OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke()
    else:
        print("Provide records + assignment and call fit(); see _smoke() for usage.")
