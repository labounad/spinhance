"""
model.train
==============
Training loop for Task 4.  Changes from the original:
  - DiagnosticsWriter integration (run_dir, status.json, metrics.jsonl, events.jsonl)
  - Step-level metric logging (every log_every_steps optimizer steps)
  - Canonical run directory: model/runs/<run_id>/checkpoints/
  - ProbeEvaluator + failure-case tables every probe_every_epochs
  - Hungarian-matched metrics logged at validation
  - train_epoch now returns (metrics_dict, global_step)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue as _queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import diff_renderer_torch as renderer
from model.dataset import (SpectrumMatrixDataset, BucketByDegeneracySampler,
                              collate_fn, worker_init_fn)
from model.diagnostics import DiagnosticsWriter
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
    field_low: float = 90.0
    field_high: float = 600.0
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
    render_subset_frac: float = 0.2
    linewidth_hz: float = 1.0
    eigh_eps: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {
        "shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5})
    patience: int = 10
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype: str = "bf16"
    ckpt_path: str = "model/checkpoints/spinhance.pt"
    s3_ckpt_prefix: str = ""
    num_workers: int = -1
    val_every: int = 1
    # ── Diagnostics ────────────────────────────────────────────────────────────
    diagnostics_enabled: bool = True
    run_dir: str = ""           # auto-generated if empty
    run_name: str = ""          # suffix for auto-generated run_id
    log_every_steps: int = 25   # step-level metric logging frequency
    probe_every_epochs: int = 5
    probe_count: int = 16
    save_probe_plots: bool = True
    save_failure_tables: bool = True


# ── Run directory ──────────────────────────────────────────────────────────────

def _make_run_dir(cfg: TrainConfig) -> Path:
    if cfg.run_dir:
        return Path(cfg.run_dir)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = cfg.run_name or Path(cfg.ckpt_path).stem
    cfg_s  = json.dumps(
        {k: v for k, v in vars(cfg).items() if k not in ("run_dir", "run_name")},
        sort_keys=True, default=str,
    ).encode()
    h      = hashlib.sha256(cfg_s).hexdigest()[:6]
    return Path(__file__).parent / "runs" / f"{ts}_{suffix}_{h}"


# ── Differentiable decode ──────────────────────────────────────────────────────

def decode_physical(pred, std: Standardizer):
    shifts = pred["shifts"] * std.shift_std + std.shift_mean
    jmag   = pred["j_mag"] * std.j_std + std.j_mean
    gate   = torch.sigmoid(pred["j_presence"])
    jmag   = jmag * gate
    B, G   = shifts.shape
    iu     = torch.triu_indices(G, G, 1, device=shifts.device)
    C      = torch.zeros(B, G, G, device=shifts.device, dtype=shifts.dtype)
    C[:, iu[0], iu[1]] = jmag
    C[:, iu[1], iu[0]] = jmag
    return {"shifts": shifts, "couplings": C}


# ── Loaders ────────────────────────────────────────────────────────────────────

def build_datasets(records, assignment, cfg: TrainConfig):
    by_fold = {"train": [], "val": [], "test": []}
    for r in records:
        f = assignment.get(r["mol_id"])
        if f:
            by_fold[f].append(r)
    vocab = DegeneracyVocab()
    std   = Standardizer().fit(by_fold["train"], vocab)
    mk    = lambda recs, aug: SpectrumMatrixDataset(
        recs, vocab, std, spectrum_field=cfg.spectrum_field, augment=aug,
        ppm_from=cfg.ppm_from, ppm_to=cfg.ppm_to, seed=cfg.seed)
    return dict(train=mk(by_fold["train"], True), val=mk(by_fold["val"], False),
                test=mk(by_fold["test"], False)), std, vocab


# ── Spectral term ──────────────────────────────────────────────────────────────

def _spectral_term(pred_phys, batch, cfg, device):
    deg = batch["shared_degeneracy"]
    if deg is None:
        z = torch.zeros((), device=device)
        return z, z
    B   = batch["spectrum"].shape[0]
    k   = max(1, int(round(cfg.render_subset_frac * B)))
    sel = torch.randperm(B, device=device)[:k]
    deg_list = [int(x) for x in deg.tolist()]
    struct   = renderer._structure(deg_list, device, pred_phys["shifts"].dtype)
    sub      = {"shifts": pred_phys["shifts"][sel], "couplings": pred_phys["couplings"][sel]}
    ref      = batch["spectrum_ref"][sel]
    loss, w1 = spectral_loss(
        sub, ref, batch["degeneracy"][sel], cfg.field_low, renderer, struct=struct,
        points=cfg.points, ppm_from=cfg.ppm_from, ppm_to=cfg.ppm_to,
        linewidth_hz=cfg.linewidth_hz, eigh_eps=cfg.eigh_eps)
    return loss, w1.mean()


# ── Epoch loops ────────────────────────────────────────────────────────────────

def train_epoch(
    model, loader, opt, sched, scaler, cfg, std, epoch, device,
    amp_ctx, balance=None, diagnostics=None, global_step_start: int = 0,
) -> tuple[dict, int]:
    """Returns (epoch_metrics, global_step_after_epoch)."""
    model.train()
    bal    = balance or {}
    w_mat, w_spec = curriculum_weights(epoch, cfg.stage1_epochs, cfg.ramp_epochs,
                                       cfg.spectral_max, cfg.matrix_anchor)
    stage   = 1 if epoch < cfg.stage1_epochs else 2
    running = {}
    step    = global_step_start

    for batch_idx, batch in enumerate(_Prefetcher(loader, device)):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            pred             = model(batch["spectrum"])
            mloss, comps     = matrix_loss(pred, batch, weights=cfg.loss_weights,
                                           deg_class_weight=bal.get("deg_weights"),
                                           presence_pos_weight=bal.get("presence_pos_weight"))
            total = w_mat * mloss
            if w_spec > 0:
                pred_phys    = decode_physical(pred, std)
                sloss, w1    = _spectral_term(pred_phys, batch, cfg, device)
                total        = total + w_spec * sloss
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
                "loss_total":    float(total.detach()),
                "loss_shift":    float(comps.get("shift",      0.0)),
                "loss_jmag":     float(comps.get("jmag",       0.0)),
                "loss_presence": float(comps.get("presence",   0.0)),
                "loss_deg":      float(comps.get("deg",        0.0)),
                "spectral_w1":   float(comps.get("spectral_w1", 0.0)),
                "lr":            float(sched.get_last_lr()[0]),
                "w_mat":         float(w_mat),
                "w_spec":        float(w_spec),
                "grad_norm":     gnorm,
                "seconds_per_step": step_secs,
                "amp_scale":     amp_scale,
            }
            if torch.cuda.is_available():
                step_metrics["cuda_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                step_metrics["cuda_reserved_gb"]  = torch.cuda.memory_reserved(device)  / 1e9
            diagnostics.log_metrics(
                split="train_step", epoch=epoch, step=step,
                metrics=step_metrics, extra={"stage": stage, "batch_idx": batch_idx},
            )

        step += 1

    n = max(1, len(loader))
    epoch_metrics = {k: v / n for k, v in running.items()} | {"w_mat": w_mat, "w_spec": w_spec}
    return epoch_metrics, step


@torch.no_grad()
def evaluate(model, loader, cfg, std, vocab, device, amp_ctx, balance=None):
    model.eval()
    bal = balance or {}
    agg, nb = {}, 0
    for batch in _Prefetcher(loader, device):
        with amp_ctx():
            pred  = model(batch["spectrum"])
            mloss, _ = matrix_loss(pred, batch, weights=cfg.loss_weights,
                                   deg_class_weight=bal.get("deg_weights"),
                                   presence_pos_weight=bal.get("presence_pos_weight"))
        pred_np = {k: pred[k].float().cpu().numpy() for k in pred}
        tgt_np  = {k: batch[k].cpu().numpy()
                   for k in ("shifts", "j_mag", "j_presence", "deg_class")}
        met     = compute_metrics(pred_np, tgt_np, std, vocab)
        met["matrix_loss"] = float(mloss)
        for k, v in met.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    return {k: v / max(1, nb) for k, v in agg.items()}


# ── CUDA-stream prefetcher ─────────────────────────────────────────────────────

class _Prefetcher:
    def __init__(self, loader, device):
        self._n      = len(loader)
        self._loader = iter(loader)
        self._device = device
        self._stream = torch.cuda.Stream() if device != "cpu" else None
        self._next   = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self._loader)
        except StopIteration:
            self._next = None
            return
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                batch = {k: v.to(self._device, non_blocking=True)
                         if torch.is_tensor(v) else v
                         for k, v in batch.items()}
        self._next = batch

    def __iter__(self):
        return self

    def __next__(self):
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __len__(self):
        return self._n


# ── Background checkpoint worker ───────────────────────────────────────────────

def _do_checkpoint(ckpt, last_path: str, s3_dest: str, best_path: str, legacy_path: str) -> None:
    torch.save(ckpt, last_path)
    if s3_dest:
        subprocess.run(["aws", "s3", "cp", last_path, s3_dest], capture_output=True)
    if best_path:
        torch.save(ckpt, best_path)
    if legacy_path:
        torch.save(ckpt, legacy_path)


def _ckpt_worker(q: "_queue.Queue"):
    while True:
        item = q.get()
        if item is None:
            return
        _do_checkpoint(*item)


# ── Fit ────────────────────────────────────────────────────────────────────────

def _amp_context(cfg, device):
    if cfg.amp_dtype == "none" or device == "cpu":
        import contextlib
        return (lambda: contextlib.nullcontext()), None
    dt     = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if dt == torch.float16 else None
    return (lambda: torch.autocast(device_type="cuda", dtype=dt)), scaler


def fit(records, assignment, cfg: TrainConfig, model=None):
    torch.manual_seed(cfg.seed)
    device = cfg.device

    # ── Run directory + diagnostics ────────────────────────────────────────────
    run_dir  = _make_run_dir(cfg)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id   = run_dir.name

    diag = DiagnosticsWriter(run_dir, enabled=cfg.diagnostics_enabled)
    diag.write_config(vars(cfg))
    diag.log_event("run_start", {"run_id": run_id, "device": device, "epochs": cfg.epochs})

    # ── Datasets + loaders ─────────────────────────────────────────────────────
    ds, std, vocab = build_datasets(records, assignment, cfg)

    pin  = (device != "cpu")
    nw   = cfg.num_workers if cfg.num_workers >= 0 else os.cpu_count() or 4
    dl_kw = dict(collate_fn=collate_fn, num_workers=nw, pin_memory=pin,
                 persistent_workers=nw > 0, worker_init_fn=worker_init_fn)
    plain       = DataLoader(ds["train"], batch_size=cfg.batch_size, shuffle=True,
                             drop_last=True, **dl_kw)
    bucket_samp = BucketByDegeneracySampler(ds["train"].bucket_keys,
                                            cfg.batch_size, seed=cfg.seed)
    bucketed    = DataLoader(ds["train"], batch_sampler=bucket_samp, **dl_kw)
    val_dl      = DataLoader(ds["val"], batch_size=cfg.batch_size,
                             shuffle=False, **dl_kw)

    model = (model or SpinHanceModel(n_groups=8, n_deg_classes=len(vocab))).to(device)

    from model.targets import class_balance
    train_recs = [r for r in records if assignment.get(r["mol_id"]) == "train"]
    val_recs   = [r for r in records if assignment.get(r["mol_id"]) == "val"]
    cb = class_balance(train_recs, vocab)
    balance = {
        "deg_weights":        torch.tensor(cb["deg_weights"],        device=device),
        "presence_pos_weight": torch.tensor(cb["presence_pos_weight"], device=device),
    }
    print(f"class balance: deg_counts={cb['deg_counts'].tolist()} "
          f"presence_pos_weight={cb['presence_pos_weight']:.2f}")

    # ── Probe evaluator ────────────────────────────────────────────────────────
    probe_eval = None
    if cfg.diagnostics_enabled and cfg.probe_count > 0 and len(val_recs) > 0:
        try:
            from model.probes import ProbeEvaluator
            probe_eval = ProbeEvaluator(
                val_recs, ds["val"], vocab, std,
                probe_count=cfg.probe_count, device=device,
                run_dir=run_dir, save_plots=cfg.save_probe_plots,
            )
        except Exception as e:
            print(f"[train] ProbeEvaluator init failed ({e}) — skipping probes")

    # ── Optimiser + schedule ───────────────────────────────────────────────────
    opt        = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per  = max(1, len(plain))
    total_steps = steps_per * cfg.epochs
    warmup     = int(cfg.warmup_frac * total_steps)
    sched      = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warmup, total_steps))
    amp_ctx, scaler = _amp_context(cfg, device)

    will_run_stage2 = cfg.stage1_epochs < cfg.epochs
    best, bad, best_epoch = float("inf"), 0, 0
    va = {}

    # ── Checkpoint thread ──────────────────────────────────────────────────────
    ckpt_q      = _queue.Queue(maxsize=1)
    ckpt_thread = threading.Thread(target=_ckpt_worker, args=(ckpt_q,), daemon=True)
    ckpt_thread.start()

    global_step = 0
    last_stage  = 1

    for epoch in range(cfg.epochs):
        stage = 1 if epoch < cfg.stage1_epochs else 2

        if stage != last_stage:
            diag.log_event("stage_change", {"from_stage": last_stage, "to_stage": stage, "epoch": epoch})
            bad       = 0
            last_stage = stage

        loader = plain if epoch < cfg.stage1_epochs else bucketed
        tr, global_step = train_epoch(
            model, loader, opt, sched, scaler, cfg, std, epoch, device,
            amp_ctx, balance=balance,
            diagnostics=diag, global_step_start=global_step,
        )

        tr_log = {k: v for k, v in tr.items()
                  if k not in ("w_mat", "w_spec") and isinstance(v, (int, float))}
        diag.log_metrics(split="train", epoch=epoch, step=global_step,
                         metrics=tr_log, extra={"stage": stage, "w_mat": tr["w_mat"],
                                                "w_spec": tr["w_spec"]})

        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.epochs - 1)
        if do_val:
            va = evaluate(model, val_dl, cfg, std, vocab, device, amp_ctx, balance=balance)
            diag.log_metrics(split="val", epoch=epoch, step=global_step, metrics=va,
                             extra={"stage": stage})

        score            = (va.get("shift_mae_ppm", float("inf")) +
                            va.get("j_mae_hz",      float("inf")) / 10.0)
        earlystop_active = (epoch >= cfg.stage1_epochs) or (not will_run_stage2)
        is_best          = do_val and score < best

        if is_best:
            best, bad, best_epoch = score, 0, epoch
            diag.log_event("best_checkpoint", {"epoch": epoch, "score": float(score), **va})
        elif do_val and earlystop_active:
            bad += 1
            if bad >= cfg.patience:
                diag.log_event("early_stop", {"epoch": epoch, "patience": cfg.patience})
                print(f"early stop at epoch {epoch}")
                break

        # Status (written atomically — safe for concurrent dashboard reads)
        diag.update_status({
            "state":             "running",
            "run_id":            run_id,
            "epoch":             epoch,
            "epochs":            cfg.epochs,
            "stage":             stage,
            "global_step":       global_step,
            "best_score":        float(best) if best != float("inf") else None,
            "best_epoch":        best_epoch,
            "last_update_time":  time.time(),
            "device":            cfg.device,
            "checkpoint_best":   "checkpoints/best.pt",
            "checkpoint_last":   "checkpoints/last.pt",
        })

        print(f"epoch {epoch:3d} | stage {stage} | w_spec {tr['w_spec']:.2f} "
              f"| train {tr.get('total', 0):.4f}"
              + (f" | val shift {va.get('shift_mae_ppm', 0):.3f}ppm "
                 f"J {va.get('j_mae_hz', 0):.2f}Hz "
                 f"h_shift {va.get('h_shift_mae_ppm', va.get('shift_mae_ppm', 0)):.3f}ppm "
                 f"f1 {va.get('presence_f1', 0):.3f} deg {va.get('deg_acc_balanced', 0):.3f}"
                 if do_val else " | val skipped"))

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {
            "model":       {k: v.cpu() for k, v in model.state_dict().items()},
            "standardizer": vars(std),
            "cfg":          vars(cfg),
            "epoch":        epoch,
            "metrics":      va,
        }
        last_path  = str(ckpt_dir / "last.pt")
        best_path  = str(ckpt_dir / "best.pt") if is_best else ""
        legacy     = cfg.ckpt_path if is_best else ""
        s3_dest    = f"{cfg.s3_ckpt_prefix}/last.pt" if cfg.s3_ckpt_prefix else ""
        ckpt_q.put((ckpt, last_path, s3_dest, best_path, legacy))

        # ── Probes + failure analysis ──────────────────────────────────────────
        run_probe = (epoch % cfg.probe_every_epochs == 0) or (epoch == cfg.epochs - 1)
        if run_probe and cfg.diagnostics_enabled:
            if probe_eval is not None:
                try:
                    pagg = probe_eval.run(model, epoch, amp_ctx)
                    if pagg:
                        diag.log_metrics(split="probe", epoch=epoch, step=global_step,
                                         metrics=pagg)
                except Exception as e:
                    print(f"[train] probe run failed at epoch {epoch}: {e}")

            if cfg.save_failure_tables and val_recs:
                try:
                    from model.failure_analysis import per_sample_evaluate, save_failure_cases
                    per_sample = per_sample_evaluate(
                        model, val_recs, ds["val"], cfg, std, vocab, device, amp_ctx, balance)
                    fsummary = save_failure_cases(per_sample, run_dir, epoch)
                    diag.log_event("failure_analysis", {"epoch": epoch, **fsummary})
                except Exception as e:
                    print(f"[train] failure analysis failed at epoch {epoch}: {e}")

    ckpt_q.put(None)
    ckpt_thread.join()

    # ── Finalize ───────────────────────────────────────────────────────────────
    # Read the most recent failure summary written by failure_analysis (if any)
    failure_summary: dict = {}
    dominant = ""
    probe_dir = run_dir / "probes"
    if probe_dir.exists():
        epoch_dirs = sorted([d for d in probe_dir.iterdir() if d.is_dir()], reverse=True)
        if epoch_dirs:
            fs_path = epoch_dirs[0] / "failure_summary.json"
            if fs_path.exists():
                try:
                    failure_summary = json.loads(fs_path.read_text())
                    dominant = failure_summary.get("dominant_failure", "")
                except Exception:
                    pass

    _HINTS = {
        "large_shift_error":        "Increase shift loss weight or use Hungarian matching loss",
        "false_negative_couplings": "Increase presence_pos_weight",
        "false_positive_couplings": "Lower presence threshold or up-weight absence class",
        "bad_j_magnitude":          "Increase j_mag loss weight",
        "wrong_degeneracy":         "Try integration-aware features or degeneracy vocab check",
    }

    summary = {
        "run_id":          run_id,
        "state":           "finished",
        "best_epoch":      best_epoch,
        "best_score":      float(best) if best != float("inf") else None,
        "best_metrics":    va,
        "score_formula":   "shift_mae_ppm + j_mae_hz / 10.0",
        "failure_summary": failure_summary,
        "recommendation":  _HINTS.get(dominant, ""),
    }
    diag.finalize(summary)
    diag.update_status({
        "state":             "finished",
        "run_id":            run_id,
        "epoch":             best_epoch,
        "epochs":            cfg.epochs,
        "stage":             last_stage,
        "global_step":       global_step,
        "best_score":        float(best) if best != float("inf") else None,
        "best_epoch":        best_epoch,
        "last_update_time":  time.time(),
        "device":            cfg.device,
        "checkpoint_best":   "checkpoints/best.pt",
        "checkpoint_last":   "checkpoints/last.pt",
    })
    diag.log_event("run_end", {"run_id": run_id, "best_epoch": best_epoch,
                               "best_score": float(best) if best != float("inf") else None})
    print(f"Run complete → {run_dir}")

    return model, std, vocab


# ── Synthetic smoke test ───────────────────────────────────────────────────────

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
                      ckpt_path="/tmp/spinhance_smoke.pt",
                      log_every_steps=5, probe_every_epochs=2, probe_count=4,
                      save_probe_plots=False)
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
