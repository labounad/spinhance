"""
model.train
===========
Infrastructure and orchestration for Task-4 training.

Stage logic lives in dedicated modules:
  model.stage1  matrix-only training + validation (stage 1)
  model.stage2  matrix + spectral training          (stage 2)

This file provides:
  TrainConfig        hyper-parameter dataclass
  build_datasets     record -> Dataset/DataLoader construction
  fit                main training loop (calls stage1 / stage2)
  _Prefetcher        CUDA-stream data prefetcher passed to stage functions
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

import model.stage1 as stage1
import model.stage2 as stage2
from model.dataset import (SpectrumMatrixDataset, BucketByDegeneracySampler,
                            collate_fn, worker_init_fn)
from model.diagnostics import DiagnosticsWriter
from model.model import SpinHanceModel
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
    run_dir: str = ""
    run_name: str = ""
    log_every_steps: int = 25
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
    h = hashlib.sha256(cfg_s).hexdigest()[:6]
    return Path(__file__).parent / "runs" / f"{ts}_{suffix}_{h}"


# ── Datasets ───────────────────────────────────────────────────────────────────

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


# ── CUDA-stream prefetcher ─────────────────────────────────────────────────────

class _Prefetcher:
    """Wraps a DataLoader; moves each batch to device on a side stream so the
    next batch is ready before the training step finishes."""

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

def _save_checkpoint_file(ckpt, path: str) -> None:
    """Save a checkpoint after ensuring the parent directory exists."""
    if not path:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, p)


def _do_checkpoint(ckpt, last_path: str, s3_dest: str, best_path: str, legacy_path: str) -> None:
    _save_checkpoint_file(ckpt, last_path)

    if s3_dest:
        subprocess.run(["aws", "s3", "cp", last_path, s3_dest], capture_output=True)

    if best_path:
        _save_checkpoint_file(ckpt, best_path)

    if legacy_path:
        _save_checkpoint_file(ckpt, legacy_path)


def _ckpt_worker(q: "_queue.Queue"):
    while True:
        item = q.get()
        if item is None:
            return
        _do_checkpoint(*item)


# ── AMP context factory ────────────────────────────────────────────────────────

def _amp_context(cfg, device):
    if cfg.amp_dtype == "none" or device == "cpu":
        import contextlib
        return (lambda: contextlib.nullcontext()), None
    dt     = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler() if dt == torch.float16 else None
    return (lambda: torch.autocast(device_type="cuda", dtype=dt)), scaler


# ── Fit ────────────────────────────────────────────────────────────────────────

def fit(records, assignment, cfg: TrainConfig, model=None):
    torch.manual_seed(cfg.seed)
    device = cfg.device

    # ── Run directory + diagnostics ────────────────────────────────────────────
    run_dir  = _make_run_dir(cfg)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_id   = run_dir.name

    diag = DiagnosticsWriter(run_dir, enabled=cfg.diagnostics_enabled)
    if diag is not None:
        diag.reset_live_files()
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
        "deg_weights":         torch.tensor(cb["deg_weights"],         device=device),
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
    from model.schedules import lr_factor
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
        cur_stage = 1 if epoch < cfg.stage1_epochs else 2

        if cur_stage != last_stage:
            diag.log_event("stage_change",
                           {"from_stage": last_stage, "to_stage": cur_stage, "epoch": epoch})
            bad        = 0
            last_stage = cur_stage

        # ── Dispatch to the correct stage module ───────────────────────────────
        if cur_stage == 1:
            tr, global_step = stage1.train_epoch(
                model, _Prefetcher(plain, device),
                opt, sched, scaler, cfg, std, epoch, device,
                amp_ctx, balance=balance,
                diagnostics=diag, global_step_start=global_step,
            )
        else:
            tr, global_step = stage2.train_epoch(
                model, _Prefetcher(bucketed, device),
                opt, sched, scaler, cfg, std, epoch, device,
                amp_ctx, balance=balance,
                diagnostics=diag, global_step_start=global_step,
            )

        tr_log = {k: v for k, v in tr.items()
                  if k not in ("w_mat", "w_spec") and isinstance(v, (int, float))}
        diag.log_metrics(split="train", epoch=epoch, step=global_step,
                         metrics=tr_log, extra={"stage": cur_stage,
                                                "w_mat": tr["w_mat"],
                                                "w_spec": tr["w_spec"]})

        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.epochs - 1)
        if do_val:
            va = stage1.evaluate(
                model, _Prefetcher(val_dl, device),
                cfg, std, vocab, device, amp_ctx, balance=balance,
            )
            diag.log_metrics(split="val", epoch=epoch, step=global_step, metrics=va,
                             extra={"stage": cur_stage})

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

        diag.update_status({
            "state":            "running",
            "run_id":           run_id,
            "epoch":            epoch,
            "epochs":           cfg.epochs,
            "stage":            cur_stage,
            "global_step":      global_step,
            "best_score":       float(best) if best != float("inf") else None,
            "best_epoch":       best_epoch,
            "last_update_time": time.time(),
            "device":           cfg.device,
            "checkpoint_best":  "checkpoints/best.pt",
            "checkpoint_last":  "checkpoints/last.pt",
        })

        print(f"epoch {epoch:3d} | stage {cur_stage} | w_spec {tr['w_spec']:.2f} "
              f"| train {tr.get('total', 0):.4f}"
              + (f" | val shift {va.get('shift_mae_ppm', 0):.3f}ppm "
                 f"J {va.get('j_mae_hz', 0):.2f}Hz "
                 f"h_shift {va.get('h_shift_mae_ppm', va.get('shift_mae_ppm', 0)):.3f}ppm "
                 f"f1 {va.get('presence_f1', 0):.3f} deg {va.get('deg_acc_balanced', 0):.3f}"
                 if do_val else " | val skipped"))

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt = {
            "model":        {k: v.cpu() for k, v in model.state_dict().items()},
            "standardizer": vars(std),
            "cfg":          vars(cfg),
            "epoch":        epoch,
            "metrics":      va,
        }
        last_path = str(ckpt_dir / "last.pt")
        best_path = str(ckpt_dir / "best.pt") if is_best else ""
        legacy    = cfg.ckpt_path if is_best else ""
        s3_dest   = f"{cfg.s3_ckpt_prefix}/epoch_{epoch:03d}.pt" if cfg.s3_ckpt_prefix else ""
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
        "state":            "finished",
        "run_id":           run_id,
        "epoch":            best_epoch,
        "epochs":           cfg.epochs,
        "stage":            last_stage,
        "global_step":      global_step,
        "best_score":       float(best) if best != float("inf") else None,
        "best_epoch":       best_epoch,
        "last_update_time": time.time(),
        "device":           cfg.device,
        "checkpoint_best":  "checkpoints/best.pt",
        "checkpoint_last":  "checkpoints/last.pt",
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
