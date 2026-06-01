"""
model.training.trainer
=====================
Orchestrates a run: build datasets/model/loss/optimizer, run the epoch loop,
validate, checkpoint, and write the canonical diagnostics artifacts. Contains no
loss math or renderer internals.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.architectures import build_architecture
from model.data.collate import collate_spin_batch
from model.data.dataset import SpectrumMatrixDataset, worker_init_fn
from model.data.standardization import DegeneracyVocab, Standardizer, class_balance
from model.diagnostics import DiagnosticsWriter
from model.losses import build_composite
from model.training.checkpointing import save_checkpoint
from model.training.config import Config
from model.training.loops import evaluate, train_epoch
from model.training.optimizer import amp_context, build_optimizer_and_scheduler
from model.training.seed import seed_everything


# Coarse next-experiment hints keyed by the dominant failure tag (mirrors
# autoai/run_reader); surfaced in summary.json for the dashboard + AutoAI.
_FAILURE_HINTS = {
    "large_shift_error": "Increase shift loss weight or use the Hungarian graph loss",
    "false_negative_couplings": "Increase presence_pos_weight",
    "false_positive_couplings": "Lower presence threshold or up-weight the absence class",
    "bad_j_magnitude": "Increase j_mag loss weight",
    "wrong_degeneracy": "Try integration-aware features or check the degeneracy vocab",
}


def _resolve_device(d):
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_run_dir(cfg: Config) -> Path:
    if cfg.run.dir:                       # explicit, deterministic run dir
        return Path(cfg.run.dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(json.dumps(cfg.raw, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return Path(cfg.run.output_dir) / f"{ts}_{cfg.run.name}_{h}"


class Trainer:
    def __init__(self, config: Config, records, assignment):
        self.cfg = config
        self.records = records
        self.assignment = assignment
        self.device = _resolve_device(config.training.device)

    # ── setup ──────────────────────────────────────────────────────────────────

    def _build(self):
        cfg = self.cfg
        by_fold = {"train": [], "val": [], "test": []}
        for r in self.records:
            f = self.assignment.get(r["mol_id"])
            if f:
                by_fold[f].append(r)

        vocab = DegeneracyVocab()
        std = Standardizer().fit(by_fold["train"], vocab)
        field = cfg.data.field
        sf = f"spec{field}"
        mk = lambda recs, aug: SpectrumMatrixDataset(recs, vocab, std, spectrum_field=sf,
                                                     augment=aug, seed=cfg.training.seed)
        ds = {"train": mk(by_fold["train"], True), "val": mk(by_fold["val"], False)}

        cb = class_balance(by_fold["train"], vocab)

        model_cfg = dict(cfg.model)
        name = model_cfg.pop("name")
        model = build_architecture(name, n_deg_classes=len(vocab), **model_cfg).to(self.device)

        loss_fn = build_composite(cfg.loss["terms"],
                                  deg_class_weight=cb["deg_weights"],
                                  presence_pos_weight=cb["presence_pos_weight"],
                                  # de-standardization + field for surrogate_spectral
                                  # (ignored by losses that don't accept them)
                                  field=field, deg_vocab=vocab.vocab,
                                  **std.state_dict())
        return ds, std, vocab, model, loss_fn, cb, by_fold["val"]

    # ── fit ────────────────────────────────────────────────────────────────────

    def fit(self):
        cfg = self.cfg
        seed_everything(cfg.training.seed)
        ds, std, vocab, model, loss_fn, cb, val_records = self._build()

        nw = cfg.training.num_workers
        pin = self.device != "cpu"
        dl_kw = dict(collate_fn=collate_spin_batch, num_workers=nw, pin_memory=pin,
                     persistent_workers=nw > 0, worker_init_fn=worker_init_fn)
        train_dl = DataLoader(ds["train"], batch_size=cfg.training.batch_size,
                              shuffle=True, drop_last=True, **dl_kw)
        val_dl = DataLoader(ds["val"], batch_size=cfg.training.batch_size,
                            shuffle=False, **dl_kw)

        opt, sched = build_optimizer_and_scheduler(
            model, cfg.training.lr, cfg.training.weight_decay, cfg.training.warmup_frac,
            max(1, len(train_dl)), cfg.training.epochs,
            min_factor=cfg.training.lr_min_factor, stable_frac=cfg.training.lr_stable_frac)
        amp_ctx, scaler = amp_context(cfg.training.amp, self.device)

        run_dir = _make_run_dir(cfg)
        run_id = run_dir.name
        ckpt_dir = run_dir / "checkpoints"
        diag = DiagnosticsWriter(run_dir, enabled=cfg.diagnostics.enabled)
        diag.reset_live_files()
        diag.write_config(cfg.raw)
        diag.log_event("run_start", {"run_id": run_id, "device": self.device,
                                     "epochs": cfg.training.epochs,
                                     "n_train": len(ds["train"]), "n_val": len(ds["val"]),
                                     "deg_counts": cb["deg_counts"].tolist()})
        print(f"[trainer] run {run_id} | device {self.device} | "
              f"train {len(ds['train'])} val {len(ds['val'])} | params {model.n_params/1e6:.2f}M")

        # Probe evaluator (fixed stratified val molecules). Guarded so a probe
        # failure never kills training.
        probe_eval = None
        if cfg.diagnostics.enabled and cfg.diagnostics.probe_count > 0 and val_records:
            try:
                from model.evaluation.probes import ProbeEvaluator
                probe_eval = ProbeEvaluator(val_records, ds["val"], vocab, std,
                                            probe_count=cfg.diagnostics.probe_count,
                                            device=self.device, run_dir=run_dir)
            except Exception as e:
                print(f"[trainer] ProbeEvaluator init failed ({e}) — skipping probes")

        best, best_epoch, bad = float("inf"), 0, 0
        global_step = 0
        va: dict[str, float] = {}

        for epoch in range(cfg.training.epochs):
            loss_fn.set_epoch(epoch)
            tr, global_step = train_epoch(
                model, train_dl, loss_fn, opt, sched, scaler, amp_ctx, self.device,
                epoch=epoch, global_step=global_step, grad_clip=cfg.training.grad_clip,
                log_every_steps=cfg.diagnostics.log_every_steps, stage="1",
                diagnostics=diag)
            diag.log_metrics(split="train", epoch=epoch, step=global_step, metrics=tr,
                             extra={"stage": "1"})

            do_val = (epoch % cfg.training.val_every == 0) or (epoch == cfg.training.epochs - 1)
            if do_val:
                va = evaluate(model, val_dl, loss_fn, std, vocab, amp_ctx, self.device)
                diag.log_metrics(split="val", epoch=epoch, step=global_step, metrics=va,
                                 extra={"stage": "1"})

            score = va.get("shift_mae_ppm", float("inf")) + va.get("j_mae_hz", float("inf")) / 10.0
            is_best = do_val and score < best
            if is_best:
                best, best_epoch, bad = score, epoch, 0
                diag.log_event("best_checkpoint", {"epoch": epoch, "score": float(score)})
            elif do_val:
                bad += 1

            save_checkpoint(ckpt_dir / "last.pt", model, std, cfg.raw, epoch, va)
            if is_best:
                save_checkpoint(ckpt_dir / "best.pt", model, std, cfg.raw, epoch, va)
            # Per-epoch snapshots so the viewer can inspect any epoch's weights.
            if cfg.training.save_every and epoch % cfg.training.save_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", model, std, cfg.raw, epoch, va)

            diag.update_status({
                "state": "running", "run_id": run_id, "epoch": epoch,
                "epochs": cfg.training.epochs, "stage": "1", "global_step": global_step,
                "best_score": (float(best) if best != float("inf") else None),
                "best_epoch": best_epoch, "device": self.device,
                "last_update_time": time.time(),
                "checkpoint_best": "checkpoints/best.pt", "checkpoint_last": "checkpoints/last.pt",
            })
            print(f"epoch {epoch:3d} | train {tr.get('total', 0):.4f}"
                  + (f" | val shift {va.get('shift_mae_ppm', 0):.3f}ppm "
                     f"J {va.get('j_mae_hz', 0):.2f}Hz f1 {va.get('presence_f1', 0):.3f} "
                     f"deg {va.get('deg_acc_balanced', 0):.3f}" if do_val else " | val skipped"))

            # ── Probes + failure analysis (every probe_every_epochs + last) ──────
            run_probe = (cfg.diagnostics.enabled
                         and (epoch % cfg.diagnostics.probe_every_epochs == 0
                              or epoch == cfg.training.epochs - 1))
            if run_probe and probe_eval is not None:
                try:
                    pagg = probe_eval.run(model, epoch, amp_ctx)
                    if pagg:
                        diag.log_metrics(split="probe", epoch=epoch, step=global_step, metrics=pagg)
                    from model.evaluation.failure_analysis import (
                        per_sample_evaluate, save_failure_cases)
                    per_sample = per_sample_evaluate(
                        model, val_records, ds["val"], cfg.training.batch_size,
                        std, vocab, self.device, amp_ctx)
                    fsummary = save_failure_cases(per_sample, run_dir, epoch)
                    diag.log_event("failure_analysis", {"epoch": epoch, **fsummary})
                except Exception as e:
                    print(f"[trainer] probe/failure analysis failed at epoch {epoch}: {e}")

            if bad >= cfg.training.patience:
                diag.log_event("early_stop", {"epoch": epoch, "patience": cfg.training.patience})
                print(f"early stop at epoch {epoch}")
                break

        # Latest failure summary -> summary.json + a coarse recommendation.
        failure_summary, recommendation = {}, ""
        probe_dir = run_dir / "probes"
        if probe_dir.exists():
            eps = sorted((d for d in probe_dir.iterdir() if d.is_dir()), reverse=True)
            for d in eps:
                fp = d / "failure_summary.json"
                if fp.exists():
                    import json as _json
                    failure_summary = _json.loads(fp.read_text())
                    recommendation = _FAILURE_HINTS.get(
                        failure_summary.get("dominant_failure", ""), "")
                    break

        summary = {
            "run_id": run_id, "state": "finished", "best_epoch": best_epoch,
            "best_score": (float(best) if best != float("inf") else None),
            "best_metrics": va, "score_formula": "shift_mae_ppm + j_mae_hz / 10.0",
            "failure_summary": failure_summary, "recommendation": recommendation,
        }
        diag.finalize(summary)
        diag.update_status({
            "state": "finished", "run_id": run_id, "epoch": best_epoch,
            "epochs": cfg.training.epochs, "stage": "1", "global_step": global_step,
            "best_score": (float(best) if best != float("inf") else None),
            "best_epoch": best_epoch, "device": self.device, "last_update_time": time.time(),
            "checkpoint_best": "checkpoints/best.pt", "checkpoint_last": "checkpoints/last.pt",
        })
        diag.log_event("run_end", {"run_id": run_id, "best_epoch": best_epoch})
        print(f"[trainer] done -> {run_dir}")
        return {"run_dir": str(run_dir), "best_metrics": va, "best_epoch": best_epoch,
                "model": model, "standardizer": std, "vocab": vocab}
