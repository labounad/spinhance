"""
model.training.surrogate
========================
Supervised trainer for the differentiable surrogate renderer (Branch 5):
surrogate(matrix, field) -> spectrum, fit to the pyspin ground-truth spectrum
with a Wasserstein-1 (+ MSE) spectral loss. Field is alternated per batch (90 /
600 MHz) so the one field-conditioned model learns both.

Writes the canonical run-dir artifact contract (config/status/metrics/events/
summary + checkpoints) so the live dashboard + AutoAI work unchanged. Validation
reports W1 / MSE / cosine vs the ground-truth spectra on held-out molecules — the
fidelity bar the surrogate must clear before it can teach the matrix model.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.data.records import load_records
from model.data.splits import make_splits
from model.data.surrogate_dataset import SurrogateSpectrumDataset, make_surrogate_collate
from model.diagnostics import DiagnosticsWriter
from model.evaluation.spectral_metrics import wasserstein1, smoothed_mse, cosine_similarity
from model.renderers import build_renderer
from model.schemas.constants import N_POINTS, PPM_FROM, PPM_TO
from model.training.optimizer import amp_context, build_optimizer_and_scheduler
from model.training.seed import seed_everything


def _g(cfg: dict, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_device(d):
    return d if d else ("cuda" if torch.cuda.is_available() else "cpu")


def _run_dir(cfg: dict) -> Path:
    if _g(cfg, "run.dir"):
        return Path(_g(cfg, "run.dir"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    import hashlib
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return Path(_g(cfg, "run.output_dir", "model/runs")) / f"{ts}_{_g(cfg,'run.name','surrogate')}_{h}"


class SurrogateTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fields = tuple(int(f) for f in _g(cfg, "data.fields", [90, 600]))
        self.points = int(_g(cfg, "model.points", N_POINTS))
        self.dx = (PPM_TO - PPM_FROM) / self.points
        self.mse_w = float(_g(cfg, "loss.mse_weight", 0.3))
        self.device = _resolve_device(_g(cfg, "training.device"))

    # ── data + model ─────────────────────────────────────────────────────────

    def _build(self):
        cfg = self.cfg
        recs = load_records(_g(cfg, "data.records"), _g(cfg, "data.spectra"), fields=self.fields)
        if _g(cfg, "data.max_mol"):
            recs = recs[: int(_g(cfg, "data.max_mol"))]
        assignment, report = make_splits(recs, seed=int(_g(cfg, "training.seed", 0)),
                                         compute_scaffold=(_g(cfg, "data.split", "none") == "scaffold"))
        by = {"train": [], "val": []}
        for r in recs:
            f = assignment.get(r["mol_id"])
            if f in by:
                by[f].append(r)
        ds = {k: SurrogateSpectrumDataset(v, fields=self.fields) for k, v in by.items()}

        mcfg = {k: v for k, v in (_g(cfg, "model", {}) or {}).items() if k != "name"}
        mcfg.setdefault("points", self.points)
        model = build_renderer("surrogate", **mcfg).to(self.device)
        return ds, model, report

    # ── loss for one (single-field) batch ─────────────────────────────────────

    def _step_loss(self, model, batch, field):
        shifts = batch["shifts"].to(self.device)
        cpl = batch["couplings"].to(self.device)
        deg = batch["degeneracy"].to(self.device)
        target = batch[f"spec{field}"].to(self.device)
        pred = model(shifts, cpl, deg, float(field))
        w1 = wasserstein1(pred, target, dx=self.dx).mean()
        mse = smoothed_mse(pred, target).mean()
        loss = w1 + self.mse_w * mse
        return loss, {"w1": float(w1.detach()), "mse": float(mse.detach()),
                      "cosine": float(cosine_similarity(pred, target).mean().detach())}

    @torch.no_grad()
    def _validate(self, model, loader, amp_ctx):
        model.eval()
        agg, nb = {}, 0
        for batch in loader:
            for field in self.fields:
                with amp_ctx():
                    _, m = self._step_loss(model, batch, field)
                for k, v in m.items():
                    agg[f"{k}_{field}"] = agg.get(f"{k}_{field}", 0.0) + v
            nb += 1
        out = {k: v / max(1, nb) for k, v in agg.items()}
        out["w1"] = sum(out[f"w1_{f}"] for f in self.fields) / len(self.fields)  # mean across fields
        return out

    # ── fit ────────────────────────────────────────────────────────────────--

    def fit(self):
        cfg = self.cfg
        seed_everything(int(_g(cfg, "training.seed", 0)))
        ds, model, report = self._build()

        nw = int(_g(cfg, "training.num_workers", 0))
        bs = int(_g(cfg, "training.batch_size", 256))
        collate = make_surrogate_collate(self.fields)
        pin = self.device != "cpu"
        train_dl = DataLoader(ds["train"], batch_size=bs, shuffle=True, drop_last=True,
                              collate_fn=collate, num_workers=nw, pin_memory=pin)
        val_dl = DataLoader(ds["val"], batch_size=bs, shuffle=False,
                            collate_fn=collate, num_workers=nw, pin_memory=pin)

        epochs = int(_g(cfg, "training.epochs", 80))
        opt, sched = build_optimizer_and_scheduler(
            model, float(_g(cfg, "training.lr", 3e-4)), float(_g(cfg, "training.weight_decay", 1e-2)),
            float(_g(cfg, "training.warmup_frac", 0.03)), max(1, len(train_dl)), epochs)
        amp_ctx, scaler = amp_context(_g(cfg, "training.amp", "bf16"), self.device)
        grad_clip = float(_g(cfg, "training.grad_clip", 1.0))
        log_every = int(_g(cfg, "diagnostics.log_every_steps", 50))
        save_every = int(_g(cfg, "training.save_every", 1))

        run_dir = _run_dir(cfg)
        ckpt_dir = run_dir / "checkpoints"
        diag = DiagnosticsWriter(run_dir, enabled=bool(_g(cfg, "diagnostics.enabled", True)))
        diag.reset_live_files()
        diag.write_config(cfg)
        diag.log_event("run_start", {"run_id": run_dir.name, "device": self.device,
                                     "epochs": epochs, "n_train": len(ds["train"]),
                                     "n_val": len(ds["val"]), "fields": list(self.fields),
                                     "params": sum(p.numel() for p in model.parameters())})
        print(f"[surrogate] run {run_dir.name} | device {self.device} | "
              f"train {len(ds['train'])} val {len(ds['val'])} | "
              f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        best, best_epoch, step, bad = float("inf"), 0, 0, 0
        patience = int(_g(cfg, "training.patience", 0))   # 0 = disabled (run all epochs)
        va = {}
        for epoch in range(epochs):
            model.train()
            running = {}
            for bi, batch in enumerate(train_dl):
                field = self.fields[step % len(self.fields)]   # alternate fields
                opt.zero_grad(set_to_none=True)
                t0 = time.time()
                with amp_ctx():
                    loss, comps = self._step_loss(model, batch, field)
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
                    opt.step()
                sched.step()
                for k, v in comps.items():
                    running[k] = running.get(k, 0.0) + v
                running["loss_total"] = running.get("loss_total", 0.0) + float(loss.detach())
                if diag.enabled and step % log_every == 0:
                    sm = {"loss_total": float(loss.detach()), "lr": float(sched.get_last_lr()[0]),
                          "grad_norm": gnorm, "seconds_per_step": time.time() - t0,
                          "field": float(field), **comps}
                    if torch.cuda.is_available():
                        sm["cuda_allocated_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
                        sm["cuda_reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1e9
                    diag.log_metrics(split="train_step", epoch=epoch, step=step, metrics=sm)
                step += 1
            n = max(1, len(train_dl))
            diag.log_metrics(split="train", epoch=epoch, step=step,
                             metrics={k: v / n for k, v in running.items()})

            if epoch % int(_g(cfg, "training.val_every", 1)) == 0 or epoch == epochs - 1:
                va = self._validate(model, val_dl, amp_ctx)
                diag.log_metrics(split="val", epoch=epoch, step=step, metrics=va)

            score = va.get("w1", float("inf"))
            is_best = score < best
            if is_best:
                best, best_epoch, bad = score, epoch, 0
                diag.log_event("best_checkpoint", {"epoch": epoch, "w1": float(score)})
            else:
                bad += 1

            self._save(ckpt_dir / "last.pt", model, epoch, va)
            if is_best:
                self._save(ckpt_dir / "best.pt", model, epoch, va)
            if save_every and epoch % save_every == 0:
                self._save(ckpt_dir / f"epoch_{epoch:04d}.pt", model, epoch, va)

            diag.update_status({"state": "running", "run_id": run_dir.name, "epoch": epoch,
                                "epochs": epochs, "stage": "surrogate", "global_step": step,
                                "best_score": (float(best) if best != float("inf") else None),
                                "best_epoch": best_epoch, "device": self.device,
                                "last_update_time": time.time(),
                                "checkpoint_best": "checkpoints/best.pt",
                                "checkpoint_last": "checkpoints/last.pt"})
            print(f"epoch {epoch:3d} | train {running.get('loss_total',0)/n:.4f}"
                  + (f" | val W1 {va.get('w1',0):.4f} "
                     f"(90 {va.get('w1_90',0):.4f} / 600 {va.get('w1_600',0):.4f}) "
                     f"cos90 {va.get('cosine_90',0):.3f}" if va else "")
                  + (f" | no-improve {bad}/{patience}" if patience else ""))

            if patience and bad >= patience:
                diag.log_event("early_stop", {"epoch": epoch, "patience": patience,
                                              "best_epoch": best_epoch})
                print(f"early stop at epoch {epoch} (best epoch {best_epoch}, W1 {best:.4f})")
                break

        summary = {"run_id": run_dir.name, "state": "finished", "best_epoch": best_epoch,
                   "best_score": (float(best) if best != float("inf") else None),
                   "best_metrics": va, "score_formula": "mean W1 over fields (lower better)",
                   "split": report["counts"]}
        diag.finalize(summary)
        diag.update_status({"state": "finished", "run_id": run_dir.name, "epoch": best_epoch,
                            "epochs": epochs, "stage": "surrogate", "global_step": step,
                            "best_score": (float(best) if best != float("inf") else None),
                            "best_epoch": best_epoch, "device": self.device,
                            "last_update_time": time.time(),
                            "checkpoint_best": "checkpoints/best.pt",
                            "checkpoint_last": "checkpoints/last.pt"})
        diag.log_event("run_end", {"run_id": run_dir.name, "best_epoch": best_epoch})
        print(f"[surrogate] done -> {run_dir}")
        return {"run_dir": str(run_dir), "best_metrics": va, "model": model}

    def _save(self, path, model, epoch, metrics):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "cfg": self.cfg, "epoch": epoch, "metrics": metrics}, path)
