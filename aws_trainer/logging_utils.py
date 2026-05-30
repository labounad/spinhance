"""
aws_trainer.logging_utils
==========================
RunLogger: W&B (if available) + CSV (always).

Usage:
    logger = RunLogger(cfg, is_main=True)
    logger.log(epoch, {"shift_mae": 0.12, "j_mae": 1.4})
    logger.finish()
"""

from __future__ import annotations

import csv
import time
from pathlib import Path


class RunLogger:
    def __init__(self, cfg, is_main: bool = True):
        self._main = is_main
        self._wb = None
        self._csv_path: Path | None = None
        self._csv_writer = None
        self._csv_file = None
        self._start = time.monotonic()

        if not is_main:
            return

        log_dir = Path(cfg.log_dir)
        run_name = cfg.run_name or f"{cfg.model_size}_{int(time.time())}"
        self._run_dir = log_dir / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # CSV
        self._csv_path = self._run_dir / "metrics.csv"

        # W&B
        if cfg.wandb_project:
            try:
                import wandb
                self._wb = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity or None,
                    name=run_name,
                    config=_cfg_to_dict(cfg),
                    dir=str(self._run_dir),
                )
            except Exception as e:
                print(f"[logger] W&B init failed ({e}); CSV only.")

    def log(self, epoch: int, metrics: dict) -> None:
        if not self._main:
            return
        elapsed = time.monotonic() - self._start
        row = {"epoch": epoch, "elapsed_s": f"{elapsed:.0f}", **metrics}

        # CSV
        if self._csv_file is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                                   for k, v in row.items()})
        self._csv_file.flush()

        # W&B
        if self._wb is not None:
            try:
                self._wb.log(metrics, step=epoch)
            except Exception:
                pass

    def print_epoch(self, epoch: int, stage: int, w_mat: float, w_spec: float,
                    tr: dict, va: dict) -> None:
        if not self._main:
            return
        elapsed = (time.monotonic() - self._start) / 60
        print(
            f"ep {epoch:3d} | s{stage} | w_mat={w_mat:.2f} w_spec={w_spec:.2f} | "
            f"tr={tr.get('total', 0):.4f} | "
            f"va shift={va.get('shift_mae_ppm', 0):.3f}ppm "
            f"J={va.get('j_mae_hz', 0):.2f}Hz "
            f"pF1={va.get('presence_f1', 0):.3f} "
            f"deg={va.get('deg_acc_balanced', 0):.3f} "
            f"h_shift={va.get('h_shift_mae_ppm', 0):.3f}ppm "
            f"| {elapsed:.1f}min",
            flush=True)

    def finish(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        if self._wb is not None:
            try:
                self._wb.finish()
            except Exception:
                pass

    @property
    def run_dir(self) -> Path | None:
        return getattr(self, "_run_dir", None)


def _cfg_to_dict(cfg) -> dict:
    from dataclasses import asdict
    try:
        return asdict(cfg)
    except Exception:
        return {}
