"""
model.diagnostics
==================
Structured run-artifact writer. Produces append-only JSONL metrics, atomic
JSON status, and a final summary.json — consumed by the live dashboard,
autoai/run_reader.py, and any external monitoring tool.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class DiagnosticsWriter:
    """Write structured training artifacts to a run directory.

    All writes are either atomic (JSON) or append-only (JSONL), so a crash
    mid-epoch leaves existing data intact.
    """

    def __init__(self, run_dir: str | Path, enabled: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── Primitives ─────────────────────────────────────────────────────────────

    def write_json_atomic(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self.run_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"time": time.time(), **payload}
        with open(self.run_dir / name, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # ── High-level writers ─────────────────────────────────────────────────────

    def log_metrics(
        self,
        *,
        split: str,
        epoch: int,
        step: int | None,
        metrics: dict[str, float],
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.append_jsonl("metrics.jsonl", {
            "kind": "metrics",
            "split": split,
            "epoch": epoch,
            "step": step,
            "metrics": metrics,
            **(extra or {}),
        })

    def log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        self.append_jsonl("events.jsonl", {
            "kind": "event",
            "event": event,
            "payload": payload or {},
        })

    def update_status(self, payload: dict[str, Any]) -> None:
        self.write_json_atomic("status.json", payload)

    def write_config(self, cfg_dict: dict[str, Any]) -> None:
        self.write_json_atomic("config.json", cfg_dict)

    def finalize(self, summary: dict[str, Any]) -> None:
        self.write_json_atomic("summary.json", summary)
