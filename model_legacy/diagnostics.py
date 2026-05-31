"""
model.diagnostics
==================
Structured run-artifact writer. Produces append-only JSONL metrics, atomic
JSON status, and a final summary.json — consumed by the live dashboard,
autoai/run_reader.py, and any external monitoring tool.

Supports both local-filesystem runs (run_dir is a local path) and S3-backed
runs (run_dir is an ``s3://`` URI).  The backend is selected automatically
from the run_dir prefix.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class DiagnosticsWriter:
    """Write structured training artifacts to a run directory or S3 prefix.

    All writes are either atomic (JSON) or append-only (JSONL), so a crash
    mid-epoch leaves existing data intact.

    Parameters
    ----------
    run_dir:
        Local filesystem path **or** ``s3://bucket/prefix`` URI.
        S3 URIs are detected by the ``s3://`` prefix; everything else is
        treated as a local path.  Local paths are used for tests and smoke
        runs; S3 URIs are used for cloud training sessions.
    """

    def __init__(self, run_dir: str | Path, enabled: bool = True) -> None:
        self.run_dir = str(run_dir)
        self.enabled = enabled
        self._s3 = self.run_dir.startswith("s3://")
        if self._s3:
            self._buffers: dict[str, list[str]] = {}
        elif self.enabled:
            Path(self.run_dir).mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _uri(self, name: str) -> str:
        return f"{self.run_dir.rstrip('/')}/{name}"

    # ── Primitives ─────────────────────────────────────────────────────────────

    def write_json_atomic(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if self._s3:
            from model_legacy import s3io
            s3io.put_json(self._uri(name), payload)
        else:
            path = Path(self.run_dir) / name
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, path)

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = {"time": time.time(), **payload}
        if self._s3:
            from model_legacy import s3io
            line = json.dumps(record, sort_keys=True)
            buf = self._buffers.setdefault(name, [])
            buf.append(line)
            content = "\n".join(buf) + "\n"
            s3io.put_bytes(self._uri(name), content.encode(),
                           "application/x-ndjson")
        else:
            with open(Path(self.run_dir) / name, "a") as f:
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

    def reset_live_files(self) -> None:
        """Clear live append-only/status artifacts at the beginning of a fresh run.

        Prevents a new run from appending onto stale metrics.jsonl rows.
        Checkpoints are intentionally not deleted here.
        """
        if not self.enabled:
            return

        if self._s3:
            from model_legacy import s3io
            self._buffers.clear()
            for name in ("metrics.jsonl", "events.jsonl", "system.jsonl",
                         "status.json", "summary.json"):
                s3io.delete(self._uri(name))
            s3io.delete_prefix(self._uri("probes"))
        else:
            for name in ("metrics.jsonl", "events.jsonl", "system.jsonl",
                         "status.json", "summary.json"):
                path = Path(self.run_dir) / name
                if path.exists():
                    path.unlink()
            probes_dir = Path(self.run_dir) / "probes"
            if probes_dir.exists():
                import shutil
                shutil.rmtree(probes_dir)
