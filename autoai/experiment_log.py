"""
autoai/experiment_log.py — persistent JSONL experiment log.

One record per worker delegation:
  run_id, timestamp, status, task_spec, artifact_paths, metrics, errors, code_hash, lesson
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

LOG_FILE = Path(__file__).parent / "experiment_log.jsonl"


@dataclass
class ExperimentRecord:
    run_id:         str
    timestamp:      str
    status:         str           # "success" | "failure" | "partial"
    task_spec:      dict          # TaskSpec.to_dict()
    artifact_paths: dict          # WorkerResult.artifact_paths
    metrics:        dict          # read from disk — never worker-typed
    errors:         str | None
    code_hash:      str           # first 12 chars of SHA-256 of training script, "" if none
    lesson:         str           # one-line takeaway for the orchestrator


def append_record(record: ExperimentRecord) -> None:
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_records(n: int | None = None) -> list[ExperimentRecord]:
    if not LOG_FILE.exists():
        return []
    records = [
        ExperimentRecord(**json.loads(line))
        for line in LOG_FILE.read_text().splitlines()
        if line.strip()
    ]
    return records[-n:] if n is not None else records


def summarize_for_context(n: int = 5) -> str:
    """Return a compact markdown table of the last n experiment records."""
    records = load_records(n)
    if not records:
        return "_No experiments logged yet._"

    lines = ["| run | status | key metrics | lesson |",
             "|-----|--------|-------------|--------|"]
    for r in records:
        metrics_str = "  ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(r.metrics.items())[:3]
        ) or "—"
        lines.append(
            f"| {r.run_id} | {r.status} | {metrics_str} | {r.lesson} |"
        )
    return "\n".join(lines)
