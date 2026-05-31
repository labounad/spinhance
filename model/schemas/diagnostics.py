"""
model.schemas.diagnostics
=========================
Typed payloads for the run-directory diagnostics contract. These are lightweight
structures the trainer fills and the DiagnosticsWriter serializes; AutoAI and the
dashboard read the resulting JSON/JSONL. Kept as dataclasses (not free dicts) so
the field names are discoverable and stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunStatus:
    """Atomically-rewritten status.json — the file AutoAI reads first."""
    state: str                      # "running" | "finished" | "failed"
    run_id: str
    epoch: int
    epochs: int
    stage: str
    global_step: int
    best_score: float | None
    best_epoch: int | None
    device: str
    last_update_time: float
    checkpoint_best: str = f"checkpoints/best.pt"
    checkpoint_last: str = f"checkpoints/last.pt"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


@dataclass
class MetricRecord:
    """One row of metrics.jsonl."""
    split: str                      # "train_step" | "train" | "val" | "probe"
    epoch: int
    step: int | None
    metrics: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    """summary.json — written once at run end."""
    run_id: str
    state: str
    best_epoch: int | None
    best_score: float | None
    best_metrics: dict[str, float]
    score_formula: str
    failure_summary: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)
