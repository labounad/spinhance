"""
autoai/contract.py — typed contract between orchestrator and worker.

TaskSpec  : orchestrator → worker  (what to do)
WorkerResult: worker → orchestrator (what happened, grounded in artifacts)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskSpec:
    objective:        str
    loss_function:    str
    output_artifacts: list[str]       # repo-relative paths the worker MUST produce
    success_criteria: str
    architecture:     str             = ""
    training_config:  dict            = field(default_factory=dict)
    constraints:      list[str]       = field(default_factory=list)
    notes:            str             = ""

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(
            objective        = d["objective"],
            loss_function    = d["loss_function"],
            output_artifacts = d["output_artifacts"],
            success_criteria = d["success_criteria"],
            architecture     = d.get("architecture", ""),
            training_config  = d.get("training_config", {}),
            constraints      = d.get("constraints", []),
            notes            = d.get("notes", ""),
        )

    def to_dict(self) -> dict:
        return {
            "objective":        self.objective,
            "loss_function":    self.loss_function,
            "output_artifacts": self.output_artifacts,
            "success_criteria": self.success_criteria,
            "architecture":     self.architecture,
            "training_config":  self.training_config,
            "constraints":      self.constraints,
            "notes":            self.notes,
        }


@dataclass
class WorkerResult:
    status:         str              # "success" | "failure" | "partial"
    artifact_paths: dict[str, str]   # artifact name → repo-relative path
    metrics:        dict             # populated by reading artifact_paths["metrics"] — never worker-typed
    errors:         str | None
    notes:          str              # untrusted worker prose

    @classmethod
    def from_submission(
        cls,
        status:         str,
        artifact_paths: dict[str, str],
        errors:         str | None,
        notes:          str,
        repo_root:      Path,
    ) -> "WorkerResult":
        """
        Build a WorkerResult by reading the metrics file from disk.
        The worker declares a path; we read the values — preventing fabrication.
        """
        metrics: dict = {}
        metrics_path = artifact_paths.get("metrics")
        if metrics_path:
            p = repo_root / metrics_path
            if p.exists():
                try:
                    metrics = json.loads(p.read_text())
                except Exception as e:
                    metrics = {"parse_error": str(e), "path": metrics_path}
            else:
                metrics = {"missing": metrics_path}
        return cls(
            status         = status,
            artifact_paths = artifact_paths,
            metrics        = metrics,
            errors         = errors,
            notes          = notes,
        )

    def to_dict(self) -> dict:
        return {
            "status":         self.status,
            "artifact_paths": self.artifact_paths,
            "metrics":        self.metrics,
            "errors":         self.errors,
            "notes":          self.notes,
        }
