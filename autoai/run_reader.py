"""
autoai.run_reader
==================
Read structured run artifacts from model/diagnostics.py.
Gives the AutoAI orchestrator machine-readable access to training runs
instead of parsing raw log text.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO / "model" / "runs"


# ── Primitive readers ──────────────────────────────────────────────────────────

def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ── Per-file accessors ─────────────────────────────────────────────────────────

def read_status(run_dir) -> dict:
    return _json(Path(run_dir) / "status.json", {})


def read_summary(run_dir) -> dict:
    return _json(Path(run_dir) / "summary.json", {})


def read_config(run_dir) -> dict:
    return _json(Path(run_dir) / "config.json", {})


def read_metrics(run_dir) -> list[dict]:
    return _jsonl(Path(run_dir) / "metrics.jsonl")


def read_events(run_dir) -> list[dict]:
    return _jsonl(Path(run_dir) / "events.jsonl")


def read_failure_summary(run_dir, epoch: int | None = None) -> dict:
    """Return the most recent (or specified) epoch's failure_summary.json."""
    probes = Path(run_dir) / "probes"
    if not probes.exists():
        return {}
    dirs = sorted(probes.iterdir(), reverse=True)
    if epoch is not None:
        dirs = [d for d in dirs if d.name == f"epoch_{epoch:04d}"]
    if not dirs:
        return {}
    return _json(dirs[0] / "failure_summary.json", {})


def read_worst_cases(run_dir, metric: str = "shift", epoch: int | None = None) -> list[dict]:
    """Load worst_<metric>_cases.json from the most recent (or given) probe epoch."""
    fname = f"worst_{metric}_cases.json"
    probes = Path(run_dir) / "probes"
    if not probes.exists():
        return []
    dirs = sorted(probes.iterdir(), reverse=True)
    if epoch is not None:
        dirs = [d for d in dirs if d.name == f"epoch_{epoch:04d}"]
    for d in dirs:
        cases = _json(d / fname)
        if cases is not None:
            return cases
    return []


# ── Run analysis ───────────────────────────────────────────────────────────────

_FAILURE_HINTS: dict[str, str] = {
    "large_shift_error":          "Increase shift loss weight or use Hungarian matching loss",
    "false_negative_couplings":   "Increase presence_pos_weight; check BCE weight for minority class",
    "false_positive_couplings":   "Lower presence threshold or up-weight absence class",
    "bad_j_magnitude":            "Increase j_mag loss weight; verify masked Huber covers true-present pairs",
    "wrong_degeneracy":           "Check degeneracy vocab; try integration-aware token features",
    "ok":                         "Metrics look healthy — consider Hungarian loss or spectral consistency",
    "none":                       "No probe data yet",
}


def analyze_run(run_dir) -> dict:
    """Return a machine-readable analysis dict for the AutoAI orchestrator."""
    run_dir = Path(run_dir)
    status  = read_status(run_dir)
    summary = read_summary(run_dir)
    failure = read_failure_summary(run_dir)

    # Best metrics: from summary, else compute from metrics.jsonl
    best_metrics = summary.get("best_metrics", {})
    if not best_metrics:
        rows = read_metrics(run_dir)
        val_rows = [r for r in rows if r.get("split") == "val"]
        if val_rows:
            def _score(r):
                m = r.get("metrics", {})
                return (m.get("shift_mae_ppm", 999) + m.get("j_mae_hz", 999) / 10.0)
            best_row = min(val_rows, key=_score)
            best_metrics = best_row.get("metrics", {})

    # Training instability: coefficient of variation of recent train-step loss
    rows = read_metrics(run_dir)
    train_rows = [r for r in rows if r.get("split") == "train_step"]
    instability = None
    if len(train_rows) >= 20:
        losses = [r["metrics"]["loss_total"] for r in train_rows[-20:]
                  if "metrics" in r and "loss_total" in r.get("metrics", {})]
        if len(losses) >= 10:
            cv = float(np.std(losses) / (np.mean(losses) + 1e-9))
            instability = "high" if cv > 0.5 else "moderate" if cv > 0.2 else "low"

    dominant    = failure.get("dominant_failure", "none")
    hint        = _FAILURE_HINTS.get(dominant, _FAILURE_HINTS["ok"])

    return {
        "run_id":         run_dir.name,
        "state":          status.get("state", "unknown"),
        "best_epoch":     status.get("best_epoch"),
        "best_score":     status.get("best_score"),
        "best_metrics":   best_metrics,
        "failure_summary": failure,
        "instability":    instability,
        "recommendation": hint,
    }


def find_latest_run(runs_root: str | Path | None = None) -> Path | None:
    root = Path(runs_root) if runs_root else RUNS_ROOT
    if not root.exists():
        return None
    dirs = [d for d in root.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def list_runs(runs_root: str | Path | None = None) -> list[Path]:
    root = Path(runs_root) if runs_root else RUNS_ROOT
    if not root.exists():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime, reverse=True)
