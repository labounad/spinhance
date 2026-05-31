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
# ── Artifact-path integration ─────────────────────────────────────────────────

def _repo_path(path: str | Path, repo_root: str | Path | None = None) -> Path:
    p = Path(path)
    root = Path(repo_root) if repo_root is not None else REPO
    return p if p.is_absolute() else root / p


def infer_run_dir_from_artifacts(
    artifact_paths: dict,
    repo_root: str | Path | None = None,
) -> Path | None:
    """Infer a model/runs/<run_id> directory from worker artifact paths.

    Preferred explicit keys:
      - run_dir
      - train_run
      - diagnostics_run_dir

    Fallbacks:
      - checkpoint path inside a checkpoints/ directory
      - metrics/status/summary path inside a canonical run directory
    """
    root = Path(repo_root) if repo_root is not None else REPO

    for key in ("run_dir", "train_run", "diagnostics_run_dir"):
        value = artifact_paths.get(key)
        if value:
            candidate = _repo_path(value, root)
            if candidate.exists() and candidate.is_dir():
                return candidate

    for key in ("checkpoint", "best_checkpoint", "last_checkpoint"):
        value = artifact_paths.get(key)
        if not value:
            continue

        p = _repo_path(value, root)
        parts = p.parts
        if "checkpoints" in parts:
            idx = parts.index("checkpoints")
            candidate = Path(*parts[:idx])
            if candidate.exists() and (candidate / "metrics.jsonl").exists():
                return candidate

    for key in ("metrics", "summary", "status", "log"):
        value = artifact_paths.get(key)
        if not value:
            continue

        p = _repo_path(value, root)
        if p.name in {"metrics.jsonl", "summary.json", "status.json"}:
            candidate = p.parent
            if candidate.exists() and (candidate / "metrics.jsonl").exists():
                return candidate

        if p.name == "metrics.json" and p.parent.exists():
            # AutoAI cycle metrics file, not necessarily the training run.
            nested = p.parent / "train_run"
            if nested.exists() and (nested / "metrics.jsonl").exists():
                return nested

    return None


def latest_probe_epoch(run_dir: str | Path) -> str | None:
    probes = Path(run_dir) / "probes"
    if not probes.exists():
        return None

    dirs = sorted([d for d in probes.iterdir() if d.is_dir()])
    return dirs[-1].name if dirs else None


def summarize_metrics_for_agent(metrics: dict) -> dict:
    """Compact metric subset for experiment-log display and model-selection prompts."""
    keys = (
        "shift_mae_ppm",
        "h_shift_mae_ppm",
        "j_mae_hz",
        "h_j_mae_hz",
        "presence_f1",
        "deg_acc",
        "deg_acc_balanced",
        "matrix_loss",
    )
    return {k: metrics[k] for k in keys if k in metrics}


def analyze_artifact_paths(
    artifact_paths: dict,
    repo_root: str | Path | None = None,
) -> dict:
    """Analyze canonical training diagnostics from a WorkerResult artifact map."""
    run_dir = infer_run_dir_from_artifacts(artifact_paths, repo_root=repo_root)
    if run_dir is None:
        return {
            "available": False,
            "reason": "no canonical model run directory found in artifact_paths",
        }

    analysis = analyze_run(run_dir)
    analysis["available"] = True
    analysis["run_dir"] = str(run_dir)
    analysis["latest_probe_epoch"] = latest_probe_epoch(run_dir)

    best_metrics = analysis.get("best_metrics") or {}
    analysis["compact_best_metrics"] = summarize_metrics_for_agent(best_metrics)

    failure = analysis.get("failure_summary") or {}
    analysis["dominant_failure"] = failure.get("dominant_failure", "none")

    return analysis


def write_analysis_json(
    run_dir: str | Path,
    out_path: str | Path,
) -> dict:
    """Analyze a run directory and write the result to JSON."""
    out_path = Path(out_path)
    analysis = analyze_run(run_dir)
    analysis["available"] = True
    analysis["run_dir"] = str(run_dir)
    analysis["latest_probe_epoch"] = latest_probe_epoch(run_dir)
    analysis["compact_best_metrics"] = summarize_metrics_for_agent(
        analysis.get("best_metrics") or {}
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    return analysis
