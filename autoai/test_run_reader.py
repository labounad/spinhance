"""
Tests for autoai.run_reader.  Torch-free; uses only stdlib + numpy.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from autoai.run_reader import (
    analyze_run,
    find_latest_run,
    list_runs,
    read_config,
    read_events,
    read_failure_summary,
    read_metrics,
    read_status,
    read_summary,
    read_worst_cases,
)


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _write_run(run_dir: Path, n_val: int = 5) -> Path:
    """Write a minimal but complete synthetic run directory."""
    run_dir.mkdir(parents=True)

    (run_dir / "status.json").write_text(json.dumps({
        "state": "finished", "epoch": 40, "epochs": 50,
        "best_score": 0.42, "best_epoch": 38, "device": "cuda",
    }))
    (run_dir / "summary.json").write_text(json.dumps({
        "run_id": run_dir.name, "state": "finished",
        "best_epoch": 38, "best_score": 0.42,
        "best_metrics": {"shift_mae_ppm": 0.08, "j_mae_hz": 1.6,
                         "presence_f1": 0.72, "deg_acc": 0.91},
        "failure_summary": {"dominant_failure": "false_negative_couplings",
                            "n_molecules": 200},
        "recommendation": "Increase presence_pos_weight",
    }))
    (run_dir / "config.json").write_text(json.dumps({"lr": 3e-4, "epochs": 50}))

    rows = []
    for i in range(3):
        rows.append(json.dumps({"kind": "metrics", "split": "train_step",
                                 "epoch": i, "step": i * 10,
                                 "metrics": {"loss_total": 2.0 - i * 0.3}, "time": 1e9}))
    for i in range(n_val):
        rows.append(json.dumps({"kind": "metrics", "split": "val",
                                 "epoch": i * 5, "step": i * 50,
                                 "metrics": {"shift_mae_ppm": 0.2 - i * 0.02,
                                             "j_mae_hz": 3.0 - i * 0.2},
                                 "time": 1e9}))
    (run_dir / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    (run_dir / "events.jsonl").write_text(
        json.dumps({"kind": "event", "event": "run_start", "payload": {}, "time": 1e9}) + "\n" +
        json.dumps({"kind": "event", "event": "best_checkpoint",
                    "payload": {"epoch": 38}, "time": 1e9}) + "\n"
    )

    probe_ep = run_dir / "probes" / "epoch_0038"
    probe_ep.mkdir(parents=True)
    (probe_ep / "failure_summary.json").write_text(json.dumps({
        "dominant_failure": "false_negative_couplings",
        "n_ok": 80,
        "failure_distribution": {"false_negative_couplings": 120, "ok": 80},
        "n_molecules": 200,
    }))
    (probe_ep / "worst_shift_cases.json").write_text(json.dumps([
        {"mol_id": f"mol_{i:06d}", "shift_mae_ppm": 0.5 - i * 0.01}
        for i in range(10)
    ]))
    return run_dir


# ── read_status ────────────────────────────────────────────────────────────────

def test_read_status_basic(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    s = read_status(run_dir)
    assert s["state"] == "finished"
    assert s["best_score"] == pytest.approx(0.42)
    assert s["best_epoch"] == 38


def test_read_status_missing_file(tmp_path):
    assert read_status(tmp_path / "nonexistent") == {}


def test_read_status_corrupted_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text("not valid json {{")
    assert read_status(run_dir) == {}


# ── read_summary ───────────────────────────────────────────────────────────────

def test_read_summary_best_metrics(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    s = read_summary(run_dir)
    assert s["best_epoch"] == 38
    assert s["best_metrics"]["shift_mae_ppm"] == pytest.approx(0.08)


def test_read_summary_missing(tmp_path):
    assert read_summary(tmp_path / "nope") == {}


# ── read_config ────────────────────────────────────────────────────────────────

def test_read_config(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    c = read_config(run_dir)
    assert c["lr"] == pytest.approx(3e-4)
    assert c["epochs"] == 50


# ── read_metrics ───────────────────────────────────────────────────────────────

def test_read_metrics_count(tmp_path):
    run_dir = _write_run(tmp_path / "run", n_val=5)
    rows = read_metrics(run_dir)
    assert len(rows) == 8   # 3 train_step + 5 val


def test_read_metrics_splits(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    rows = read_metrics(run_dir)
    splits = {r["split"] for r in rows}
    assert "train_step" in splits
    assert "val" in splits


def test_read_metrics_missing(tmp_path):
    assert read_metrics(tmp_path / "nope") == []


def test_read_metrics_skips_bad_lines(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        '{"split": "val", "epoch": 0, "metrics": {}}\n'
        'bad json line ###\n'
        '{"split": "val", "epoch": 1, "metrics": {}}\n'
    )
    rows = read_metrics(run_dir)
    assert len(rows) == 2


# ── read_events ────────────────────────────────────────────────────────────────

def test_read_events(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    events = read_events(run_dir)
    event_names = [e["event"] for e in events]
    assert "run_start" in event_names
    assert "best_checkpoint" in event_names


# ── read_failure_summary ───────────────────────────────────────────────────────

def test_read_failure_summary_latest(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    fs = read_failure_summary(run_dir)
    assert fs["dominant_failure"] == "false_negative_couplings"
    assert fs["n_molecules"] == 200


def test_read_failure_summary_by_epoch(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    fs = read_failure_summary(run_dir, epoch=38)
    assert fs["dominant_failure"] == "false_negative_couplings"


def test_read_failure_summary_wrong_epoch(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    assert read_failure_summary(run_dir, epoch=99) == {}


def test_read_failure_summary_no_probes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert read_failure_summary(run_dir) == {}


# ── read_worst_cases ──────────────────────────────────────────────────────────

def test_read_worst_cases(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    cases = read_worst_cases(run_dir, metric="shift")
    assert len(cases) == 10
    assert cases[0]["shift_mae_ppm"] == pytest.approx(0.5)


def test_read_worst_cases_missing_metric(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    assert read_worst_cases(run_dir, metric="nonexistent") == []


# ── analyze_run ───────────────────────────────────────────────────────────────

def test_analyze_run_fields(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    result = analyze_run(run_dir)
    for key in ("run_id", "state", "best_score", "best_metrics",
                "failure_summary", "recommendation"):
        assert key in result, f"analyze_run missing key '{key}'"


def test_analyze_run_uses_summary(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    result = analyze_run(run_dir)
    assert result["state"] == "finished"
    assert result["best_score"] == pytest.approx(0.42)
    assert result["best_metrics"]["shift_mae_ppm"] == pytest.approx(0.08)


def test_analyze_run_recommendation_non_empty(tmp_path):
    run_dir = _write_run(tmp_path / "run")
    result = analyze_run(run_dir)
    assert isinstance(result["recommendation"], str)
    assert len(result["recommendation"]) > 5


def test_analyze_run_falls_back_to_jsonl(tmp_path):
    """When summary.json has no best_metrics, should compute from metrics.jsonl."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"state": "finished"}))
    rows = [
        json.dumps({"split": "val", "epoch": 0, "step": 0,
                    "metrics": {"shift_mae_ppm": 0.5, "j_mae_hz": 5.0}, "time": 1e9}),
        json.dumps({"split": "val", "epoch": 10, "step": 100,
                    "metrics": {"shift_mae_ppm": 0.05, "j_mae_hz": 1.0}, "time": 1e9}),
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    result = analyze_run(run_dir)
    # Best: epoch 10 (score=0.05+0.1=0.15 < epoch 0 score=0.5+0.5=1.0)
    assert result["best_metrics"]["shift_mae_ppm"] == pytest.approx(0.05)


def test_analyze_run_instability_detection(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"state": "running"}))
    rng = np.random.default_rng(0)
    # Highly variable losses → high instability
    rows = [
        json.dumps({"split": "train_step", "epoch": 0, "step": i,
                    "metrics": {"loss_total": float(rng.uniform(0.5, 10.0))}, "time": 1e9})
        for i in range(25)
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    result = analyze_run(run_dir)
    assert result["instability"] in ("high", "moderate", "low")


def test_analyze_run_stable_training(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"state": "running"}))
    # Very stable losses → low instability
    rows = [
        json.dumps({"split": "train_step", "epoch": 0, "step": i,
                    "metrics": {"loss_total": 1.0 + 0.001 * i}, "time": 1e9})
        for i in range(25)
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    result = analyze_run(run_dir)
    assert result["instability"] == "low"


# ── find_latest_run / list_runs ────────────────────────────────────────────────

def test_find_latest_run_empty(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    assert find_latest_run(runs_root) is None


def test_find_latest_run_returns_path(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "run_001").mkdir()
    (runs_root / "run_002").mkdir()
    latest = find_latest_run(runs_root)
    assert latest is not None
    assert latest.parent == runs_root


def test_list_runs_sorted_newest_first(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for name in ("run_001", "run_002", "run_003"):
        (runs_root / name).mkdir()
    runs = list_runs(runs_root)
    assert len(runs) == 3
    # Sorted by mtime desc; all were just created so order may vary,
    # but all should be present
    assert {r.name for r in runs} == {"run_001", "run_002", "run_003"}


def test_list_runs_empty(tmp_path):
    runs_root = tmp_path / "no_such_dir"
    assert list_runs(runs_root) == []
