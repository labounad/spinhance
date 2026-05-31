"""
Tests for model.diagnostics (DiagnosticsWriter) and the non-torch parts of
model.probes (_probe_indices).  Torch-free; uses only stdlib + numpy.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from model.diagnostics import DiagnosticsWriter


# ── DiagnosticsWriter ──────────────────────────────────────────────────────────

def test_run_dir_created_on_init(tmp_path):
    run_dir = tmp_path / "deeply" / "nested" / "run"
    DiagnosticsWriter(run_dir)
    assert run_dir.exists()


def test_disabled_creates_no_dir(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run", enabled=False)
    d.write_json_atomic("status.json", {"x": 1})
    d.append_jsonl("metrics.jsonl", {"x": 1})
    d.log_metrics(split="val", epoch=0, step=0, metrics={"a": 1.0})
    d.log_event("test")
    d.update_status({"x": 1})
    d.write_config({"lr": 1e-3})
    d.finalize({"best": 0.5})
    assert not (tmp_path / "run").exists()


def test_write_json_atomic_content(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.write_json_atomic("status.json", {"state": "running", "epoch": 5})
    data = json.loads((tmp_path / "run" / "status.json").read_text())
    assert data["state"] == "running"
    assert data["epoch"] == 5


def test_write_json_atomic_no_tmp_leftover(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.write_json_atomic("status.json", {"x": 1})
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_write_json_atomic_overwrites(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.write_json_atomic("status.json", {"epoch": 0})
    d.write_json_atomic("status.json", {"epoch": 99})
    data = json.loads((tmp_path / "run" / "status.json").read_text())
    assert data["epoch"] == 99


def test_append_jsonl_creates_file(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.append_jsonl("events.jsonl", {"event": "start"})
    assert (tmp_path / "run" / "events.jsonl").exists()


def test_append_jsonl_accumulates(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    for i in range(5):
        d.append_jsonl("metrics.jsonl", {"step": i})
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().strip().split("\n")
    assert len(lines) == 5
    for i, line in enumerate(lines):
        assert json.loads(line)["step"] == i


def test_append_jsonl_injects_timestamp(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.append_jsonl("metrics.jsonl", {"x": 1})
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text())
    assert "time" in row
    assert isinstance(row["time"], float)


def test_log_metrics_structure(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.log_metrics(split="val", epoch=3, step=100,
                  metrics={"shift_mae_ppm": 0.15, "j_mae_hz": 2.1},
                  extra={"stage": 2})
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text())
    assert row["kind"] == "metrics"
    assert row["split"] == "val"
    assert row["epoch"] == 3
    assert row["step"] == 100
    assert row["metrics"]["shift_mae_ppm"] == pytest.approx(0.15)
    assert row["stage"] == 2


def test_log_metrics_extra_none(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.log_metrics(split="train", epoch=0, step=0, metrics={"loss": 1.0})
    row = json.loads((tmp_path / "run" / "metrics.jsonl").read_text())
    assert row["kind"] == "metrics"


def test_log_event_with_payload(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.log_event("run_start", {"device": "cuda", "epochs": 100})
    row = json.loads((tmp_path / "run" / "events.jsonl").read_text())
    assert row["kind"] == "event"
    assert row["event"] == "run_start"
    assert row["payload"]["device"] == "cuda"


def test_log_event_no_payload_defaults_to_empty(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.log_event("early_stop")
    row = json.loads((tmp_path / "run" / "events.jsonl").read_text())
    assert row["payload"] == {}


def test_update_status_is_atomic(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.update_status({"state": "running", "epoch": 0})
    d.update_status({"state": "running", "epoch": 7})
    data = json.loads((tmp_path / "run" / "status.json").read_text())
    assert data["epoch"] == 7
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_write_config(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.write_config({"lr": 3e-4, "epochs": 100, "amp_dtype": "bf16"})
    cfg = json.loads((tmp_path / "run" / "config.json").read_text())
    assert cfg["epochs"] == 100
    assert cfg["lr"] == pytest.approx(3e-4)


def test_finalize(tmp_path):
    d = DiagnosticsWriter(tmp_path / "run")
    d.finalize({"run_id": "abc", "best_epoch": 42, "best_score": 0.31,
                "best_metrics": {"shift_mae_ppm": 0.05}})
    s = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert s["best_epoch"] == 42
    assert s["best_metrics"]["shift_mae_ppm"] == pytest.approx(0.05)


def test_multiple_appends_survive_multiple_writers(tmp_path):
    # Two DiagnosticsWriter instances pointing at the same run_dir
    # (simulates resume or concurrent writes — appends are safe)
    d1 = DiagnosticsWriter(tmp_path / "run")
    d2 = DiagnosticsWriter(tmp_path / "run")
    d1.log_event("first")
    d2.log_event("second")
    lines = (tmp_path / "run" / "events.jsonl").read_text().strip().split("\n")
    events = [json.loads(l)["event"] for l in lines]
    assert "first" in events and "second" in events


# ── _probe_indices (non-torch) ─────────────────────────────────────────────────

def test_probe_indices_length(tmp_path):
    from model.probes import _probe_indices
    rng = np.random.default_rng(0)
    records = [
        {"shifts": rng.uniform(0, 10, 8),
         "degeneracy": rng.choice([1, 2, 3], 8)}
        for _ in range(80)
    ]
    idxs = _probe_indices(records, 16)
    assert len(idxs) == 16


def test_probe_indices_no_duplicates(tmp_path):
    from model.probes import _probe_indices
    rng = np.random.default_rng(0)
    records = [
        {"shifts": rng.uniform(0, 10, 8),
         "degeneracy": rng.choice([1, 2, 3], 8)}
        for _ in range(60)
    ]
    idxs = _probe_indices(records, 12)
    assert len(set(idxs)) == len(idxs)


def test_probe_indices_valid_range(tmp_path):
    from model.probes import _probe_indices
    rng = np.random.default_rng(1)
    n = 50
    records = [{"shifts": rng.uniform(0, 10, 8), "degeneracy": np.array([1]*8)} for _ in range(n)]
    idxs = _probe_indices(records, 10)
    assert all(0 <= i < n for i in idxs)


def test_probe_indices_small_dataset_returns_all():
    from model.probes import _probe_indices
    records = [{"shifts": np.array([1.0, 7.5]), "degeneracy": np.array([1, 3])}
               for _ in range(4)]
    idxs = _probe_indices(records, 20)
    assert len(idxs) == 4


def test_probe_indices_covers_aromatic_and_aliphatic():
    """Probe set should include both aromatic (max_shift>median) and aliphatic molecules."""
    from model.probes import _probe_indices
    rng = np.random.default_rng(7)
    records = (
        [{"shifts": rng.uniform(0.5, 3.0, 8), "degeneracy": np.array([1]*8)} for _ in range(40)]
        + [{"shifts": rng.uniform(6.5, 9.0, 8), "degeneracy": np.array([1]*8)} for _ in range(40)]
    )
    idxs = _probe_indices(records, 8)
    max_shifts = [records[i]["shifts"].max() for i in idxs]
    assert any(s < 5.0 for s in max_shifts), "no aliphatic molecules selected"
    assert any(s > 5.0 for s in max_shifts), "no aromatic molecules selected"
