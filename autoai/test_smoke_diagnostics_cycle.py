import json
from pathlib import Path

import autoai.smoke_diagnostics_cycle as smoke


def _write_fake_model_run(root: Path) -> Path:
    run_dir = root / "model" / "runs" / "fake_run"
    run_dir.mkdir(parents=True)

    (run_dir / "status.json").write_text(json.dumps({
        "state": "finished",
        "best_epoch": 1,
        "best_score": 0.5,
    }))

    (run_dir / "summary.json").write_text(json.dumps({
        "state": "finished",
        "best_epoch": 1,
        "best_score": 0.5,
        "best_metrics": {
            "shift_mae_ppm": 0.1,
            "h_shift_mae_ppm": 0.09,
            "j_mae_hz": 1.5,
            "presence_f1": 0.8,
            "deg_acc": 0.75,
        },
    }))

    (run_dir / "metrics.jsonl").write_text(
        json.dumps({
            "kind": "metrics",
            "split": "val",
            "epoch": 1,
            "step": 10,
            "metrics": {
                "shift_mae_ppm": 0.1,
                "j_mae_hz": 1.5,
            },
        }) + "\n"
    )

    probe_dir = run_dir / "probes" / "epoch_0001"
    probe_dir.mkdir(parents=True)
    (probe_dir / "failure_summary.json").write_text(json.dumps({
        "dominant_failure": "large_shift_error",
        "recommendation": "Try Hungarian matching loss.",
    }))

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "best.pt").write_text("placeholder checkpoint")

    return run_dir


def test_write_cycle_artifacts_reads_model_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "REPO_ROOT", tmp_path)

    model_run_dir = _write_fake_model_run(tmp_path)
    cycle_dir = tmp_path / "autoai" / "runs" / "cycle_001"

    analysis = smoke.write_cycle_artifacts(
        model_run_dir=model_run_dir,
        cycle_dir=cycle_dir,
    )

    assert analysis["available"] is True
    assert analysis["run_id"] == "fake_run"
    assert analysis["dominant_failure"] == "large_shift_error"

    diagnostics_path = cycle_dir / "diagnostics_analysis.json"
    metrics_path = cycle_dir / "metrics.json"
    worker_path = cycle_dir / "worker_result.json"
    summary_path = cycle_dir / "summary.md"

    assert diagnostics_path.exists()
    assert metrics_path.exists()
    assert worker_path.exists()
    assert summary_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["diagnostics_available"] is True
    assert metrics["dominant_failure"] == "large_shift_error"
    assert metrics["compact_best_metrics"]["shift_mae_ppm"] == 0.1

    worker = json.loads(worker_path.read_text())
    assert worker["artifact_paths"]["run_dir"] == "model/runs/fake_run"
    assert worker["artifact_paths"]["diagnostics_analysis"] == "autoai/runs/cycle_001/diagnostics_analysis.json"
