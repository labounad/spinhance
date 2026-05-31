import json
from pathlib import Path

from autoai.run_reader import (
    analyze_artifact_paths,
    infer_run_dir_from_artifacts,
    latest_probe_epoch,
    summarize_metrics_for_agent,
)


def _write_model_run(root: Path) -> Path:
    run_dir = root / "model" / "runs" / "run_a"
    run_dir.mkdir(parents=True)

    (run_dir / "status.json").write_text(json.dumps({
        "state": "finished",
        "best_epoch": 1,
        "best_score": 1.23,
    }))

    (run_dir / "summary.json").write_text(json.dumps({
        "best_metrics": {
            "shift_mae_ppm": 0.12,
            "h_shift_mae_ppm": 0.11,
            "j_mae_hz": 1.8,
            "presence_f1": 0.7,
            "deg_acc": 0.9,
            "extra_metric": 999,
        }
    }))

    (run_dir / "metrics.jsonl").write_text(
        json.dumps({
            "kind": "metrics",
            "split": "val",
            "epoch": 1,
            "step": 10,
            "metrics": {
                "shift_mae_ppm": 0.12,
                "j_mae_hz": 1.8,
            },
        }) + "\n"
    )

    probe_dir = run_dir / "probes" / "epoch_0001"
    probe_dir.mkdir(parents=True)
    (probe_dir / "failure_summary.json").write_text(json.dumps({
        "dominant_failure": "wrong_degeneracy",
        "n_molecules": 16,
    }))

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "best.pt").write_text("placeholder\n")

    return run_dir


def test_infer_run_dir_from_explicit_artifact_path(tmp_path):
    run_dir = _write_model_run(tmp_path)
    got = infer_run_dir_from_artifacts(
        {"run_dir": "model/runs/run_a"},
        repo_root=tmp_path,
    )
    assert got == run_dir


def test_infer_run_dir_from_checkpoint_path(tmp_path):
    run_dir = _write_model_run(tmp_path)
    got = infer_run_dir_from_artifacts(
        {"checkpoint": "model/runs/run_a/checkpoints/best.pt"},
        repo_root=tmp_path,
    )
    assert got == run_dir


def test_latest_probe_epoch(tmp_path):
    run_dir = _write_model_run(tmp_path)
    assert latest_probe_epoch(run_dir) == "epoch_0001"


def test_summarize_metrics_for_agent_filters_to_key_metrics():
    compact = summarize_metrics_for_agent({
        "shift_mae_ppm": 0.12,
        "j_mae_hz": 1.8,
        "presence_f1": 0.7,
        "extra_metric": 999,
    })
    assert compact == {
        "shift_mae_ppm": 0.12,
        "j_mae_hz": 1.8,
        "presence_f1": 0.7,
    }


def test_analyze_artifact_paths_reads_canonical_diagnostics(tmp_path):
    _write_model_run(tmp_path)
    analysis = analyze_artifact_paths(
        {
            "run_dir": "model/runs/run_a",
            "checkpoint": "model/runs/run_a/checkpoints/best.pt",
        },
        repo_root=tmp_path,
    )

    assert analysis["available"] is True
    assert analysis["run_id"] == "run_a"
    assert analysis["latest_probe_epoch"] == "epoch_0001"
    assert analysis["dominant_failure"] == "wrong_degeneracy"
    assert analysis["compact_best_metrics"]["shift_mae_ppm"] == 0.12


def test_analyze_artifact_paths_reports_missing_diagnostics(tmp_path):
    analysis = analyze_artifact_paths(
        {"metrics": "autoai/runs/run_001/metrics.json"},
        repo_root=tmp_path,
    )
    assert analysis["available"] is False
    assert "reason" in analysis
