from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from autoai.run_reader import analyze_artifact_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_tiny_training(model_run_dir: Path, *, clean: bool = True) -> None:
    if clean and model_run_dir.exists():
        shutil.rmtree(model_run_dir)

    ckpt = model_run_dir / "checkpoints" / "spinhance.pt"

    cmd = [
        sys.executable,
        "-m",
        "model.run_experiment",
        "--small",
        "--epochs",
        "2",
        "--batch",
        "16",
        "--max-mol",
        "128",
        "--run-dir",
        _rel(model_run_dir),
        "--ckpt",
        _rel(ckpt),
        "--log-every-steps",
        "1",
    ]

    print("Running training smoke command:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def write_cycle_artifacts(
    *,
    model_run_dir: Path,
    cycle_dir: Path,
) -> dict[str, Any]:
    checkpoint = model_run_dir / "checkpoints" / "best.pt"
    artifact_paths = {
        "run_dir": _rel(model_run_dir),
        "checkpoint": _rel(checkpoint),
        "metrics": _rel(cycle_dir / "metrics.json"),
        "summary": _rel(cycle_dir / "summary.md"),
    }

    analysis = analyze_artifact_paths(artifact_paths, repo_root=REPO_ROOT)
    if not analysis.get("available"):
        raise RuntimeError(f"Diagnostics analysis unavailable: {analysis}")

    cycle_dir.mkdir(parents=True, exist_ok=True)

    _write_json(cycle_dir / "diagnostics_analysis.json", analysis)

    compact = analysis.get("compact_best_metrics") or {}
    metrics = {
        "status": "success",
        "diagnostics_available": True,
        "dominant_failure": analysis.get("dominant_failure"),
        "run_dir": artifact_paths["run_dir"],
        "checkpoint": artifact_paths["checkpoint"],
        "compact_best_metrics": compact,
        "diagnostics": analysis,
    }
    _write_json(cycle_dir / "metrics.json", metrics)

    worker_result = {
        "status": "success",
        "artifact_paths": {
            **artifact_paths,
            "diagnostics_analysis": _rel(cycle_dir / "diagnostics_analysis.json"),
        },
        "metrics": metrics,
        "errors": "",
    }
    _write_json(cycle_dir / "worker_result.json", worker_result)

    summary_lines = [
        "# Local AutoAI diagnostics smoke cycle",
        "",
        f"- Model run directory: `{artifact_paths['run_dir']}`",
        f"- Checkpoint: `{artifact_paths['checkpoint']}`",
        f"- Diagnostics available: `{analysis.get('available')}`",
        f"- Run id: `{analysis.get('run_id')}`",
        f"- State: `{analysis.get('state')}`",
        f"- Best epoch: `{analysis.get('best_epoch')}`",
        f"- Best score: `{analysis.get('best_score')}`",
        f"- Dominant failure: `{analysis.get('dominant_failure')}`",
        "",
        "## Compact best metrics",
        "",
    ]

    if compact:
        for key, value in compact.items():
            summary_lines.append(f"- `{key}`: `{value}`")
    else:
        summary_lines.append("- No compact metrics found.")

    summary_lines.extend(
        [
            "",
            "## Purpose",
            "",
            "This smoke cycle verifies that AutoAI can consume the canonical `model/runs/<run_id>` diagnostics bundle and write an AutoAI-style parsed diagnostics artifact.",
            "",
        ]
    )

    (cycle_dir / "summary.md").write_text("\n".join(summary_lines))

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local AutoAI diagnostics smoke cycle.")
    parser.add_argument(
        "--run-id",
        default="autoai_diagnostics_smoke",
        help="model/runs/<run-id> name for the tiny training run",
    )
    parser.add_argument(
        "--cycle-id",
        default="",
        help="autoai/runs/<cycle-id> name; default uses timestamp",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="reuse an existing model run directory instead of launching training",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="do not remove the model run directory before training",
    )
    args = parser.parse_args()

    model_run_dir = REPO_ROOT / "model" / "runs" / args.run_id
    cycle_id = args.cycle_id or f"smoke_diagnostics_{time.strftime('%Y%m%d_%H%M%S')}"
    cycle_dir = REPO_ROOT / "autoai" / "runs" / cycle_id

    if not args.skip_training:
        run_tiny_training(model_run_dir, clean=not args.no_clean)

    analysis = write_cycle_artifacts(
        model_run_dir=model_run_dir,
        cycle_dir=cycle_dir,
    )

    print()
    print("Smoke diagnostics cycle complete.")
    print(f"Model run:  {_rel(model_run_dir)}")
    print(f"AutoAI run: {_rel(cycle_dir)}")
    print(f"Dominant failure: {analysis.get('dominant_failure')}")
    print(f"Diagnostics analysis: {_rel(cycle_dir / 'diagnostics_analysis.json')}")


if __name__ == "__main__":
    main()
