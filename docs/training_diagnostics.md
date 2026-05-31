# SpinHance Training Diagnostics

SpinHance model training now writes a durable, file-based diagnostics bundle for every run. This diagnostics bundle is the canonical interface between human debugging, the Streamlit live dashboard, probe/failure analysis, and AutoAI experiment selection.

The source of truth is the run directory, not stdout, screenshots, or prose summaries.

## Canonical run directory

Every training run should write to:

~~~text
model/runs/<run_id>/
├── config.json
├── status.json
├── metrics.jsonl
├── events.jsonl
├── summary.json
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── <optional user checkpoint>.pt
└── probes/
    └── epoch_XXXX/
        ├── probe_metrics.json
        ├── predictions.json
        ├── failure_summary.json
        ├── worst_cases.json
        ├── worst_shift_cases.json
        ├── worst_j_cases.json
        ├── worst_presence_cases.json
        ├── worst_deg_cases.json
        └── matrix_*.png
~~~

The `run_dir` is the primary artifact. Checkpoints, scalar metrics, probe outputs, and failure analysis all live underneath it.

## Smoke run

~~~bash
PYTHONPATH=. python -m model.run_experiment \
  --small \
  --epochs 2 \
  --batch 16 \
  --max-mol 128 \
  --run-dir model/runs/diagnostics_smoke \
  --ckpt model/runs/diagnostics_smoke/checkpoints/spinhance.pt \
  --log-every-steps 1
~~~

For longer runs, `--log-every-steps 25` is usually enough.

## Live dashboard

Launch:

~~~bash
PYTHONPATH=. streamlit run model/live_dashboard.py
~~~

Point the dashboard at a run directory, for example:

~~~text
model/runs/diagnostics_smoke
~~~

The dashboard reads artifacts from disk. It does not require access to the active training process.

Expected dashboard panels:

- run status, epoch, stage, device, and step;
- validation curves;
- training-step loss and learning-rate curves;
- curriculum weights;
- best metrics;
- probe diagnostics;
- worst probe cases;
- matrix plots;
- failure-analysis summary.

## Artifact meanings

### `config.json`

Static serialized training configuration. Use this to determine dataset, model options, loss settings, checkpoint path, diagnostics settings, and probe/failure-analysis settings.

### `status.json`

Atomic snapshot of the current run state. This file is repeatedly overwritten during training and is the best file for live monitoring.

Typical fields:

~~~json
{
  "state": "running",
  "epoch": 12,
  "epochs": 80,
  "stage": 1,
  "step": 1234,
  "best_score": 0.42,
  "device": "cuda",
  "checkpoint_best": "model/runs/<run_id>/checkpoints/best.pt",
  "checkpoint_last": "model/runs/<run_id>/checkpoints/last.pt"
}
~~~

### `metrics.jsonl`

Append-only stream of scalar metrics. Each line is one JSON object.

Important splits include:

- `train_step`
- `train`
- `val`
- `probe`

Example row:

~~~json
{
  "kind": "metrics",
  "split": "val",
  "epoch": 3,
  "step": 320,
  "metrics": {
    "shift_mae_ppm": 0.12,
    "j_mae_hz": 1.8,
    "presence_f1": 0.71,
    "deg_acc": 0.84,
    "h_shift_mae_ppm": 0.10,
    "h_j_mae_hz": 1.6
  }
}
~~~

AutoAI should read this file directly rather than scraping stdout.

### `events.jsonl`

Append-only stream of lifecycle events, warnings, and future infrastructure events.

### `summary.json`

Final run summary. This is the first place to look for best metrics and final checkpoint paths after a run finishes.

### `checkpoints/`

Canonical checkpoint files:

~~~text
checkpoints/best.pt
checkpoints/last.pt
~~~

If `--ckpt` points to another filename, that checkpoint is also written. Checkpoint parent directories are created automatically.

### `probes/epoch_XXXX/`

Probe artifacts are fixed-example diagnostics emitted during training.

Important files:

- `probe_metrics.json`
- `predictions.json`
- `failure_summary.json`
- `worst_cases.json`
- `worst_shift_cases.json`
- `worst_j_cases.json`
- `worst_presence_cases.json`
- `worst_deg_cases.json`
- `matrix_*.png`

The latest probe epoch is usually the most useful for AutoAI.

## Reusing run directories

The diagnostics writer resets live files when a run starts. This prevents a new run from appending rows onto stale `metrics.jsonl`.

Files reset at run start:

~~~text
metrics.jsonl
events.jsonl
system.jsonl
status.json
summary.json
probes/
~~~

Checkpoint files are not proactively deleted by the diagnostics reset, but checkpoint paths are overwritten when training saves.

For production AutoAI runs, prefer unique directories:

~~~text
model/runs/<timestamp>_<experiment_name>_<seed>
~~~

## Failure-analysis categories

Failure summaries currently use categories such as:

~~~text
large_shift_error
wrong_degeneracy
false_positive_couplings
false_negative_couplings
bad_j_magnitude
~~~

AutoAI should use these categories to guide the next experiment.

| Dominant failure | Suggested next direction |
|---|---|
| `large_shift_error` | increase shift loss; try Hungarian loss; improve spectral localization |
| `wrong_degeneracy` | add integration metadata; rebalance degeneracy classes |
| `false_negative_couplings` | increase coupling-presence loss; improve edge decoder |
| `false_positive_couplings` | tune coupling threshold; add sparsity prior |
| `bad_j_magnitude` | adjust J regression loss; add peak-shape features |

## AutoAI contract

Workers that launch `model.run_experiment` should return a `WorkerResult` whose `artifact_paths` include:

~~~json
{
  "run_dir": "model/runs/<run_id>",
  "checkpoint": "model/runs/<run_id>/checkpoints/best.pt",
  "metrics": "autoai/runs/<cycle>/metrics.json",
  "summary": "autoai/runs/<cycle>/summary.md"
}
~~~

`run_dir` is the most important path. It lets the orchestrator read:

- `summary.json`
- `metrics.jsonl`
- `status.json`
- latest `probes/*/failure_summary.json`
- checkpoint paths

Worker prose can be useful, but AutoAI should trust these files first.

## Relevant tests

~~~bash
PYTHONPATH=. pytest -q \
  model/test_diagnostics.py \
  model/test_diagnostics_reset.py \
  model/test_checkpoint_saving.py \
  model/test_failure_analysis.py \
  model/test_train_infra.py \
  autoai/test_run_reader.py
~~~

After AutoAI diagnostics integration, also test that:

- `run_dir` can be inferred from `artifact_paths`;
- `summary.json` and `metrics.jsonl` are parsed;
- latest probe failure summaries are attached to AutoAI records;
- missing diagnostics fail softly with a clear reason.

## Design principle

The diagnostics system is file-first.

W&B, TensorBoard, Streamlit, or notebooks may mirror these artifacts, but the canonical source of truth is the run directory.

