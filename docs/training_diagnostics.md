# SpinHance Training Diagnostics

SpinHance model training writes a durable, S3-backed diagnostics bundle for every run. This bundle is the canonical interface between human debugging, the Streamlit live dashboard, probe/failure analysis, and AutoAI experiment selection.

The source of truth is the S3 session, not stdout, screenshots, or prose summaries.

## Canonical S3 layout

Every training run writes to a session directory inside S3:

~~~text
s3://spinhance-data/training/sessionXXX/
├── config.json
├── status.json
├── metrics.jsonl
├── events.jsonl
├── summary.json
├── checkpoints/
│   ├── best.pt
│   ├── last.pt
│   └── <optional local backup>.pt
└── probes/
    └── epoch_XXXX/
        ├── checkpoint.pt
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

`sessionXXX` is a short identifier (e.g. `session001`) supplied via
`--session-id` or auto-generated as a timestamped local path when running
without S3.

The per-epoch checkpoint is stored at:

~~~text
s3://spinhance-data/training/sessionXXX/probes/epoch_XXXX/checkpoint.pt
~~~

Best and last checkpoints live at:

~~~text
s3://spinhance-data/training/sessionXXX/checkpoints/best.pt
s3://spinhance-data/training/sessionXXX/checkpoints/last.pt
~~~

## Training launch

### S3 session (cloud / EC2)

~~~bash
PYTHONPATH=. python -m model.run_experiment \
  --session-id session001 \
  --small \
  --epochs 60 \
  --batch 64
~~~

Or with a full URI:

~~~bash
PYTHONPATH=. python -m model.run_experiment \
  --session-id s3://spinhance-data/training/session001 \
  --small --epochs 60 --batch 64
~~~

### Auto-generated session

Omit `--session-id` to auto-generate a session named `session_<timestamp>`:

~~~bash
PYTHONPATH=. python -m model.run_experiment \
  --small \
  --epochs 2 \
  --batch 16 \
  --max-mol 128 \
  --log-every-steps 1
~~~

Artifacts go to `s3://spinhance-data/training/session_<timestamp>/`.

For longer runs, `--log-every-steps 25` is usually enough.

## Live dashboard

~~~bash
PYTHONPATH=. streamlit run model/live_dashboard.py
~~~

The sidebar lists sessions from `s3://spinhance-data/training/`. Select one to
monitor it. The dashboard reads all artifacts directly from S3 and
auto-refreshes every 5 s when live mode is on.

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

## Session analysis GUI

~~~bash
conda run -n spinhance streamlit run model/gui.py
~~~

The GUI lists sessions at `s3://spinhance-data/training/`, displays a
per-epoch validation-score bar chart, and lets you browse the test set with
ground-truth vs. predicted matrices and spectra.  Epoch checkpoints are
downloaded from `probes/epoch_XXXX/checkpoint.pt` within the session.

## Artifact meanings

### `config.json`

Static serialized training configuration written at run start.

### `status.json`

Atomic snapshot of the current run state (repeatedly overwritten during
training).  Checkpoint paths are absolute S3 URIs:

~~~json
{
  "state": "running",
  "epoch": 12,
  "epochs": 80,
  "stage": 1,
  "best_score": 0.42,
  "device": "cuda",
  "checkpoint_best": "s3://spinhance-data/training/sessionXXX/checkpoints/best.pt",
  "checkpoint_last": "s3://spinhance-data/training/sessionXXX/checkpoints/last.pt"
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

Final run summary. First place to look for best metrics and final checkpoint
paths after a run finishes.

### `checkpoints/`

Canonical checkpoint files:

~~~text
checkpoints/best.pt
checkpoints/last.pt
~~~

Checkpoint parent directories are created automatically.

### `probes/epoch_XXXX/`

Per-epoch probe artifacts. Written every `probe_every_epochs` epochs.

Important files:

- `checkpoint.pt` — model weights snapshot for this epoch
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

## Resetting a session

The diagnostics writer resets live files when a run starts, preventing a
new run from appending onto stale `metrics.jsonl` rows.  Files reset at
run start:

~~~text
metrics.jsonl
events.jsonl
system.jsonl
status.json
summary.json
probes/   (entire prefix deleted from S3)
~~~

Checkpoint files are **not** proactively deleted.

For production AutoAI runs, prefer unique session IDs:

~~~text
session_<timestamp>_<experiment_name>_<seed>
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

Workers that launch `model.run_experiment` should return a `WorkerResult`
whose `artifact_paths` include:

~~~json
{
  "run_dir": "s3://spinhance-data/training/sessionXXX",
  "checkpoint": "s3://spinhance-data/training/sessionXXX/checkpoints/best.pt"
}
~~~

`run_dir` is the most important path. It lets the orchestrator read:

- `summary.json`
- `metrics.jsonl`
- `status.json`
- latest `probes/*/failure_summary.json`
- checkpoint paths

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

All tests use local `tmp_path` fixtures and exercise the local-filesystem code
path.  No AWS credentials are required to run the test suite.

## Design principle

The diagnostics system is S3-first for cloud training, local-filesystem for
tests and smoke runs.

`DiagnosticsWriter`, `ProbeEvaluator`, and `save_failure_cases` all
auto-detect the backend from the `run_dir` prefix: an `s3://` prefix routes
to S3 via `model.s3io`; anything else uses the local filesystem.
