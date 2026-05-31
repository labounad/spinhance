# AutoAI — Autonomous SpinHance Model Search

`autoai/` is the autonomous experiment loop for SpinHance. Its job is to propose, implement, run, evaluate, and summarize model-training experiments.

The loop should optimize against durable artifacts, not console logs or prose-only summaries.

## Current architecture

Each cycle should:

1. read `autoai/IDEAS.md`;
2. read recent experiment summaries;
3. inspect the repo structure;
4. create or modify training code;
5. run a training experiment;
6. write AutoAI cycle artifacts;
7. read the canonical model-training diagnostics directory;
8. summarize what worked, what failed, and what should be tried next.

## Canonical training diagnostics

Model training writes a canonical run directory under `model/runs/<run_id>/`.

See `docs/training_diagnostics.md` for the full artifact contract.

## Required WorkerResult artifact paths

When a worker launches `model.run_experiment`, it should return artifact paths like:

~~~json
{
  "run_dir": "model/runs/<run_id>",
  "checkpoint": "model/runs/<run_id>/checkpoints/best.pt",
  "metrics": "autoai/runs/<cycle>/metrics.json",
  "summary": "autoai/runs/<cycle>/summary.md"
}
~~~

The `run_dir` key is the most important path. It lets the orchestrator read `summary.json`, `metrics.jsonl`, `status.json`, latest `probes/*/failure_summary.json`, and checkpoint files.

## AutoAI cycle artifacts

Each AutoAI cycle directory should contain:

~~~text
autoai/runs/<cycle>/
├── training.py
├── metrics.json
├── summary.md
├── worker_result.json
└── diagnostics_analysis.json
~~~

`metrics.json` is the worker's compact metric report. `diagnostics_analysis.json` is the orchestrator's parsed view of the canonical `model/runs/<run_id>` training run.

## Metrics to prefer

For model selection, prefer metrics parsed from the training diagnostics bundle:

~~~text
shift_mae_ppm
h_shift_mae_ppm
j_mae_hz
h_j_mae_hz
presence_f1
deg_acc
deg_acc_balanced
matrix_loss
~~~

The `h_*` metrics are Hungarian-matched metrics and are important because spin-group labels are permutation-arbitrary.

## Failure-driven iteration

AutoAI should inspect `failure_summary.json` and use the dominant failure to choose the next experiment.

| Dominant failure | Good next experiment |
|---|---|
| `large_shift_error` | increase shift loss; try Hungarian loss; improve spectral localization |
| `wrong_degeneracy` | add integration metadata; rebalance degeneracy classes |
| `false_negative_couplings` | increase coupling-presence loss; improve edge decoder |
| `false_positive_couplings` | tune coupling threshold; add sparsity prior |
| `bad_j_magnitude` | adjust J regression loss; add peak-shape features |

## Local smoke run

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

Dashboard:

~~~bash
PYTHONPATH=. streamlit run model/live_dashboard.py
~~~

## Running AutoAI

Make sure AWS credentials are active if using EC2:

~~~bash
./context/setup_aws_login.sh
~~~

Start the loop:

~~~bash
PYTHONPATH=. python autoai/orchestrator.py
~~~
