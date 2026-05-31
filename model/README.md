# SpinHance `model/` — modular training package

Rebuilt from the original flat monolith (the pre-rebuild layout; see git history)
around explicit, typed contracts so architectures, losses, renderers, training,
and diagnostics can be developed, tested, and swapped independently.

The one rule that drives the layout:

```
data ── SpinBatch ──▶ architecture ── ModelOutput ──▶ loss ── LossOutput ──▶ trainer
                                                renderer ── RendererOutput ──┘
```

No layer reaches into another's internals — they communicate only through the
dataclasses in `model/schemas`.

## Package map

| dir | role |
|---|---|
| `schemas/` | typed contracts: `SpinBatch`, `ModelOutput`, `LossOutput`, `RendererOutput`, diagnostics payloads, shared constants |
| `registry.py` | generic name→component `Registry`; one instance per layer |
| `data/` | records adapter, splits, standardization, transforms, dataset, collate → `SpinBatch` |
| `architectures/` | spectrum → `ModelOutput` models (registered) |
| `heads/` | typed output heads (shifts / couplings / presence / degeneracy) |
| `losses/` | `ModelOutput`+`SpinBatch` → `LossOutput` (matrix, hungarian, surrogate/exact spectral, region, composite) |
| `renderers/` | spin params → spectrum/summary: `exact_no_grad`, `exact_autograd_experimental`, `surrogate`, `region` |
| `training/` | config, trainer, loops, schedules, optimizer, checkpointing, seed, runner |
| `evaluation/` | metrics, hungarian matching, spectral metrics, probes, failure analysis |
| `diagnostics/` | run-dir writer, run reader, plots, live dashboard |
| `experiments/` | CLI entrypoints (`train`, `evaluate`, `profile_*`) |
| `configs/` | YAML run configs |
| `tests/` | unit + smoke tests |

## Training stages (see the master plan)

- **Stage 0** smoke/debug — seconds.
- **Stage 1** supervised matrix (or Hungarian) training — the stable baseline.
- **Stage 2A** exact **no-grad** spectral *evaluation* (metric only).
- **Stage 2B** surrogate spectral *training* (cheap, bounded memory).
- **Stage 2C** region-level spectral training.
- **Stage 2D** exact tiny-case autograd — experimental, disabled by default.

The exact differentiable quantum renderer is **never** the default Stage 2 loss.
It lives as a no-grad evaluator, a surrogate teacher, a probe diagnostic, and a
post-hoc refinement backend.

## Run / test

```bash
# unit tests
PYTHONPATH=. python -m pytest model/tests -q

# train from a config (Branch 2+)
PYTHONPATH=. python -m model.experiments.train --config model/configs/baseline_matrix.yaml
PYTHONPATH=. python -m model.experiments.train --config model/configs/baseline_matrix.yaml --set training.epochs=2 --set run.name=smoke
```

Every run writes the canonical artifact directory consumed by AutoAI and the dashboard:

```
model/runs/<run_id>/
├── config.json  status.json  metrics.jsonl  events.jsonl  summary.json
├── checkpoints/
└── probes/
```
