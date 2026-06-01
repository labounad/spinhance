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

## Production model — `spingraph_decoder`

ResNet1D conv stem → ppm-positioned global tokens → pre-LN Transformer encoder →
8 learned spin-group queries → Transformer decoder → per-node heads (shift +
degeneracy) + symmetric `PairwiseEdgeHead`. Sizes via `model.size`:
`medium` ≈ 10M (production), **`xl` ≈ 57M** (`dim512/enc4/dec6`, for the 3M+ regime).
See `RESULTS.md` for the ablation; production recipe (025) is matrix loss with
`shift` weighted 2× + WSD LR (`train_64k_spingraph_shift2x_matrixonly.yaml`).

Two optional, default-off inductive biases (added in session026):
- **`model.use_peak_channel`** — feeds a 2nd conv input channel: an in-model
  peak-emphasis map (local maxima above a per-sample threshold, Gaussian-smoothed)
  computed from the spectrum in `forward`. A shift-localization prior with no
  data-pipeline change. `ResNet1DEncoder` gains `in_channels`.
- **`soft_equiv` loss** — `PairwiseEdgeHead` emits a 3rd per-edge logit
  (`ModelOutput.auxiliary["soft_equiv_logits"]`) flagging *soft-equivalent* groups
  (same shift, different couplings — accidental degeneracy). `SoftEquivLoss`
  supervises it (BCE vs `|δᵢ−δⱼ|≤tol`) + pulls those predicted shifts together;
  `evaluation.metrics.decode` averages flagged groups so a degenerate pair renders
  as one peak, not a split doublet.

## Data paths

- **Per-file** (default, ChEMBL 64k): `data.records` JSON + `data.spectra` dir of
  `<mol_id>.npy` (`load_records`).
- **Stacked shards** (PubChem 3M+): set `data.parts` to a dir of `part_NNNNN.npy`
  (1000 spectra each, 90 MHz only) keyed by record order to a `.json[.gz]`
  `data.records`. `StackedSpectra` mmaps shards lazily (headers-only index);
  `load_pubchem_records` streams the gz. Configs `train_3M_spingraph_xl_{025,026}.yaml`.
  ⚠️ Use `num_workers≤2` at ≥500k records on a ≤16 GB box (per-worker record-list
  copies OOM; COW is broken by Python refcounting); full 3.2M needs ≥32 GB RAM.

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
