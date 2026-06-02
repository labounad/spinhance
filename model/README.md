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
| `losses/` | `ModelOutput`+`SpinBatch` → `LossOutput` (`matrix`, `hungarian`, `surrogate_spectral`, `soft_equiv`, `composite`) |
| `renderers/` | spin params → spectrum/summary: `exact_no_grad`, `exact_autograd_experimental`, `surrogate` |
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
See `RESULTS.md` for the ablation; the production model is **light-025**
(`train_500k_light_025.yaml`) — the 025 recipe (matrix loss, `shift` weighted 2× + WSD LR)
trained on 500k PubChem (0.036 ppm / 0.51 Hz, best ep43), superseding the 64k session-025.

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

## Component registry

The three swappable layers are name-registered; build by string key
(`build_architecture`, `build_loss`, `build_renderer`).

**Architectures** (`model/architectures/`):

| key | role |
|---|---|
| `resnet1d` | dense-CNN baseline: ResNet1D stem → global avg-pool → `TypedMatrixHead`. The floor. |
| `resnet1d_attention_pool` | ResNet1D + multi-head attention pooling (IDEAS Family B). Works, not primary. |
| `spingraph_decoder` | **production** — ResNet1D stem → ppm-positioned tokens → Transformer encoder → 8 spin-group queries → Transformer decoder → `NodeHead` + `PairwiseEdgeHead`. |

**Losses** (`model/losses/`):

| key | what it computes |
|---|---|
| `matrix` | canonical supervised anchor: shift Huber + presence-masked J Huber + presence BCE + degeneracy CE (standardized space). Per-component `weights` (default `shift 1·jmag 1·presence .5·deg .5`; production overrides `shift: 2.0`). |
| `hungarian` | permutation-invariant set-matched variant. **Hurts on this distinct-shift data — RESULTS §2; not used.** |
| `surrogate_spectral` | renders the predicted matrix through the frozen `surrogate` renderer → `w1_weight·W1 + cosine_weight·(1−cos)` vs the (clean) target spectrum. Gradients flow through the frozen teacher; ramp in via `composite`. |
| `soft_equiv` | edge-flag BCE (`\|δᵢ−δⱼ\|≤tol_ppm`) + shift-consistency penalty for accidental-degeneracy pairs. No-op unless the arch emits `auxiliary["soft_equiv_logits"]`. |
| `composite` | config-driven weighted sum of the above with per-term curriculum (`init_weight`→`weight` over `start_epoch`/`ramp_epochs`, optional `decay_start_epoch`/`decay_epochs`→`end_weight`). The trainer drives it via `set_epoch`. |

**Renderers** (`model/renderers/`):

| key | role |
|---|---|
| `exact_no_grad` | exact quantum simulator, no gradients — Stage-2A evaluation metric. |
| `exact_autograd_experimental` | exact + autograd, tiny systems only — disabled by default. |
| `surrogate` | learned differentiable teacher; frozen and used as the `surrogate_spectral` backend. |

**Heads** (`model/heads/`):

| class | used by | outputs |
|---|---|---|
| `TypedMatrixHead` | `resnet1d`, `resnet1d_attention_pool` | shift `(B,G)`, J mag `(B,E)`, J presence `(B,E)`, degeneracy `(B,G,C)` |
| `NodeHead` | `spingraph_decoder` | per-node shift `(B,G)` + degeneracy logits `(B,G,C)` |
| `PairwiseEdgeHead` | `spingraph_decoder` | symmetric per-edge J mag + presence (+ soft-equiv logit when `use_soft_equiv`); `edge_ij = MLP([h_i+h_j, \|h_i−h_j\|])` |

**Size presets** (`model.size`; `SIZE_PRESETS` in `resnet1d.py`, conv stem shared by all archs):

| size | (stem, stage widths, blocks) | spingraph params | use |
|---|---|---|---|
| `tiny` / `small` | (16/24, …, 1-block) | <2M | smoke / debug |
| `medium` | (32, (64,128,256,512), (2,2,2,2)) | **~10M** | **production** (64k–500k) |
| `large` | (48, (96,192,384,768), (2,2,3,2)) | ~20M | intermediate |
| `xl` | (64, (128,256,512,768), (2,3,4,3)) + `dim512/enc4/dec6` | **~57M** | 3M+ PubChem |

**Config index** (`model/configs/`; full ablation history in `RESULTS.md`):

| config(s) | arch · loss · data |
|---|---|
| `baseline_matrix.yaml`, `train_64k.yaml` | resnet1d · matrix · 64k ChEMBL (CNN floor) |
| `hungarian_matrix.yaml` | resnet1d · hungarian · 64k (deprecated approach) |
| `surrogate.yaml`, `surrogate_large.yaml` | train the `surrogate` renderer (Stage-2 teacher) |
| `train_64k_surrogate_spectral*.yaml` (5) | resnet1d · composite(matrix+spectral) · 64k — sessions 015–020 ablations |
| `train_64k_spingraph_canonical.yaml` | spingraph · composite(matrix+spectral) · 64k — session 022 |
| `train_64k_spingraph_shift2x_matrixonly.yaml` | spingraph · matrix(shift 2×) · 64k — session 025 (superseded by light-025) |
| `train_64k_spingraph_shift2x_spectral.yaml` | session 025 + spectral variant |
| `train_64k_spingraph_regions.yaml` | spingraph + region tokens · 64k — session 023 (abandoned, slower/no gain) |
| `train_64k_026_peaks_softequiv.yaml` | spingraph(peak+soft-equiv) · matrix+soft_equiv · 64k — session 026 |
| `train_500k_light_025.yaml` | spingraph medium · 025 recipe · 500k PubChem — **production** (0.036 ppm, ep43) |
| `train_500k_light_026.yaml` | spingraph medium · 026 recipe · 500k PubChem |
| `train_3M_spingraph_xl_{025,026}.yaml` | spingraph **xl** · 025/026 recipe · 3.2M PubChem |

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
