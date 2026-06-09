# Spinhance `model/` — modular training package

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
degeneracy) + symmetric `PairwiseEdgeHead`. Sizes via `model.size` — the three
production **tiers** are `light` (64k, ~10M), `med` (500k, ~57M), and `xl`
(3M, ~137M); see the tier table below (the single source of truth).
See `RESULTS.md` for the full **025–030 recipe sweep**. The entire fleet is currently
being **retrained on the corrected v2 data** (Audit-2; see `RESULTS.md`) across all three
tiers; the prior (pre-correction) metrics are **superseded** and new numbers are **pending**.
Held-out scoring uses a **leakage-controlled** global 10% held-out test split of all 3,126,829
molecules: near-duplicates are grouped by union-find (matrix fingerprint + InChIKey), the clusters
are placed by a single random shuffle (`default_rng` seed 0), and the last ~10% of molecules (whole
clusters) are held out (~2.81M train / ~0.31M test) — so no near-duplicate spans train/test.
The model also supports a decode-time **test-time refinement** step
(`model/inference/refine.py`) that polishes predicted shifts against the input 90 MHz spectrum
(+43–77% shift-MAE — see `RESULTS.md` and `DESIGN.md` §12).

Two optional, default-off inductive biases (added in the 026 recipe):
- **`model.use_peak_channel`** — feeds a 2nd conv input channel: an in-model
  peak-emphasis map (local maxima above a per-sample threshold, Gaussian-smoothed)
  computed from the spectrum in `forward`. A shift-localization prior with no
  data-pipeline change. `ResNet1DEncoder` gains `in_channels`.
- **`soft_equiv` loss** — `PairwiseEdgeHead` emits a 3rd per-edge logit
  (`ModelOutput.auxiliary["soft_equiv_logits"]`) flagging *soft-equivalent* groups
  (chemically equivalent — same canonical symmetry orbit — but kept as distinct
  nodes, e.g. AA'BB': identical shift, different coupling rows). `SoftEquivLoss`
  supervises it (BCE vs the **symmetry-orbit** label `batch.soft_equiv_target`,
  **not** shift proximity) + pulls those predicted shifts together;
  `evaluation.metrics.decode` averages groups whose *predicted* flag fires so a degenerate pair renders
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
| `soft_equiv` | edge-flag BCE vs the canonical **symmetry-orbit** label (`batch.soft_equiv_target`; chemically-equivalent groups, *not* shift proximity) + a shift-consistency penalty pulling those pairs together. No-op unless the arch emits `auxiliary["soft_equiv_logits"]`. |
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

**Model tiers** (`model.size`; `TIER_PRESETS` in `resnet1d.py` — **the single source of
truth** for the data-scaling fleet). A tier fully defines the model size (conv stem PLUS
transformer width/depth); fleet configs set only `model.size` and nothing else size-related:

| tier | data tier | conv stem · transformer | spingraph params | configs |
|---|---|---|---|---|
| `light` | 64k  | `medium` stem · `dim256/enc2/dec4` | **~10M**  | `train_64k_*` |
| `med`   | 500k | `deep` stem · `dim512/enc4/dec6`   | **~57M**  | `train_500k_*` |
| `xl`    | 3M   | `deep` stem · `dim768/enc6/dec8`   | **~137M** | `train_3M_*` |

The raw conv-stem presets (`SIZE_PRESETS` in `resnet1d.py`: `tiny`/`small`/`medium`/`large`/`deep`,
shared by all archs) remain usable directly for experiments/back-compat, but a **tier name takes
precedence** and is the convention for all production runs. Enforced by `model/tests/test_model_tiers.py`.

**Config index** (`model/configs/`; full ablation history in `RESULTS.md`):

| config(s) | arch · loss · data |
|---|---|
| `baseline_matrix.yaml`, `train_64k.yaml` | resnet1d · matrix · 64k (the **old CNN reference floor** every architecture must beat) |
| `hungarian_matrix.yaml` | resnet1d · hungarian · 64k (deprecated approach) |
| `surrogate.yaml`, `surrogate_large.yaml` | train the `surrogate` renderer (Stage-2 teacher) |
| `train_64k_surrogate_spectral*.yaml` (5) | resnet1d · composite(matrix+spectral) · 64k — sessions 015–020 ablations |
| `train_64k_spingraph_canonical.yaml` | spingraph · composite(matrix+spectral) · 64k — session 022 |
| `train_64k_spingraph_shift2x_matrixonly.yaml` | spingraph · matrix(shift 2×) · 64k — session 025 (superseded by the 025 fleet) |
| `train_64k_spingraph_shift2x_spectral.yaml` | session 025 + spectral variant |
| `train_64k_spingraph_regions.yaml` | spingraph + region tokens · 64k — session 023 (abandoned, slower/no gain) |
| `train_64k_026_peaks_softequiv.yaml` | spingraph(peak+soft-equiv) · matrix+soft_equiv · 64k — recipe 026 |
| 64k recipes `025`–`030` (`train_64k_*`) | spingraph **light** (~10M) · the 025–030 ladder · 64k PubChem (the ablation tier) |
| 500k recipes `025`–`030` (`train_500k_*`) | spingraph **med** (~57M) · the 025–030 ladder · 500k PubChem |
| 3M recipes (`025`, `027`) (`train_3M_*`) | spingraph **xl** (~137M) · 3M PubChem |

> The 025–030 ladder is being **retrained on the corrected v2 data** (64k 025–030,
> 500k 025–030, 3M 025+027); per-tier peak LR + WSD short plateau + grad-spike guard.
> All numbers are **pending** — see `RESULTS.md`.

## Data paths

- **Per-file** (default, 64k ablation tier): `data.records` JSON + `data.spectra` dir of
  `<mol_id>.npy` (`load_records`).
- **Stacked shards** (PubChem 500k–3M): set `data.parts` to a dir of `part_NNNNN.npy`
  (1000 spectra each, 90 MHz only, 16384-point) keyed by record order to a `.json[.gz]`
  `data.records`. The shard reader mmaps shards lazily (headers-only index);
  `load_pubchem_records` streams the gz; `data.sample_n` reservoir-samples the tier.
  ⚠️ Use `num_workers≤2` at ≥500k records on a ≤16 GB box (per-worker record-list
  copies OOM; COW is broken by Python refcounting); full 3M needs ≥32 GB RAM.

On the **Garibaldi HPC**, the corrected **v2** dataset lives in the group filesystem at
`/gpfs/group/shenvi/Users/labounader/spinhance/consolidated_v2/` —
`records_train_shuf.json.gz`, `records_test.json.gz`, `parts/` (stacked 90 MHz spectra),
and `preload_train_full.npy`. The regenerated per-shard source is under `rebuild3M_v2/`.
`runs/` holds checkpoints (`model/runs` is a **symlink** into here). The repo checkout and
conda env stay in `$HOME`.

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

Every run writes the canonical artifact directory consumed by the live dashboard + monitoring tools:

```
model/runs/<run_id>/
├── config.json  status.json  metrics.jsonl  events.jsonl  summary.json
├── checkpoints/
└── probes/
```
