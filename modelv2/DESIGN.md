# modelv2 — Design Document

## Problem statement

Predict the field-independent ¹H NMR spin system (shifts, scalar couplings,
degeneracies) from a single low-field (90 MHz) spectrum.

**Input:** one normalized ¹H spectrum, 16 384 intensity points over 0–12 ppm,
∫ = 1, simulated at 90 MHz (strongly coupled, non-first-order).

**Output:** the spin system as an 8-group matrix:

- Chemical shifts (ppm) — one per group
- Scalar couplings J (Hz) — one per group pair (upper triangle)
- Coupling presence — binary flag per pair (is there a coupling at all?)
- Proton degeneracy — one integer per group from the vocab {1, 2, 3, 4, 6, 9, 12, 18}

The target is defined **modulo S₈** (the 8 group labels are arbitrary).

## Why modelv2

`model/` accumulated too many files over too many commits. The code became
fragmented across 20+ modules, hard to debug, and ultimately stopped working.
`modelv2/` is a clean rewrite that makes no scientific compromises — the
physics is identical — but fits into four files that a person can hold in
their head at once.

## File structure

```text
modelv2/
├── DESIGN.md          this file
├── data.py            splits, target encoding, dataset, augmentation, standardization
├── model.py           neural network (encoder + heads)
├── train.py           losses, metrics, training loop, evaluation, CLI
└── gui.py             Streamlit app for evaluating training diagnostics and model output
```

Four files. No more, no less. Every piece of the pipeline lives in one of
them and only one. No circular imports, no indirection.

---

## data.py — Data pipeline

Everything that touches molecules before the model sees them, in one file.

**Inputs**

- There are two ground truth files available to the model:
    - `mol_to_spin_system/data/spin_systems_pubchem.json.tar.gz` — json file containing chemical identifiers, spin groups, and coupling constants for over 2 million molecules
    - `simulation/data/spectra/90MHz/mol_all.tar.gz` — physically realistic generated spectra for the entire molecule set
- These files need to be set as non-optional command line arguments

**Target encoding:**

- `canonical_order(shifts, couplings, degeneracy)` — lexsort by shift↓,
  degeneracy↓, |J| row-sum↓. Resolves S₈ arbitrariness into a deterministic
  ordering. Slight label noise for near-equal shifts; acceptable at this stage.
- `encode_target(rec, vocab)` — returns standardized shifts, J magnitudes
  (masked to present couplings), presence flags, degeneracy class indices.
- `Standardizer` — fits z-score parameters (shift mean/std, J mean/std over
  present pairs) on the training set and applies them. Lives here; passed to
  the model at inference.
- `DegeneracyVocab` — maps integer degeneracy values to class indices and back.

**Splits:**

- `make_splits(records, ratios=(0.7, 0.2, 0.1), seed=0)` — molecule-level
  scaffold split (Bemis-Murcko via RDKit, or SMILES-free matrix-dedup fallback)
  with near-duplicate detection and stratified assignment. Returns
  `{mol_id: fold}`.
- Union-find groups molecules that share a scaffold or a near-identical matrix
  (shift tolerance 0.02 ppm, J tolerance 0.5 Hz); whole groups go to one fold.

**Spectra cache:**

- `SpectraCache(records, archive=".../mol_all.tar.gz")` — at construction,
  streams the archive once into a single fp16 array held in RAM and indexes it
  by `mol_id`. Defined at module level so it is picklable. The full set is a few
  GB against 512 GiB of host RAM, so this is the default path; `np.load` is never
  called per item.

**Dataset:**

- `SpinDataset(records, vocab, std, cache, augment, ...)` — one item = one
  molecule; reads its spectrum as a slice of the `SpectraCache` (falls back to
  `rec["spec90_path"]` only if no cache is given). Returns the raw spectrum —
  augmentation is applied later, on-GPU, to the whole batch (see Runtime &
  hardware).
- `collate(batch)` — stacks tensors into a batch.

**Augmentation** (`augment_spectrum`, applied on-GPU to the batch):

- Small global referencing shift (sub-pixel, interpolated)
- Gaussian noise (fraction of peak height)
- Low-frequency sinusoidal baseline drift
- Optional Gaussian broadening (linewidth jitter)

---

## model.py — Neural network

Two classes, nothing else.

**`ResNet1D`** — 1-D residual encoder.

- Stem: Conv1d(1 → C₀, large kernel) + GroupNorm + ReLU + MaxPool
- 4 stages of `BasicBlock1D` (each: Conv-GN-ReLU-Conv-GN + residual, stride-2
  first block per stage)
- Global average pool → (B, C) embedding
- GroupNorm throughout (not BatchNorm): independent of batch composition, so
  small or uneven batches don't distort normalization statistics.

**`SpinHanceModel`** — encoder + 4 typed heads. The two regression heads also
emit a log-variance channel (μ and log σ²), used first as a diagnostic and only
later, with care, as a β-NLL loss.

```
shifts      head: Linear(emb → H) → ReLU → Dropout → Linear(H → 2G)
                  → split into μ and log σ² (per group)
j_mag       head: Linear(emb → H) → ReLU → Dropout → Linear(H → 2·G*(G-1)/2)
                  → split into μ and log σ² (per pair)
j_presence  head: same shape as the j_mag μ, outputs logits
deg_logits  head: Linear(emb → H) → ReLU → Dropout → Linear(H → G*C), view to (B,G,C)
```

Forward returns a dict:
`{shifts, shift_logvar, j_mag, jmag_logvar, j_presence, deg_logits}`.
log σ² is computed in fp32 and clamped (e.g. [−10, 5]) so `exp` can't overflow.

---

## train.py — Training, losses, metrics, CLI

Everything needed to take a model from random weights to a trained checkpoint.

### Losses

- `shift_loss`: smooth-L1 (Huber) on standardized shifts
- `jmag_loss`: smooth-L1, masked by ground-truth presence (absent couplings
  excluded — they are standardized to 0 but their error would dominate)
- `presence_loss`: BCE-with-logits, with `pos_weight` to counter sparsity
  (~70% of pairs are absent)
- `deg_loss`: cross-entropy with tempered inverse-frequency class weights
  (degeneracy is ~89% d=1; without weighting the head collapses)
- Initially, loss from chemical_shift is high

### Training regime

Single-stage training on the matrix loss. All four heads are trained jointly
from epoch 0, and the encoder regresses the spin-system matrix directly from
the input spectrum.

The loss is deliberately kept minimal until the diagnostics justify more. The
following are **candidate** additions, gated on what the diagnostics show, in
rough order of safety:

- **One-sided variance matching** `max(0, σ_target − σ_pred)²` per cell — the
  first anti-collapse term to try; targets under-dispersion directly without a
  scale-invariant optimum. (Pearson r is *not* used as a loss — monitoring only.)
- **ε-band local matching** — permute slots only among near-degenerate shifts
  (`|δ_i − δ_j| < ε`), canonical order elsewhere. Removes the canonical-ordering
  label noise without discarding the slot semantics the J head depends on. Full
  Hungarian matching stays an *eval metric* only.
- **β-NLL on the uncertainty head** — only after an MSE warmup; plain NLL lets
  the model inflate σ² to avoid learning hard cells.
- **Curriculum** (form → shifts → J → joint) — a hypothesis, not a default;
  A/B it against fixed multitask weights at equal compute before adopting.

The loss-weight schedule framework (a `Ramp` per term: start/end weight over a
progress window, cosine/linear/const) is built regardless — constant schedules
express the fixed-weight arm, so supporting both arms of that A/B costs nothing.

### EMA (exponential moving average)

- Keep a shadow copy of the weights, updated every optimizer step:
  `shadow = decay·shadow + (1 − decay)·live`, with `decay = ema_decay`
  (default 0.999).
- Validation, metrics, diagnostics, and checkpoint selection all use the
  **shadow** weights — the EMA model generalizes better, especially late in
  training where the live model plateaus.
- Decay is applied **per step**, not per epoch: over 60 epochs a per-epoch
  0.999 would barely move the shadow. If updating per epoch instead, use
  `decay ≈ 0.9`.

### Metrics (eval, no grad)

- `shift_mae_ppm` — mean absolute error in ppm (canonical ordering)
- `j_mae_hz` — MAE in Hz over ground-truth-present couplings only
- `presence_f1` — F1 on the binary coupling presence
- `deg_acc_balanced` — per-class recall averaged over degeneracy classes
- Hungarian-matched `shift_mae_ppm` / `j_mae_hz` (scipy `linear_sum_assignment`;
  G = 8, so the matching cost is negligible)

### Diagnostics

The point is to find *why* the model plateaus before adding objective terms —
mean-collapse is only one candidate; insufficient capacity, poor optimization,
and genuine information limits all produce a similar-looking output. Split into
instrumentation (emitted every run, on the shadow model, written to the probes
for the GUI) and control experiments (one training run each, as time allows).

**Per-run instrumentation**

- **Constant-mean baseline** — MAE of always predicting the per-slot training
  mean. Beating it shows the model learns *something*; what matters is the gap
  relative to the achievable floor (below), not to the baseline itself.
- **Per-cell Var(pred) / Var(target)** over the val set — ≪ 1 means that cell
  has collapsed to its mean.
- **Per-cell Pearson r** between prediction and target — ≈ 0 at low loss flags
  mean-collapse. Monitoring only, never a training objective (it is
  scale/offset-invariant; see Training regime).
- **Per-head gradient norms** — logged every `log_every` steps. Judge task
  balance from these, not from optimizer theory: if one head already dominates
  encoder updates, up-weighting it makes the imbalance worse.
- **Predicted log σ²** from the uncertainty head (see model.py) — per-cell
  aleatoric estimate. High σ² where error is high suggests genuine ambiguity;
  low σ² with high error suggests optimization/capacity (ignorance, not noise).
- **Train-vs-val gap** on every metric — separates optimization/capacity limits
  from generalization/information limits.

**Control experiments**

- **Shuffled-input control** — retrain with molecule↔spectrum assignments
  permuted. If val metrics match the real run, the encoder extracts no
  input-specific signal (a clean signal-usage test and pipeline sanity check).
- **Capacity / floor probe** — overfit a deliberately larger model until train
  error is near zero, then compare train vs. val vs. baseline. Separates "can't
  fit" (capacity/optimization) from "can't generalize" (data/information).

### Training loop

```
fit(records, assignment, cfg) -> model, std, vocab
```

- Builds the spectra RAM cache, `SpinDataset`, and train/val loaders
- Fits `Standardizer` and `DegeneracyVocab` on train records
- Computes class balance weights on train records
- AdamW, linear warmup → cosine decay, bfloat16 AMP, grad clip
- Logs every `log_every` steps to `events.jsonl` (no per-step host sync): total
  loss, every loss term (raw and weighted), and per-head gradient norms; per
  **epoch** to `metrics.jsonl`: all metrics and diagnostics
- Per-epoch: train → val/metrics/diagnostics **on EMA weights** → checkpoint
  (`best.pt` on improvement, plus `last.pt` every `save_every` epochs)
- Early stopping on `shift_mae_ppm + j_mae_hz / 10`
- Checkpoints: `{model_state, ema_state, optimizer_state, standardizer, vocab,
  cfg, epoch, metrics}` as a plain dict saved with `torch.save`

### Data storage

- train.py is run remotely on AWS EC2
- All data is saved to AWS S3
- train.py saves an entire structure of data for the training session
- Diagnostic data is saved per each epoch
- At session start, train.py samples **500 molecules from the held-out test
  fold** (seeded, fixed for the session) and writes them to the session root as
  `diagnostic_set.json` (input records + ground-truth spin systems) and
  `diagnostic_spectra.npy` (their spectra, fp16). Materializing the set
  server-side is the guarantee that GUI diagnosis never touches a trained-on
  molecule — the GUI reads this set rather than re-deriving the split
- The data structure is as follows

```text
s3://spinhance-data/training/sessionXXX/
├── config.json
├── status.json
├── metrics.jsonl
├── events.jsonl
├── summary.json
├── diagnostic_set.json
├── diagnostic_spectra.npy
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── probes/
    └── epochXXX/
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
```

### CLI

```bash
PYTHONPATH=. python -m modelv2.train --spin_systems=<spin_systems_in_file> --spectra=<spectra_in_file>
```

Key flags: `--epochs`, `--batch`, `--lr`, `--ckpt`, `--device`, `--seed`,
`--dry-run`.

`--dry-run` exercises the entire data path (adapter → splits → standardizer →
target encoding → renderable mask) without touching torch.

---

## gui.py

**Purpose**

- All in one GUI for visualization, diagnostic, and exploration of model quality
- The human element for designing the training of the neural network
- Essential to the success of this project

**Framework**

- The GUI is a Streamlit app, run with `streamlit run modelv2/gui.py`. This is a
  binding design decision: Streamlit is *the* GUI framework for modelv2, not a
  placeholder or one option among several.
- Chosen because it matches the "low code, short solutions" goal — dashboards,
  S3/file browsing, and plots come with minimal boilerplate.

**Design**

- Reuses the layout and feature set of the existing model/gui.py and model/live_dashboard.py
- Emphasis on low code, short solutions
- Usability and responsiveness comes second to functionality

**Storage**

- Because the models are trained remotely on S3, the data (checkpoints, diagnostic files, etc) will be stored on S3
- All of the data must be retrieved from S3
- The viewer and diagnostics run on the held-out `diagnostic_set.json` (+ `diagnostic_spectra.npy`) — 500 molecules the model never saw — so being held out is true by construction, not re-derived from the split
- Simple caching is a plus (e.g. Streamlit's `st.cache_data` for S3-retrieved data)
- The diagnostic data stored during the training loop (vide infra) is based on the requirements for display in gui.py

---

## Scientific decisions carried forward unchanged

| Decision | What | Why kept |
|---|---|---|
| ResNet-1D encoder | 1-D conv stack, global pool | Fast, stable, well-matched to multiplet structure |
| Four typed heads | Shift/J-mag (regression) + presence (binary) + deg (classification) | Separating *whether* from *how big* keeps magnitude head from fighting structural zeros |
| Canonical ordering | Lexsort shift↓ / deg↓ / J-rowsum↓ | Makes S₈-invariant problem trainable with per-element loss |
| Scaffold + dedup split | Molecule-level, scaffold-grouped, near-dup-grouped | No leakage between train/val/test at the spin-system level |
| Class balance | Tempered inverse-freq deg weights + BCE pos_weight | Prevents degeneracy and presence heads from collapsing to majority class |
| GroupNorm | In encoder blocks | Independent of batch composition and batch size |

## What is dropped

| Dropped | Reason |
|---|---|
| `probes.py`, `failure_analysis.py` as separate files | Folded into `train.py` (Diagnostics) and the per-epoch probe outputs |
| `data_adapter.py`, `run_experiment.py` as separate files | Folded into `data.py` and the CLI section of `train.py` |
| `live_dashboard.py` | Infrastructure, not model |
| `schedules.py` as a separate file | Two small functions; inlined into `train.py` |
| `metrics.py` as a separate file | Inlined into `train.py` |

## Configuration

One `TrainConfig` dataclass in `train.py`. No YAML, no JSON config files, no
nested config systems. Defaults are sane for the full dataset on one GPU.

```python
@dataclass
class TrainConfig:
    # Data
    points: int = 16384
    ppm_from: float = 0.0
    ppm_to: float = 12.0
    spectrum_field: str = "spec90"
    seed: int = 0
    # Model
    n_groups: int = 8
    # Training
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    epochs: int = 60
    warmup_frac: float = 0.03
    linewidth_hz: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {
        "shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5})
    patience: int = 10
    ema_decay: float = 0.999        # per-step EMA; 0 disables
    save_every: int = 10            # periodic last.pt for crash safety
    log_every: int = 50             # steps between event-log flushes (no per-step sync)
    # Infrastructure
    device: str = "cuda"
    amp_dtype: str = "bf16"
    ckpt_path: str = "checkpoint.pt"
    cache_spectra: bool = True      # stream mol_all.tar.gz into RAM at startup
    gpu_augment: bool = True        # vectorized augmentation on-GPU per batch
    num_workers: int = 0            # RAM cache + GPU aug: no CPU loader needed
    compile: bool = False           # opt-in torch.compile; falls back to eager
    val_every: int = 1
```

## Runtime & hardware

Target: a single **AWS g6e.16xlarge** — 1× NVIDIA L40S (48 GB, Ada
Lovelace), 64 vCPU, 512 GiB RAM, ~1.9 TB local NVMe. Functionality first,
efficiency second: every item below must degrade gracefully if unavailable.

**Single GPU.** No DDP or multi-GPU paths. 48 GB easily holds batch 256 of the
1-D ResNet, so no gradient accumulation is needed — the GPU is not the
constraint.

**Keep the GPU fed (avoid CPU blocking).**

- **Spectra RAM cache** (`cache_spectra`): stream `mol_all.tar.gz` once into a
  single fp16 array at startup and serve slices from RAM. Zero per-epoch disk
  I/O; never `np.load` per item.
- **GPU-side augmentation** (`gpu_augment`): apply `augment_spectrum`
  vectorized to the batch on the GPU, not per item on the CPU. This keeps
  `num_workers = 0`, sidesteps the Python 3.14 `forkserver` pickling pitfalls
  entirely, and removes the data loader as a bottleneck.
- `pin_memory=True` with non-blocking host→device copies for the cached inputs.

**L40S throughput knobs.** bf16 AMP (no loss scaling needed on Ada);
`torch.set_float32_matmul_precision("high")` and `allow_tf32 = True` for matmul
and cuDNN; `cudnn.benchmark = True` (input length is fixed at 16384, so
autotuning pays off).

**Crash safety.** `last.pt` is written unconditionally every `save_every`
epochs, in addition to `best.pt` on improvement.

## Design constraints

Ordered by priority: a run must first *not fail*, then *be correct*, then *be
fast*. The stability and correctness items are hard constraints (assert /
fail-fast); the speed items are defaults.

**Numerical stability — the run must not NaN or diverge.**

- bf16 autocast only — never fp16 / `GradScaler`. bf16 keeps fp32's exponent
  range, so there is no overflow path. Master weights, every loss, and every
  reported metric stay in **fp32**; upcast logits before BCE/CE and predictions
  before the regression losses. (bf16's 7-bit mantissa is a ~0.4% relative
  floor — left inside the loss it would cap exactly the fine shift/J precision
  we are trying to recover.)
- Guard every masked or weighted reduction against divide-by-zero: the
  presence-masked J mean (empty mask → 0/0), the tempered inverse-frequency deg
  weights and BCE `pos_weight` (zero-count class → ∞), and the `Standardizer`
  std (zero-variance cell → ∞). Floor all denominators, and clamp the predicted
  `log σ²` before `exp`.
- Always on: linear warmup → cosine LR, global grad-norm clip ≤ 1.0 every step,
  AdamW. A non-finite loss skips + logs the batch and aborts if the skip rate
  crosses a threshold.
- GroupNorm group count must divide every channel width (assert at build). It
  also has no running buffers (unlike BatchNorm), so EMA tracks parameters only
  and batch composition never affects normalization — keep it.

**Correctness — it must learn the right thing.**

- One shared `canonical_order` and one upper-triangle index map (i < j,
  row-major) used by target encoding, the heads, the losses, and the metrics. A
  mismatch silently corrupts J/presence learning and is nearly invisible.
- `Standardizer` is fit on **train only**, saved in the checkpoint, and reused
  verbatim at val and inference — never re-fit per split (leakage / scale drift).
- EMA shadow is initialized from the live weights and updated after the
  optimizer step; val, checkpoint selection, and diagnostics always read the
  shadow.
- Validate at load and fail fast in `--dry-run`: all degeneracies ∈ vocab, all
  shift/J std > 0, presence and J index maps agree. `--dry-run` and `--smoke`
  must pass before any GPU run.

**Speed — only once the above hold.**

- No host↔device sync in the inner loop: no `.item()`, `.cpu()`, or `print` per
  step. Accumulate metrics on-GPU and flush every `log_every` steps. This is
  usually the line between GPU-bound and stalled.
- Keep the GPU fed: RAM cache + GPU augmentation + `num_workers = 0` + pinned,
  non-blocking H→D copies + `zero_grad(set_to_none=True)`; fixed 16384 length ⇒
  `cudnn.benchmark`.
- Batch size stays at 256. Do **not** inflate it to fill 48 GB: larger batches
  need LR re-tuning and smooth the gradient toward the per-cell mean — they make
  the underlearning *worse*. Spend the headroom on model size or more epochs.
- `torch.compile` and gradient checkpointing are **off** by default. compile is
  opt-in behind a `try/except → eager` fallback (enable only after a stable
  baseline); checkpointing only slows a run that is not memory-bound.

## Running

```bash
# Validate the data path, no training, no torch:
PYTHONPATH=. python -m modelv2.train --dry-run

# Train
PYTHONPATH=. python -m modelv2.train --spin_systems=mol_to_spin_system/data/spin_systems_pubchem.json.tar.gz --spectra=simulation/data/spectra/90MHz/mol_all.tar.gz

# Visualize (Streamlit app)
PYTHONPATH=. streamlit run modelv2/gui.py

# Smoke test (synthetic data, no real spectra needed):
PYTHONPATH=. python -m modelv2.train --smoke
```
