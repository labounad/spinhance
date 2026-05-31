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

```
modelv2/
├── DESIGN.md          this file
├── renderer.py        differentiable NMR spectrum renderer (composite manifold reduction)
├── data.py            splits, target encoding, dataset, augmentation, standardization
├── model.py           neural network (encoder + heads)
└── train.py           losses, metrics, training loop, evaluation, CLI
```

Four files. No more, no less. Every piece of the pipeline lives in one of
them and only one. No circular imports, no indirection.

---

## renderer.py — Differentiable NMR renderer

The most scientifically critical and most intrinsically complex piece.
Unchanged in spirit from `composite_diff.py` + `diff_renderer_torch.py`, but
kept in one file with no external imports beyond `numpy` and `torch`.

**Physics:** manifold reduction + Mz block-diagonalization. Each group of _d_
equivalent spins reduces to its total-spin manifolds; we diagonalize the small
Mz blocks — never a dense 2^N Hamiltonian. Cost is bounded by the largest
Mz block, not total spins; this covers ~100% of the dataset (vs ~89% for the
explicit 2^N renderer).

**Key components:**
- `build_plan(degeneracy)` — parameter-independent structure (Mz blocks, F⁺
  matrices, coupling pair indices). Cached per degeneracy pattern. Shared
  between the numpy oracle and the torch forward.
- `simulate_np(shifts, couplings, degeneracy, field_mhz, ...)` — numpy oracle.
  Used for ground-truth spectrum generation and gradient checking.
- `simulate_batch(shifts, couplings, degeneracy, field_mhz, ...)` — batched
  PyTorch forward. All samples in a batch must share a degeneracy pattern
  (single-bucket assumption from the bucketed sampler).
- `RegularizedEigh` — custom autograd Function. Backward uses the
  Lorentzian-regularized VJP `F_ij = ΔE / (ΔE² + ε²)` to handle the exact
  degeneracies that equivalent-spin expansion produces. ε ≈ 1 Hz (linewidth
  scale). No `eigh` in bfloat16/float16 — upcasts to float32 for the
  eigendecomposition only.
- `broaden_fft_batch(centers, amps, ...)` — bin sticks via linear interpolation
  then FFT-convolve with a Lorentzian kernel. O(P log P), independent of the
  number of transitions.

**No connected-component splitting.** Predicted couplings are continuous and
soft-gated at training time, so the coupling graph is effectively dense and
component structure would be non-differentiable across J = 0. Manifold
reduction needs no zero-coupling assumption and is fully differentiable.

---

## data.py — Data pipeline

Everything that touches molecules before the model sees them, in one file.

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

**Dataset:**
- `SpinDataset(records, vocab, std, augment, ...)` — one item = one molecule.
  Loads spectrum from `rec["spec90"]` (in-memory array) or `rec["spec90_path"]`
  (memory-mapped .npy). Applies `augment_spectrum` on-the-fly if `augment=True`.
- `BucketSampler(bucket_keys, batch_size)` — yields batches where all samples
  share a degeneracy pattern (same `tuple(degeneracy)` after canonical
  ordering). Enables `simulate_batch` to reuse the renderer plan across the
  batch.
- `collate(batch)` — stacks tensors; adds `shared_degeneracy` (not None iff
  all samples in the batch share a pattern).

**Augmentation** (`augment_spectrum`):
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
  the bucketed sampler and small batches don't distort normalization statistics.

**`SpinHanceModel`** — encoder + 4 typed heads.
```
shifts      head: Linear(emb → H) → ReLU → Dropout → Linear(H → G)
j_mag       head: Linear(emb → H) → ReLU → Dropout → Linear(H → G*(G-1)/2)
j_presence  head: same shape as j_mag, outputs logits
deg_logits  head: Linear(emb → H) → ReLU → Dropout → Linear(H → G*C), view to (B,G,C)
```
Forward returns a dict: `{shifts, j_mag, j_presence, deg_logits}`.

---

## train.py — Training, losses, metrics, CLI

Everything needed to take a model from random weights to a trained checkpoint.
No subprocess, no S3, no Streamlit — just training.

### Losses

**Matrix loss** (Stage 1 and Stage 2 anchor):
- `shift_loss`: smooth-L1 (Huber) on standardized shifts
- `jmag_loss`: smooth-L1, masked by ground-truth presence (absent couplings
  excluded — they are standardized to 0 but their error would dominate)
- `presence_loss`: BCE-with-logits, with `pos_weight` to counter sparsity
  (~70% of pairs are absent)
- `deg_loss`: cross-entropy with tempered inverse-frequency class weights
  (degeneracy is ~89% d=1; without weighting the head collapses)

**Spectral loss** (Stage 2):
- Decode predicted matrix to physical units; soft-gate J magnitudes with
  `sigmoid(j_presence)` so the coupling head gets gradient through the renderer
- `simulate_batch` → Wasserstein-1 vs reference spectrum
- Wasserstein-1: normalize both to probability distributions, integrate |CDF_a - CDF_b|
- Rendered at 90 MHz (self-consistency with the input)

### Training stages

**Stage 1** (epochs 0 … stage1_epochs): matrix loss only. Builds a stable,
identifiable baseline before the spectral term is introduced.

**Stage 2** (epochs stage1_epochs … total_epochs): curriculum blend.
`w_mat` decays 1.0 → `matrix_anchor` over `ramp_epochs` epochs while `w_spec`
ramps 0 → `spectral_max`. Both remain active; the matrix anchor prevents
identifiability drift (spectral signal alone can wander to a spectrum-equivalent
but wrong matrix — the problem is genuinely ill-posed at 90 MHz).

Stage-2 cost control:
- Only samples from single-bucket batches get a spectral term (others fall back
  to matrix loss only).
- A random subset of the batch (≈20%) is rendered per step; expected gradient
  is unbiased, variance is acceptable for SGD.
- K guard: count spectral lines from the renderer plan (CPU, free) before
  allocating GPU tensors. Skip the spectral term for batches where K > 1 M.
- Render one sample at a time inside the selected subset.

### Metrics (eval, no grad)

- `shift_mae_ppm` — mean absolute error in ppm (canonical ordering)
- `j_mae_hz` — MAE in Hz over ground-truth-present couplings only
- `presence_f1` — F1 on the binary coupling presence
- `deg_acc_balanced` — per-class recall averaged over degeneracy classes
- Hungarian-matched variants of the above (scipy, skipped if unavailable)

### Training loop

```
fit(records, assignment, cfg) -> model, std, vocab
```

- Builds `SpinDataset`, `BucketSampler`, val loader
- Fits `Standardizer` and `DegeneracyVocab` on train records
- Computes class balance weights on train records
- AdamW, linear warmup → cosine decay, bfloat16 AMP, grad clip
- Per-epoch: train → val → checkpoint (best + last)
- Early stopping on `shift_mae_ppm + j_mae_hz / 10` once Stage 2 is active
- Checkpoints: `{model_state, standardizer, vocab, cfg, epoch, metrics}` as a
  plain dict saved with `torch.save`

### CLI

```
python -m modelv2.train --data-json <path> --spectra <dir> [options]
```

Key flags: `--epochs`, `--stage1-epochs`, `--ramp-epochs`, `--batch`,
`--lr`, `--ckpt`, `--device`, `--seed`, `--dry-run`.

`--dry-run` exercises the entire data path (adapter → splits → standardizer →
target encoding → renderable mask) without touching torch.

---

## Scientific decisions carried forward unchanged

| Decision | What | Why kept |
|---|---|---|
| ResNet-1D encoder | 1-D conv stack, global pool | Fast, stable, well-matched to multiplet structure |
| Four typed heads | Shift/J-mag (regression) + presence (binary) + deg (classification) | Separating *whether* from *how big* keeps magnitude head from fighting structural zeros |
| Canonical ordering | Lexsort shift↓ / deg↓ / J-rowsum↓ | Makes S₈-invariant problem trainable with per-element loss |
| Composite renderer | Manifold reduction + Mz blocks | Covers 100% of dataset; bounded cost; no 2^N blow-up |
| Regularized eigh | Lorentzian VJP `F = ΔE/(ΔE²+ε²)` | Exact degeneracies from equivalent spins cause 1/0 in naive backward |
| Staged training | Matrix loss → curriculum ramp → spectral anchor | Avoids cold-start instability and identifiability drift |
| Wasserstein-1 spectral loss | CDF distance | Permutation-invariant; degrades gracefully on small misalignments |
| Bucketed sampler | Batch by degeneracy pattern | Enables renderer plan reuse; stage-2 cost is O(1) plan builds per epoch |
| Scaffold + dedup split | Molecule-level, scaffold-grouped, near-dup-grouped | No leakage between train/val/test at the spin-system level |
| Class balance | Tempered inverse-freq deg weights + BCE pos_weight | Prevents degeneracy and presence heads from collapsing to majority class |
| GroupNorm | In encoder blocks | Independent of batch composition; works with bucketed sampler |

## What is dropped

| Dropped | Reason |
|---|---|
| `diagnostics.py`, S3 I/O, live dashboard, gui.py | Out-of-scope infrastructure; a training run writing a checkpoint file is sufficient |
| `probes.py`, `failure_analysis.py` | Useful diagnostics, but not needed for the model to train and evaluate correctly; add back when the training loop itself works |
| `data_adapter.py`, `run_experiment.py` as separate files | Folded into `data.py` and the CLI section of `train.py` |
| `diff_renderer_ref.py` (explicit 2^N oracle) | Kept only for gradient checking; not needed in the training code path |
| `live_dashboard.py` | Infrastructure, not model |
| `stage1.py` / `stage2.py` as separate files | Unnecessary split; the epoch logic is ~50 lines each and reads more clearly together |
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
    stage1_epochs: int = 20
    ramp_epochs: int = 10
    spectral_max: float = 1.0
    matrix_anchor: float = 0.3
    warmup_frac: float = 0.03
    render_subset_frac: float = 0.2
    linewidth_hz: float = 1.0
    eigh_eps: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {
        "shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5})
    patience: int = 10
    # Infrastructure
    device: str = "cuda"
    amp_dtype: str = "bf16"
    ckpt_path: str = "checkpoint.pt"
    num_workers: int = -1
    val_every: int = 1
```

## Running

```bash
# Validate the data path, no training, no torch:
PYTHONPATH=. python -m modelv2.train --dry-run

# Train (Stage 1 only):
PYTHONPATH=. python -m modelv2.train --ckpt run1.pt

# Train (Stage 1 + Stage 2):
PYTHONPATH=. python -m modelv2.train --stage2 --ckpt run1.pt

# Smoke test (synthetic data, no real spectra needed):
PYTHONPATH=. python -m modelv2.train --smoke
```
