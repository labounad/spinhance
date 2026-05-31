# Suggested improvements to model/

These are things missing from `model/` that are needed for a serious training run
(especially on EC2 with the 60k dataset). Listed roughly in priority order.

---

## 1. Spectra RAM cache (SpectraCache)

**What's missing:** `model/dataset.py` calls `np.load(path)` from disk on every
`__getitem__`. For 60k molecules × N epochs, that's millions of small file reads
and will be the bottleneck on any networked or HDD-backed filesystem.

**What to add:** A cache class that preloads all spectra into a single fp16 numpy
array at startup. Loading from disk takes ~60 s once; subsequent epoch I/O is
zero. Also needs to support loading from `mol_all.tar.gz` (sequential streaming
gzip pass) as the alternative to 64k individual files.

```python
class SpectraCache:
    def __init__(self, records, field=90):  # loads all to RAM as fp16
        ...
    def __getitem__(self, idx) -> np.ndarray:  # returns float32 view
        ...
```

`SpectrumMatrixDataset` should accept an optional cache and use it in
`_load_spectrum` instead of hitting disk.

---

## 2. EMA (Exponential Moving Average)

**What's missing:** `model/train.py` has no EMA. The shadow model generalizes
better than the live model, especially late in training.

**What to add:**
- After each epoch (not each step), update shadow weights:
  `shadow = decay * shadow + (1 - decay) * live`  with `decay = 0.9999`
- Use the **shadow model** for validation and checkpoint saving, not the live model
- Save both `model` and `ema_model` state dicts in the checkpoint

One epoch-level update (not step-level) avoids the warmup bias correction
complexity and is stable for long runs.

---

## 3. Gradient accumulation

**What's missing:** `model/train.py` has no `accum_steps`. At batch_size=256 on a
T4 (16 GB), you might not fit the full batch. Gradient accumulation lets you use
micro-batches of 64 while computing gradients equivalent to batch 256.

**What to add:** `accum_steps: int = 1` in `TrainConfig`. Inside `train_epoch`,
divide loss by `accum_steps`, skip `opt.step()` on intermediate steps, and only
clip + step on the final accumulated step.

---

## 4. tar.gz spectrum loading

**What's missing:** `model/data_adapter.py` checks whether `mol_all.tar.gz`
exists (for the `require_spectra` gate), but `model/dataset.py._load_spectrum`
only reads individual `.npy` files. If the tar is the only thing present, loading
will crash at `__getitem__` time.

**What to add:** The cache class (see #1) should detect `mol_all.tar.gz` in the
parent directory and do one sequential streaming pass through it to populate the
cache, instead of reading individual files. This is the path used when uploading
spectra to S3 as a single archive rather than 64k files.

---

## 5. 90 MHz only / configurable fields

**What's missing:** `run_experiment.py` hardcodes `fields=(90, 600)` in `_load`.
This requires both 90 MHz AND 600 MHz spectra to exist for a molecule to be
included. On EC2 you only upload the 90 MHz data → 0 records loaded.

**What to fix:** Pass `fields` through from a CLI flag. Default should be `(90,)`.

```
--fields 90       # default — 90 MHz only (production run)
--fields 90,600   # both (development, requires both spectra on disk)
```

---

## 6. Scaffold split: opt-in, not opt-out

**What's missing:** The CLI uses `--no-scaffold` to skip scaffold splitting.
`MurckoScaffoldSmiles(mol=mol)` crashes on RDKit 2026.03 when a double bond has
`STEREOANY` stereo (which appears in the 60k dataset). The crash is deep in
`Canon.cpp` and is not catchable in Python.

**What to fix:** Flip to `--scaffold` (store_true, default off). Scaffold
splitting is a nice-to-have; training must not crash by default.

---

## 7. Hungarian-matched shift MAE metric

**What's missing:** `model/metrics.py` computes `shift_mae_ppm` by direct
element comparison (both sides in canonical order). This is correct for training
loss but misleading for evaluation: even if the model's predictions are in the
wrong order, Hungarian matching finds the permutation that minimizes error and
gives a tighter bound on real-world usefulness.

**What to add:** Use `scipy.optimize.linear_sum_assignment` on the cost matrix
`|pred_shifts - tgt_shifts|` to get the best-permutation MAE. Report as
`h_shift_mae_ppm` alongside the canonical `shift_mae_ppm`.

---

## 8. Checkpoint completeness + resume

**What's missing:** The checkpoint is `{"model", "standardizer", "cfg"}`. Missing:
- `epoch` — needed to know where to resume
- `metrics` — needed to compare runs without re-evaluating
- `optimizer` state — needed to resume without LR schedule reset
- `ema_model` — (once EMA is added, see #2)

**Also missing:** Any resume logic. On EC2 spot instances, interruptions happen.
Add `--resume path/to/last.pt` that restores model, optimizer, and scheduler
state and picks up from the saved epoch.

---

## 9. Periodic checkpoints (`save_every`)

**What's missing:** `model/train.py` only saves when a new best score is achieved.
If the run is interrupted or the best checkpoint is corrupted, you lose everything.

**What to add:** `save_every: int = 10` — save `last.pt` unconditionally every N
epochs regardless of score, in addition to `best.pt` on improvement.

---

## 10. Model size presets

**What's missing:** `run_experiment.py` only has a boolean `--small` flag that
hardcodes one specific small architecture. For serious training you want named
presets with documented parameter counts.

**What to add:** A `build_model(model_size: str) -> SpinHanceModel` helper with
presets like `tiny` (~0.5 M), `small` (~2 M), `medium` (~8 M), `large` (~20 M).
Wire it to `--model-size {tiny,small,medium,large}`.

---

## 11. Multi-GPU / DDP

**What's missing:** `model/train.py` is single-GPU only. A p3.8xlarge has 4 V100s
sitting idle.

**What to add:** Standard torchrun / DistributedDataParallel wiring:
- Detect `LOCAL_RANK` / `WORLD_SIZE` env vars
- `dist.init_process_group("nccl")`
- Wrap model in `DDP`
- Only log and save checkpoints on rank 0
- Use `DistributedSampler` for the data loaders

This is the one change that requires the most boilerplate but also has the biggest
throughput impact.

---

## 12. Progress logging (TensorBoard or W&B)

**What's missing:** The training loop only calls `print()`. Once you're watching a
run on EC2 via `tail -f`, you can't plot loss curves, compare runs, or set up
alerts.

**What to add:** At minimum, write a TensorBoard `SummaryWriter` to `runs/`.
Optionally wire `--wandb-project` for W&B. Both can be conditional on the library
being installed.

---

## 13. Augmentation parameter exposure

**What's missing:** `TrainConfig` has no augmentation parameters. Noise level,
reference shift jitter, and baseline amplitude are buried as defaults inside
`targets.augment_spectrum`.

**What to add:**
```python
aug_noise_frac: float = 0.01
aug_ref_shift_ppm: float = 0.01
aug_baseline_amp_frac: float = 0.02
```
Pass these through as `aug_kwargs` in `build_datasets`.

---

## 14. Python 3.14 / forkserver compatibility

**What's missing:** Python 3.14 changed the default multiprocessing start method
to `forkserver`, which requires all objects passed through DataLoader workers to be
picklable at module import time. Locally-defined classes (defined inside a function
body) are not picklable by forkserver.

**What to fix:**
- Set `num_workers=0` when using a RAM cache (no I/O benefit from workers anyway)
- Any helper class passed to DataLoader workers must be defined at module level,
  not inside a function

---

## Summary table

| # | Feature | Effort | Impact |
|---|---------|--------|--------|
| 1 | Spectra RAM cache + tar.gz load | Medium | High — eliminates I/O bottleneck |
| 2 | EMA | Small | High — better generalization |
| 3 | Gradient accumulation | Small | High — enables large effective batch on small GPU |
| 4 | tar.gz loading in dataset | Small | Required for EC2 deployment |
| 5 | 90 MHz only fields | Trivial | Required — currently loads 0 records on EC2 |
| 6 | Scaffold opt-in | Trivial | Required — default crashes on 60k dataset |
| 7 | Hungarian shift MAE | Small | Medium — honest evaluation metric |
| 8 | Checkpoint completeness + resume | Medium | High — essential for spot instances |
| 9 | save_every periodic ckpt | Trivial | Medium — crash safety |
| 10 | Model size presets | Small | Medium — ergonomics |
| 11 | Multi-GPU DDP | Large | High throughput on multi-GPU instances |
| 12 | TensorBoard / W&B | Small | Medium — remote monitoring |
| 13 | Aug parameter exposure | Trivial | Low — correctness, not performance |
| 14 | Python 3.14 forkserver | Trivial | Required on Python 3.14+ |
