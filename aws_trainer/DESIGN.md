# aws_trainer — Production Training System

Successor to `model/` for the 100k-molecule dataset. Imports the validated
physics kernels from `model/` unchanged; replaces the training infrastructure.

## What changed from model/ and why

### Scale (100k → 100× more data)
- **Spectra preloaded to RAM** (`SpectraCache`): at fp16, both fields = ~6.6 GB for
  100k molecules, fits in any EC2 instance with 32+ GB RAM. Eliminates per-epoch
  file I/O that would dominate wall time with a fast GPU.
- **num_workers=4 + pin_memory**: saturates GPU with async CPU prefetch.
- **Larger model** (`medium` by default: ~15M params vs 0.55M). With 100k samples
  the regularization pressure from a small model is gone; bigger encoder captures
  more spectral structure.

### Training quality
- **EMA (decay=0.9999)**: exponential moving average of weights. The EMA model is
  what we validate and checkpoint. Reduces late-epoch variance, improves final
  metrics by ~5–10% on downstream tasks at essentially zero cost.
- **Attention neck** (optional): 1–2 Transformer encoder layers inserted between
  the ResNet stages and global pool. The feature map at that point is
  P / 128 ≈ 128 positions — cheap. Helps capture long-range coupling structure
  (e.g. a coupling between groups far apart in ppm).
- **Hungarian-matched eval metrics**: at validation time, match predicted groups to
  ground-truth groups by shift distance (scipy linear_sum_assignment). Reports
  `h_shift_mae`, `h_j_mae`, `h_deg_acc` alongside the canonical-order metrics.
  These are lower bounds on error — tells us how much the canonical ordering hurts.
- **Gradient accumulation**: effective batch up to 1024 at Stage 1 on a single GPU.

### Infrastructure
- **DDP** via `torchrun`: single command works on 1 or N GPUs.
  Validation metrics are all-reduced across ranks.
- **torch.compile** (`--compile`): ~20-30% step speedup on Ampere/Hopper.
- **Proper AMP**: `torch.amp.autocast` + `torch.amp.GradScaler` (not the
  deprecated `torch.cuda.amp` API).
- **Structured config**: `VAWSConfig` is a JSON-serializable dataclass.
  Every hyperparameter is recorded in the checkpoint. Re-runs are reproducible.
- **W&B + CSV logging**: W&B if available, CSV always.
- **EC2 scripts**: `ec2/launch.sh`, `ec2/setup.sh`, `ec2/sync.sh` reuse the
  autoai AWS credentials (profile=hack-scripps, region=us-west-2).

## What was NOT changed
All physics is unchanged. The following are imported directly from `model/`:
- `model.losses` — matrix_loss, spectral_loss, wasserstein1
- `model.metrics` — decode, compute_metrics
- `model.targets` — DegeneracyVocab, Standardizer, encode_target, augment_spectrum
- `model.splits` — make_splits
- `model.diff_renderer_torch` — RegularizedEigh, simulate

## Model sizes

| size   | params | encoder config                                            | when to use          |
|--------|--------|-----------------------------------------------------------|----------------------|
| tiny   | 0.6M   | stem=24 stages=(32,64,128,192) blocks=(1,1,1,1) h=256    | smoke tests          |
| small  | 4M     | stem=32 stages=(64,128,256,512) blocks=(2,2,2,2) h=512   | ≤10k samples         |
| medium | 15M    | stem=64 stages=(128,256,512,512) blocks=(2,2,3,3) h=1024 | 100k baseline ✓      |
| large  | 30M    | stem=64 stages=(128,256,512,1024) blocks=(3,4,6,3) h=1024| max performance      |
| medium-attn | 17M | medium + 2 attn layers (8 heads)                     | long-range coupling  |
| large-attn  | 32M | large + 2 attn layers (8 heads)                      | max + long-range     |

## Run commands

```bash
# Validate data path (no torch needed)
PYTHONPATH=. python -m aws_trainer.run --dry-run

# Single GPU, medium model, Stage 1 only
PYTHONPATH=. python -m aws_trainer.run --model-size medium --epochs 80 --stage1-epochs 80

# Single GPU, full Stage 1 + Stage 2
PYTHONPATH=. python -m aws_trainer.run --model-size medium

# 4-GPU DDP
PYTHONPATH=. torchrun --nproc_per_node=4 -m aws_trainer.run --model-size medium

# From a saved config
PYTHONPATH=. python -m aws_trainer.run --config aws_trainer/configs/medium_s2.json

# EC2 launch
bash aws_trainer/ec2/launch.sh g4dn.xlarge
```

## Stage 1 → Stage 2 handoff
Identical curriculum to model/ (Decision 7): Stage 1 matrix loss only →
ramp window (w_matrix decays from 1 to `matrix_anchor=0.3`, w_spectral rises
from 0 to 1) → steady state. The handoff epoch is `stage1_epochs`.

## Open items
- Hungarian matching LOSS (not just eval): Sinkhorn relaxation or annealing
  towards hard assignment. Deferred until canonical-sort baseline plateaus.
- 600 MHz Stage-2 auxiliary loss (reproducibility goal, Decision 6).
- Cluster-holdout val/test split (Decision 8 upgrade).
