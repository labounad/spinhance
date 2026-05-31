# SpinHance

Extract ¹H chemical shifts and scalar coupling constants from low-field (90 MHz) ¹H NMR spectra using deep learning. Given a spectrum, the model predicts the underlying spin-system parameters — shifts δ (ppm), couplings J (Hz), and proton degeneracies — that reproduce it exactly at any field strength.

---

## Setup

```bash
micromamba env create -f environment.yml
micromamba activate spinhance
```

All commands below assume the environment is active and you are in the repo root.

---

## Pipeline Overview

```
SMILES
  └─ generate/          screen to 8 spin groups
       └─ mol_to_spin_system/    compute shift+J matrix
            └─ simulation/      simulate spectra at 90 + 600 MHz
                 └─ aws_trainer/    train model
                      └─ model/gui.py    evaluate
```

---

## Stage 1 — Generate molecule list (`generate/`)

Screens a SMILES source and filters to molecules with exactly 8 magnetically distinct ¹H spin groups.

```bash
# Screen ChEMBL (downloads automatically)
python -m generate.pipeline --source chembl --out generate/data/smiles_8group.csv

# Screen a custom SMILES file
python -m generate.pipeline --source my_smiles.csv --out generate/data/smiles_8group.csv
```

Output: `generate/data/smiles_8group.csv` — one molecule per row with SMILES, InChIKey, and group sizes.

---

## Stage 2 — Compute spin-system matrices (`mol_to_spin_system/`)

3D-embeds each molecule and computes heuristic chemical shifts (HOSE codes) and coupling constants (Karplus + literature tables).

```bash
python -m mol_to_spin_system.pipeline \
  --smiles generate/data/smiles_8group.csv \
  --out mol_to_spin_system/data/spin_systems.json
```

Output: `mol_to_spin_system/data/spin_systems.json` — a JSON array of spin-system records:

```json
{
  "chembl_id": "CHEMBL6622", "smiles": "...", "inchikey": "...",
  "labels": ["A", "B", "C", "D", "E", "F", "G", "H"],
  "spin_groups": [[4.59, 1], [4.05, 1], ...],
  "couplings": [["A", "C", 5.7], ["B", "C", -10.8], ...]
}
```

`spin_groups[i]` = `[shift ppm, #protons]` for `labels[i]`. Absent couplings are J = 0.

The production 60k dataset is already at `mol_to_spin_system/data/spin_systems_chembl.json`.

---

## Stage 3 — Simulate NMR spectra (`simulation/`)

Runs quantum spin simulation at 90 MHz and 600 MHz using the pure-Python engine (pyspin). License-free, parallel, validated against MestReNova (r ≈ 0.999).

```bash
# Simulate all molecules in the JSON (uses all CPU cores)
python -m simulation.cli run \
  --graphs mol_to_spin_system/data/spin_systems_chembl.json \
  --out simulation/data/spectra \
  --engine python \
  --workers 8
```

Output: `simulation/data/spectra/{90,600}MHz/mol_XXXXXX.npy` — 16384-point normalized intensity arrays (∫ = 1, 0–12 ppm).

---

## Stage 4 — Train the model

### Local training (CPU / single GPU)

```bash
# Validate data paths before training (torch-free)
PYTHONPATH=. python -m aws_trainer.run \
  --json mol_to_spin_system/data/spin_systems_chembl.json \
  --dry-run

# Stage 1 only (matrix loss) — fast, good for sanity checks
PYTHONPATH=. python -m aws_trainer.run \
  --json mol_to_spin_system/data/spin_systems_chembl.json \
  --model-size medium --no-stage2 --epochs 80

# Full training: Stage 1 (matrix loss) then Stage 2 (+ spectral consistency)
PYTHONPATH=. python -m aws_trainer.run \
  --json mol_to_spin_system/data/spin_systems_chembl.json \
  --model-size medium --epochs 110 --stage1-epochs 70
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--json` | `spin_systems.json` | Spin-system dataset |
| `--model-size` | `medium` | `tiny` / `small` / `medium` / `large` / `*-attn` |
| `--epochs` | 100 | Total training epochs |
| `--stage1-epochs` | 70 | Epochs of matrix-only loss before spectral loss ramps in |
| `--no-stage2` | — | Disable spectral consistency loss entirely |
| `--batch` | 256 | Batch size |
| `--amp` | `bf16` | Mixed precision: `bf16` / `fp16` / `none` |
| `--no-scaffold` | — | Skip RDKit scaffold split (faster) |
| `--ckpt-dir` | `aws_trainer/checkpoints` | Checkpoint output directory |

Checkpoints are saved to `best.pt` (best val score) and `last.pt` (most recent epoch). Each checkpoint includes model weights, EMA weights, standardizer statistics, and full config.

### EC2 training (recommended for the full 60k dataset)

**One-time: pack and upload spectra to S3:**

```bash
aws sso login --profile hack-scripps
aws s3 mb s3://spinhance-data --profile hack-scripps --region us-west-2 2>/dev/null || true

# Pack 90MHz spectra into a single tar.gz (much faster to upload than 64k individual files)
tar czf simulation/data/spectra/90MHz/mol_all.tar.gz \
  -C simulation/data/spectra/90MHz --exclude='*.tar.gz' .

aws s3 cp simulation/data/spectra/90MHz/mol_all.tar.gz \
  s3://spinhance-data/spectra/90MHz/mol_all.tar.gz \
  --profile hack-scripps --region us-west-2 --no-progress
```

**Launch a spot GPU instance and start training:**

```bash
# Stage 1+2, medium model, g4dn.xlarge (T4, ~$0.16/hr spot)
bash aws_trainer/ec2/launch.sh g4dn.xlarge aws_trainer/configs/medium_s2.json

# Stage 1 only
bash aws_trainer/ec2/launch.sh g4dn.xlarge aws_trainer/configs/medium_s1.json
```

The script auto-selects the latest Deep Learning AMI, bootstraps the environment, syncs spectra from S3, and starts training in the background (logs at `/tmp/train.log` on the instance).

**Monitor and retrieve checkpoints:**

```bash
# Print last 30 lines of training log and sync checkpoints to S3
bash aws_trainer/ec2/sync.sh <instance-id>

# Also pull best.pt to your local machine
bash aws_trainer/ec2/sync.sh <instance-id> spinhance-data --local

# Terminate when done
aws ec2 terminate-instances --instance-ids <instance-id> \
  --profile hack-scripps --region us-west-2
```

Available configs:

| Config | Epochs | Stage 2 | Notes |
|---|---|---|---|
| `aws_trainer/configs/medium_s2.json` | 110 (70+40) | yes | recommended full run |
| `aws_trainer/configs/medium_s1.json` | 80 | no | matrix loss only, faster |

---

## Stage 5 — Evaluate (`model/gui.py`)

Interactive Streamlit dashboard for inspecting predictions on individual molecules and computing aggregate metrics over a batch.

```bash
streamlit run model/gui.py
```

On first load, enter the checkpoint path (e.g. `aws_trainer/checkpoints/best.pt`) and spectra directory (`simulation/data/spectra`).

- **Single Molecule Inspector** — pick a molecule, run inference, compare predicted vs ground-truth shift+J+degeneracy matrix, overlay simulated spectra.
- **Batch Evaluation** — aggregate shift MAE, J MAE, and degeneracy accuracy over N molecules; scatter plots of predicted vs ground truth.

---

## Checkpoint format

```python
import torch
from model.targets import Standardizer

ckpt = torch.load("aws_trainer/checkpoints/best.pt", weights_only=False)
# Keys: "model", "ema_model", "standardizer", "cfg", "epoch", "metrics"

model.load_state_dict(ckpt["ema_model"])   # EMA weights are better for inference
std = Standardizer()
vars(std).update(ckpt["standardizer"])
```

---

## Key paths

| Path | Contents |
|---|---|
| `mol_to_spin_system/data/spin_systems_chembl.json` | 64k molecule spin-system dataset |
| `simulation/data/spectra/90MHz/mol_XXXXXX.npy` | Simulated 90 MHz spectra |
| `simulation/data/spectra/600MHz/mol_XXXXXX.npy` | Simulated 600 MHz spectra |
| `aws_trainer/checkpoints/best.pt` | Best model checkpoint |
| `aws_trainer/configs/medium_s2.json` | Full Stage 1+2 training config |
| `model/gui.py` | Evaluation dashboard |

## Training diagnostics

Current model training writes a canonical diagnostics directory under `model/runs/<run_id>/`. This directory is the source of truth for live dashboards, probe diagnostics, failure analysis, and AutoAI integration.

Smoke run:

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

Live dashboard:

~~~bash
PYTHONPATH=. streamlit run model/live_dashboard.py
~~~

See `docs/training_diagnostics.md` for the full artifact contract used by collaborators and AutoAI agents.
