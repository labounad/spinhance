#!/usr/bin/env bash
# aws_trainer/ec2/setup.sh — Run on EC2 to install dependencies.
# Called by launch.sh; can also be run manually after ssh.
#
# Usage:  bash setup.sh [S3_BUCKET]

set -euo pipefail
S3_BUCKET="${1:-spinhance-data}"
WORKSPACE="/home/ec2-user/spinhance"
mkdir -p "$WORKSPACE"

echo "[setup] Installing system packages..."
sudo yum install -y git wget rsync gcc gcc-c++ make 2>/dev/null || \
sudo apt-get install -y git wget rsync gcc g++ make 2>/dev/null || true

# ── Miniconda ────────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "[setup] Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  echo "export PATH=$HOME/miniconda3/bin:\$PATH" >> ~/.bashrc
  source ~/.bashrc
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda config --set always_yes true

# ── spinhance conda environment ───────────────────────────────────────────────
if ! conda info --envs | grep -q spinhance; then
  echo "[setup] Creating conda env 'spinhance'..."
  conda create -n spinhance python=3.11 -q   # 3.11 for broader CUDA wheel availability
  conda activate spinhance

  # GPU-aware PyTorch (auto-detects CUDA from driver)
  CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | \
             head -1 | cut -d. -f1 || echo "cpu")
  if [ "$CUDA_VER" -ge 525 ] 2>/dev/null; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
  else
    pip install torch torchvision -q
  fi

  # Project dependencies
  pip install numpy scipy pandas matplotlib scikit-learn tqdm rich rdkit lxml \
              streamlit plotly wandb -q
  pip install nmrglue -q || true   # optional
else
  conda activate spinhance
  echo "[setup] conda env 'spinhance' already exists — skipping install"
fi

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "[setup] GPU check:"
conda run -n spinhance python -c "
import torch
print(f'  torch {torch.__version__}  CUDA={torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ── S3: check data ────────────────────────────────────────────────────────────
echo "[setup] Checking S3 data sync (s3://$S3_BUCKET/spectra)..."
aws s3 ls "s3://$S3_BUCKET/spectra/" --no-sign-request 2>/dev/null && \
  echo "  S3 bucket accessible" || \
  echo "  WARNING: s3://$S3_BUCKET not accessible — data must be synced manually"

echo "[setup] Done."
