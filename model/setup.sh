#!/usr/bin/env bash
# setup.sh — Bootstrap micromamba + spinhance env on a fresh EC2.
# Run once after cloning the repo.
#
# Usage:
#   bash ~/spinhance/model/setup.sh

set -euo pipefail

# ── 1. Micromamba ─────────────────────────────────────────────────────────────
if ! command -v micromamba &>/dev/null; then
  echo "[1/3] Installing micromamba..."
  curl -Ls https://micro.mamba.pm/install.sh | bash
  export PATH="$HOME/.local/bin:$PATH"
  source "$HOME/.bashrc" 2>/dev/null || true
else
  echo "[1/3] micromamba already installed — skipping."
fi

MAMBA="$HOME/.local/bin/micromamba"

# ── 2. Detect CUDA ────────────────────────────────────────────────────────────
CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" | head -1 || echo "0")
if [ "$CUDA_MAJOR" -ge 12 ]; then
  CUDA_PKG="pytorch-cuda=12.1"
elif [ "$CUDA_MAJOR" -ge 11 ]; then
  CUDA_PKG="pytorch-cuda=11.8"
else
  CUDA_PKG=""
  echo "WARNING: no GPU detected — installing CPU-only PyTorch."
fi
echo "[2/3] CUDA $CUDA_MAJOR detected → $CUDA_PKG"

# ── 3. Create env ─────────────────────────────────────────────────────────────
echo "[3/3] Creating spinhance environment..."
$MAMBA create -n spinhance -y \
  -c pytorch -c nvidia -c conda-forge \
  python=3.11 pytorch $CUDA_PKG numpy scipy lxml

echo ""
echo "=== Setup complete ==="
echo "  Train: bash ~/spinhance/model/train.sh"
