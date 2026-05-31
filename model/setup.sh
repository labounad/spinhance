#!/usr/bin/env bash
# setup.sh — Bootstrap micromamba + spinhance env on a fresh EC2.
#
# Usage:
#   bash ~/spinhance/model/setup.sh

set -e

pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; exit 1; }

# ── 1. Micromamba ─────────────────────────────────────────────────────────────
echo "[1/3] Checking micromamba..."

# Install if not found in any known location
if ! command -v micromamba &>/dev/null \
    && [ ! -f "$HOME/micromamba/bin/micromamba" ] \
    && [ ! -f "$HOME/.local/bin/micromamba" ]; then
  echo "  Installing micromamba..."
  curl -Ls https://micro.mamba.pm/install.sh | bash
fi

# Find the binary — check every known install location
if   [ -f "$HOME/micromamba/bin/micromamba" ]; then MAMBA="$HOME/micromamba/bin/micromamba"
elif [ -f "$HOME/.local/bin/micromamba" ];      then MAMBA="$HOME/.local/bin/micromamba"
elif command -v micromamba &>/dev/null;           then MAMBA="$(command -v micromamba)"
else
  fail "micromamba not found after install. Open a new shell and retry."
fi

"$MAMBA" --version &>/dev/null || fail "micromamba binary at $MAMBA is not executable."
pass "micromamba $("$MAMBA" --version)  ($MAMBA)"

# ── 2. Detect CUDA ────────────────────────────────────────────────────────────
echo "[2/3] Detecting CUDA..."
CUDA_MAJOR=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+" | head -1 || echo "0")

if   [ "$CUDA_MAJOR" -ge 12 ]; then
  CUDA_PKG="pytorch-cuda=12.4"
  EXTRA_CHANNELS="-c nvidia"
elif [ "$CUDA_MAJOR" -ge 11 ]; then
  CUDA_PKG="pytorch-cuda=11.8"
  EXTRA_CHANNELS="-c nvidia"
else
  CUDA_PKG=""
  EXTRA_CHANNELS=""
fi
pass "CUDA driver version $CUDA_MAJOR → ${CUDA_PKG:-cpu-only pytorch}"

# ── 3. Git LFS ────────────────────────────────────────────────────────────────
echo "[3/4] Installing git-lfs..."
if ! command -v git-lfs &>/dev/null; then
  if command -v dnf &>/dev/null; then
    sudo dnf install -y git-lfs
  elif command -v apt-get &>/dev/null; then
    sudo apt-get install -y git-lfs
  else
    fail "No package manager found to install git-lfs."
  fi
fi
git lfs install || fail "git lfs install failed."
pass "git-lfs $(git-lfs --version)"

# ── 4. Create env ─────────────────────────────────────────────────────────────
echo "[4/4] Creating spinhance environment..."

# Remove existing env cleanly so we always get a fresh, known-good state
if "$MAMBA" env list 2>/dev/null | grep -q "spinhance"; then
  echo "  Removing existing spinhance env..."
  "$MAMBA" env remove -n spinhance -y
fi

# shellcheck disable=SC2086
"$MAMBA" create -n spinhance -y \
  -c pytorch $EXTRA_CHANNELS -c conda-forge \
  python=3.11 pytorch ${CUDA_PKG:+$CUDA_PKG} numpy scipy lxml \
  || fail "Environment creation failed."

pass "Environment created."

# ── Tests ─────────────────────────────────────────────────────────────────────
echo ""
echo "Running tests..."

# Test: python importable
"$MAMBA" run -n spinhance python -c "import sys; print(f'  python {sys.version.split()[0]}')" \
  || fail "python not working in spinhance env."
pass "python"

# Test: torch importable and correct version
"$MAMBA" run -n spinhance python -c "import torch; print(f'  torch {torch.__version__}')" \
  || fail "torch not importable."
pass "torch"

# Test: CUDA available if GPU was detected
if [ "$CUDA_MAJOR" -ge 11 ]; then
  "$MAMBA" run -n spinhance python -c "
import torch, sys
if not torch.cuda.is_available():
    print('  CUDA not available — check driver/toolkit versions')
    sys.exit(1)
print(f'  CUDA available — {torch.cuda.get_device_name(0)}')
" || fail "CUDA not available despite GPU driver being present."
  pass "CUDA"
fi

# Test: numpy and scipy
"$MAMBA" run -n spinhance python -c "import numpy, scipy" \
  || fail "numpy/scipy not importable."
pass "numpy + scipy"

# Test: git-lfs
command -v git-lfs &>/dev/null || fail "git-lfs not found."
git lfs env &>/dev/null        || fail "git lfs not initialized."
pass "git-lfs"

# Test: aws CLI available (needed by train.sh)
command -v aws &>/dev/null || fail "aws CLI not found — install it before running train.sh."
aws --version &>/dev/null || fail "aws CLI not working."
pass "aws CLI"

# Test: tmux or screen available (needed for session persistence in train.sh)
if command -v tmux &>/dev/null; then
  pass "tmux (session persistence)"
elif command -v screen &>/dev/null; then
  pass "screen (session persistence)"
else
  echo "  [WARN] Neither tmux nor screen found — training will use nohup (no reattach)."
  echo "         Install tmux: sudo yum install -y tmux   OR   sudo apt install -y tmux"
fi

# ── Pull LFS files ────────────────────────────────────────────────────────────
echo ""
echo "Pulling git-lfs files..."
REPO="$(cd "$(dirname "$0")/.." && pwd)"
git -C "$REPO" lfs pull || fail "git lfs pull failed — check your git credentials."
pass "git lfs pull"

echo ""
echo "=== All tests passed. Run: bash ~/spinhance/model/train.sh ==="
