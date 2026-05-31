#!/usr/bin/env bash
# train.sh — Download data from S3 and train SpinHance. Run ON EC2.
#
# Usage:
#   bash ~/spinhance/model/train.sh
#
# To run in background:
#   nohup bash ~/spinhance/model/train.sh > ~/train.log 2>&1 &

set -euo pipefail

BUCKET="spinhance-data"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
JSON_DST="$REPO/mol_to_spin_system/data/spin_systems_chembl.json"
SPECTRA="$REPO/simulation/data/spectra/90MHz"

# ── Python command ────────────────────────────────────────────────────────────
if command -v python &>/dev/null && python -c "import torch" 2>/dev/null; then
  PYTHON="python"
elif [ -f "$HOME/.local/bin/micromamba" ]; then
  PYTHON="$HOME/.local/bin/micromamba run -n spinhance python"
elif [ -f /opt/conda/bin/mamba ]; then
  PYTHON="/opt/conda/bin/mamba run -n spinhance python"
elif [ -f /opt/conda/bin/conda ]; then
  PYTHON="/opt/conda/bin/conda run -n spinhance python"
else
  echo "ERROR: no python with torch found. Run: bash ~/spinhance/model/setup.sh"
  exit 1
fi

# ── Session number (auto-increment from S3) ───────────────────────────────────
LAST=$(aws s3 ls "s3://$BUCKET/training/" 2>/dev/null \
  | grep -oE 'session[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 || true)
SESSION=$(printf "%03d" $(( ${LAST:-0} + 1 )))
S3_PREFIX="s3://$BUCKET/training/session$SESSION"
echo "=== SpinHance training — session $SESSION ==="
echo "  checkpoints → $S3_PREFIX"
echo ""

# ── Download data (skip if already present) ───────────────────────────────────
if [ ! -f "$JSON_DST" ]; then
  echo "Downloading spin_systems_chembl.json..."
  mkdir -p "$(dirname "$JSON_DST")"
  aws s3 cp "s3://$BUCKET/spin_systems_chembl.json" "$JSON_DST"
else
  echo "spin_systems_chembl.json already present — skipping download."
fi

if [ ! -f "$SPECTRA/mol_000000.npy" ]; then
  echo "Downloading and extracting 90MHz spectra..."
  mkdir -p "$SPECTRA"
  aws s3 cp "s3://$BUCKET/spectra/90MHz/mol_all.tar.gz" /tmp/mol_all.tar.gz
  tar xzf /tmp/mol_all.tar.gz -C "$SPECTRA"
  rm /tmp/mol_all.tar.gz
  echo "Extraction complete."
else
  echo "Spectra already present — skipping download."
fi

# ── Train ─────────────────────────────────────────────────────────────────────
LOG="$HOME/train_session${SESSION}.log"
TRAIN_CMD="cd $REPO && PYTHONPATH=. $PYTHON -m model.run_experiment \
  --json mol_to_spin_system/data/spin_systems_chembl.json \
  --spectra simulation/data/spectra \
  --fields 90 \
  --stage2 \
  --epochs 10 \
  --stage1-epochs 2 \
  --ramp-epochs 5 \
  --batch 256 \
  --s3-ckpt-prefix $S3_PREFIX 2>&1 | tee $LOG"

echo "Launching training (logs → $LOG)..."

if command -v tmux &>/dev/null; then
  TNAME="spinhance-$SESSION"
  tmux new-session -d -s "$TNAME" "bash -c '$TRAIN_CMD'"
  echo ""
  echo "=== Training running in tmux session '$TNAME' ==="
  echo "  Reattach : tmux attach -t $TNAME"
  echo "  Tail logs: tail -f $LOG"
elif command -v screen &>/dev/null; then
  screen -dmS "spinhance-$SESSION" bash -c "$TRAIN_CMD"
  echo ""
  echo "=== Training running in screen session 'spinhance-$SESSION' ==="
  echo "  Reattach : screen -r spinhance-$SESSION"
  echo "  Tail logs: tail -f $LOG"
else
  nohup bash -c "$TRAIN_CMD" &
  disown
  echo ""
  echo "=== Training running with nohup (tmux/screen not found) ==="
  echo "  Tail logs: tail -f $LOG"
fi

# ── Diagnostics sync sidecar ──────────────────────────────────────────────────
# Syncs model/runs/ JSONL artifacts (not checkpoints) to S3 every 30s so the
# live dashboard (model/live_dashboard.py) can read them locally after an
# `aws s3 sync` on your machine.
nohup bash -c "
  while true; do
    aws s3 sync \"$REPO/model/runs\" \"s3://$BUCKET/model/runs\" \
      --exclude '*.pt' --no-progress 2>>/tmp/sync.log
    sleep 30
  done
" >> /tmp/sync.log 2>&1 &
disown
echo "  Diagnostics sync → s3://$BUCKET/model/runs (every 30s, log: /tmp/sync.log)"
