#!/usr/bin/env bash
# train.sh — Download data from S3 and launch modelv2 training. Run ON EC2.
#
# Usage:  bash ~/spinhance/modelv2/train.sh

set -euo pipefail

BUCKET="spinhance-data"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

SPIN_SYSTEMS="$REPO/mol_to_spin_system/data/buckets/spin_systems_chembl_8spin.json.gz"
SPECTRA="$REPO/simulation/data/spectra/90MHz.tar.gz"

# ── Auto-increment session number from S3 ─────────────────────────────────────
LAST=$(aws s3 ls "s3://$BUCKET/training/" 2>/dev/null \
  | grep -oE 'session[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 || true)
SESSION=$(printf "%03d" $(( 10#${LAST:-0} + 1 )))
export SPINHANCE_OUT="s3://$BUCKET/training/session$SESSION"
echo "=== SpinHance training — session $SESSION → $SPINHANCE_OUT ==="

# ── Download data (skip if already present) ───────────────────────────────────
if [ ! -f "$SPIN_SYSTEMS" ]; then
  echo "Downloading spin_systems_chembl_8spin.json.gz..."
  aws s3 cp "s3://$BUCKET/mol_to_spin_system/data/buckets/spin_systems_chembl_8spin.json.gz" "$SPIN_SYSTEMS"
fi

if [ ! -f "$SPECTRA" ]; then
  echo "Downloading 90MHz.tar.gz..."
  aws s3 cp "s3://$BUCKET/simulation/data/spectra/90MHz.tar.gz" "$SPECTRA"
fi

# ── Launch training ───────────────────────────────────────────────────────────
LOG="$HOME/train_session${SESSION}.log"
TRAIN_CMD="cd $REPO && SPINHANCE_OUT=$SPINHANCE_OUT \
PYTHONPATH=. python -m modelv2.train \
  --spin_systems=$SPIN_SYSTEMS \
  --spectra=$SPECTRA 2>&1 | tee $LOG"

if command -v tmux &>/dev/null; then
  TNAME="spinhance-$SESSION"
  tmux new-session -d -s "$TNAME" "bash -c '$TRAIN_CMD'"
  echo "  tmux:  tmux attach -t $TNAME"
else
  nohup bash -c "$TRAIN_CMD" &
  disown
fi
echo "  logs:  tail -f $LOG"
