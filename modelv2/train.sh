#!/usr/bin/env bash
# modelv2/train.sh — Download data from S3 and launch modelv2 training. Run ON EC2.
#
# Usage:  bash ~/spinhance/modelv2/train.sh
# Background: nohup bash ~/spinhance/modelv2/train.sh > ~/train_launch.log 2>&1 &

set -euo pipefail

BUCKET="spinhance-data"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python}"

SPIN_SYSTEMS="$REPO/mol_to_spin_system/data/buckets/spin_systems_chembl_8spin.json.gz"
SPECTRA_TGZ="$REPO/simulation/data/spectra/90MHz.tar.gz"

# ── Sanity-check CUDA ─────────────────────────────────────────────────────────
if ! "$PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  CUDA_VER=$("$PYTHON" -c "import torch; print(torch.version.cuda or 'None')" 2>/dev/null \
             || echo "torch not found")
  echo "ERROR: CUDA not available (torch.version.cuda=$CUDA_VER)"
  echo "       Fix: micromamba install -n spinhance pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y"
  exit 1
fi

# ── Auto-increment session number from S3 ─────────────────────────────────────
LAST=$(aws s3 ls "s3://$BUCKET/training/" 2>/dev/null \
  | grep -oE 'session[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 || true)
SESSION=$(printf "%03d" $(( 10#${LAST:-0} + 1 )))
S3_OUT="s3://$BUCKET/training/session$SESSION"

echo "=== SpinHance training — session $SESSION → $S3_OUT ==="

# ── Download data if missing ──────────────────────────────────────────────────
if [ ! -f "$SPIN_SYSTEMS" ]; then
  echo "Downloading spin_systems_chembl_8spin.json.gz..."
  mkdir -p "$(dirname "$SPIN_SYSTEMS")"
  aws s3 cp "s3://$BUCKET/mol_to_spin_system/data/buckets/spin_systems_chembl_8spin.json.gz" \
    "$SPIN_SYSTEMS"
fi

if [ ! -f "$SPECTRA_TGZ" ]; then
  echo "Downloading 90MHz.tar.gz..."
  mkdir -p "$(dirname "$SPECTRA_TGZ")"
  aws s3 cp "s3://$BUCKET/simulation/data/spectra/90MHz.tar.gz" "$SPECTRA_TGZ"
fi

# ── Write training script to a real file (avoids tmux quoting issues) ─────────
LOG="$HOME/train_session${SESSION}.log"
SCRIPT=$(mktemp /tmp/spinhance_train_XXXXX.sh)

cat > "$SCRIPT" << HEREDOC
#!/usr/bin/env bash
cd "$REPO"
export PYTHONPATH="$REPO"

echo "[train] ======================================================"
echo "[train] session  : $SESSION"
echo "[train] out      : $S3_OUT"
echo "[train] log      : $LOG"
echo "[train] started  : \$(date -u)"
echo "[train] ======================================================"

"$PYTHON" -m modelv2.train \
  --spin_systems="$SPIN_SYSTEMS" \
  --spectra="$SPECTRA_TGZ" \
  --out="$S3_OUT" \
  --no_scaffold \
  2>&1 | tee "$LOG"

RC=\${PIPESTATUS[0]}
echo ""
echo "[train] ======================================================"
echo "[train] finished : \$(date -u)  exit=\$RC"
echo "[train] ======================================================"
# Keep the tmux pane alive so you can inspect on attach
exec bash -i
HEREDOC

chmod +x "$SCRIPT"

# ── Launch ────────────────────────────────────────────────────────────────────
TNAME="spinhance-$SESSION"

if command -v tmux &>/dev/null; then
  tmux new-session -d -s "$TNAME" "bash '$SCRIPT'"
  echo "  tmux : tmux attach -t $TNAME"
elif command -v screen &>/dev/null; then
  screen -dmS "$TNAME" bash "$SCRIPT"
  echo "  screen : screen -r $TNAME"
else
  nohup bash "$SCRIPT" &
  disown
  echo "  (no tmux/screen — running in background)"
fi
echo "  logs : tail -f $LOG"
