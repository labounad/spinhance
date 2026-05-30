#!/usr/bin/env bash
# aws_trainer/ec2/sync.sh — Copy checkpoints + logs from EC2 to S3 (and locally).
#
# Usage:
#   bash aws_trainer/ec2/sync.sh INSTANCE_ID [S3_BUCKET] [--local]
#
# With --local: also rsync checkpoints to ./aws_trainer/checkpoints/

set -euo pipefail
INSTANCE_ID="${1:?Usage: sync.sh INSTANCE_ID [S3_BUCKET] [--local]}"
S3_BUCKET="${2:-spinhance-data}"
LOCAL_FLAG="${3:-}"

PROFILE=$(cat /tmp/vaws_instance_profile 2>/dev/null || echo "hack-scripps")
REGION=$(cat /tmp/vaws_instance_region 2>/dev/null || echo "us-west-2")
EICE_KEY="/tmp/vaws-eice-key"
REMOTE_CKPT="/home/ec2-user/spinhance/aws_trainer/checkpoints"

_push_key() {
  aws ec2-instance-connect send-ssh-public-key \
    --profile "$PROFILE" --region "$REGION" \
    --instance-id "$INSTANCE_ID" \
    --instance-os-user ec2-user \
    --ssh-public-key "file://$EICE_KEY.pub"
}

_ssh() { _push_key; ssh -i "$EICE_KEY" -o StrictHostKeyChecking=no \
  -o "ProxyCommand=aws ec2-instance-connect open-tunnel --instance-id $INSTANCE_ID --profile $PROFILE --region $REGION" \
  "ec2-user@$INSTANCE_ID" "$@"; }

# ── Training status ───────────────────────────────────────────────────────────
echo "=== aws_trainer sync ==="
echo "Training log (last 30 lines):"
_ssh "tail -30 /tmp/train.log 2>/dev/null || echo '(no log yet)'"
echo ""

# ── Sync checkpoints EC2 → S3 ─────────────────────────────────────────────────
echo "Syncing checkpoints to s3://$S3_BUCKET/checkpoints/..."
_ssh "aws s3 sync $REMOTE_CKPT s3://$S3_BUCKET/checkpoints --no-progress" && \
  echo "  done" || echo "  S3 sync failed (check IAM permissions)"

# ── Optional: pull checkpoints locally ───────────────────────────────────────
if [ "$LOCAL_FLAG" = "--local" ]; then
  echo "Pulling checkpoints locally..."
  _push_key
  REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  mkdir -p "$REPO_ROOT/aws_trainer/checkpoints"
  scp -i "$EICE_KEY" -o StrictHostKeyChecking=no \
    -o "ProxyCommand=aws ec2-instance-connect open-tunnel --instance-id $INSTANCE_ID --profile $PROFILE --region $REGION" \
    -r "ec2-user@$INSTANCE_ID:$REMOTE_CKPT/" \
    "$REPO_ROOT/aws_trainer/checkpoints/"
  echo "  checkpoints at aws_trainer/checkpoints/"
fi
