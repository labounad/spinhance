#!/usr/bin/env bash
# aws_trainer/ec2/launch.sh — Launch a GPU spot instance and start training.
#
# Usage:
#   bash aws_trainer/ec2/launch.sh [INSTANCE_TYPE] [CONFIG_JSON]
#
# Examples:
#   bash aws_trainer/ec2/launch.sh g4dn.xlarge
#   bash aws_trainer/ec2/launch.sh p3.2xlarge aws_trainer/configs/large_s2.json
#
# Prerequisites:
#   aws sso login --profile hack-scripps   # fresh token
#   S3_BUCKET set below or via env var

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROFILE="${AWS_PROFILE:-hack-scripps}"
REGION="${AWS_REGION:-us-west-2}"
INSTANCE_TYPE="${1:-g4dn.xlarge}"          # g4dn.xlarge (T4 16 GB) or p3.2xlarge (V100 16 GB)
TRAIN_CONFIG="${2:-}"                       # optional path to VAWSConfig JSON

# Reuse the same networking setup as autoai
AMI_ID=$(aws ec2 describe-images --profile "$PROFILE" --region "$REGION" \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Amazon Linux 2023*" \
            "Name=architecture,Values=x86_64" "Name=state,Values=available" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text 2>/dev/null || echo "")
[ -z "$AMI_ID" ] && { echo "ERROR: Could not find Deep Learning AMI in $REGION"; exit 1; }
SUBNET_ID="subnet-0096ffc9c05bebab3"
SECURITY_GROUP="sg-09d5ef7889a26f56a"
INSTANCE_PROFILE="hackathon-ec2-profile"
S3_BUCKET="${S3_BUCKET:-spinhance-data}"   # set this or export S3_BUCKET=...

EICE_KEY="/tmp/vaws-eice-key"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE_WORKSPACE="/home/ec2-user/spinhance"

echo "=== aws_trainer EC2 launch ==="
echo "  instance type : $INSTANCE_TYPE"
echo "  region        : $REGION"
echo "  profile       : $PROFILE"
echo "  s3 bucket     : $S3_BUCKET"

# ── Spot instance request ─────────────────────────────────────────────────────
echo "[1/5] Requesting spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --profile "$PROFILE" --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --min-count 1 --max-count 1 \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SECURITY_GROUP" \
  --iam-instance-profile "Name=$INSTANCE_PROFILE" \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
  --metadata-options '{"HttpTokens":"required"}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spinhance-vaws}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "  instance: $INSTANCE_ID — waiting for running state..."
aws ec2 wait instance-running \
  --profile "$PROFILE" --region "$REGION" \
  --instance-ids "$INSTANCE_ID"
echo "  running. Waiting 40s for SSH daemon..."
sleep 40

# ── EICE key ──────────────────────────────────────────────────────────────────
rm -f "$EICE_KEY" "$EICE_KEY.pub"
ssh-keygen -t ed25519 -N "" -f "$EICE_KEY" -q <<< "y" 2>/dev/null || true

_push_key() {
  aws ec2-instance-connect send-ssh-public-key \
    --profile "$PROFILE" --region "$REGION" \
    --instance-id "$INSTANCE_ID" \
    --instance-os-user ec2-user \
    --ssh-public-key "file://$EICE_KEY.pub"
}

_ssh() {
  _push_key
  ssh -i "$EICE_KEY" \
    -o StrictHostKeyChecking=no \
    -o "ProxyCommand=aws ec2-instance-connect open-tunnel --instance-id $INSTANCE_ID --profile $PROFILE --region $REGION" \
    "ec2-user@$INSTANCE_ID" "$@"
}

_scp() {
  _push_key
  scp -i "$EICE_KEY" \
    -o StrictHostKeyChecking=no \
    -o "ProxyCommand=aws ec2-instance-connect open-tunnel --instance-id $INSTANCE_ID --profile $PROFILE --region $REGION" \
    "$@"
}

# ── Bootstrap ─────────────────────────────────────────────────────────────────
echo "[2/5] Bootstrapping environment..."
_scp "$REPO_ROOT/aws_trainer/ec2/setup.sh" "ec2-user@$INSTANCE_ID:/tmp/setup.sh"
_ssh "bash /tmp/setup.sh $S3_BUCKET" 2>&1 | tail -20

# ── Sync code + data ─────────────────────────────────────────────────────────
echo "[3/5] Syncing code and data..."
# Sync repo (excluding large data dirs already on S3)
_scp -r "$REPO_ROOT" "ec2-user@$INSTANCE_ID:/tmp/spinhance_src" 2>/dev/null || true
_ssh "rsync -a --exclude='simulation/data' --exclude='.git' --exclude='__pycache__' \
  /tmp/spinhance_src/ $REMOTE_WORKSPACE/"

# Download 90MHz spectra tarball from S3 (training uses 90MHz only)
_ssh "mkdir -p $REMOTE_WORKSPACE/simulation/data/spectra/90MHz && \
  aws s3 cp s3://$S3_BUCKET/spectra/90MHz/mol_all.tar.gz \
    $REMOTE_WORKSPACE/simulation/data/spectra/90MHz/mol_all.tar.gz --no-progress || \
  echo 'WARNING: S3 download failed — check bucket and permissions'"

# ── Launch training ───────────────────────────────────────────────────────────
echo "[4/5] Starting training (nohup, logs at /tmp/train.log)..."
N_GPUS=$(_ssh "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1")
echo "  GPUs detected: $N_GPUS"

CONFIG_ARG=""
if [ -n "$TRAIN_CONFIG" ]; then
  _scp "$TRAIN_CONFIG" "ec2-user@$INSTANCE_ID:$REMOTE_WORKSPACE/active_config.json"
  CONFIG_ARG="--config $REMOTE_WORKSPACE/active_config.json"
fi

TRAIN_CMD="cd $REMOTE_WORKSPACE && conda run -n spinhance \
  torchrun --nproc_per_node=$N_GPUS -m aws_trainer.run $CONFIG_ARG \
  --model-size medium --ckpt-dir $REMOTE_WORKSPACE/aws_trainer/checkpoints \
  > /tmp/train.log 2>&1"

_ssh "nohup bash -c '$TRAIN_CMD' &"
echo "  Training launched. Monitor with:"
echo "    bash aws_trainer/ec2/monitor.sh $INSTANCE_ID $PROFILE $REGION"

# ── Save instance ID ──────────────────────────────────────────────────────────
echo "[5/5] Saving instance info..."
echo "$INSTANCE_ID" > /tmp/vaws_instance_id
echo "$PROFILE"     > /tmp/vaws_instance_profile
echo "$REGION"      > /tmp/vaws_instance_region

cat <<EOF

=== Training started ===
Instance ID : $INSTANCE_ID
Logs        : tail -f /tmp/train.log  (via SSH)
Checkpoints : sync'd to S3 when training completes

To terminate when done:
  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --profile $PROFILE --region $REGION

To sync checkpoints now:
  bash aws_trainer/ec2/sync.sh $INSTANCE_ID $S3_BUCKET
EOF
