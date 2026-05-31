#!/usr/bin/env bash
# train_ec2.sh — Launch a GPU spot instance and train SpinHance.
#
# Usage:
#   bash model/train_ec2.sh [INSTANCE_TYPE] [run_experiment args...]
#
# Examples:
#   bash model/train_ec2.sh
#   bash model/train_ec2.sh g4dn.xlarge
#   bash model/train_ec2.sh p3.2xlarge --epochs 110 --stage2 --stage1-epochs 70 --batch 512
#
# Prerequisites:
#   aws sso login --profile hack-scripps

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROFILE="${AWS_PROFILE:-hack-scripps}"
REGION="${AWS_REGION:-us-west-2}"
INSTANCE_TYPE="${1:-g5.xlarge}"     # A10G 24 GB; also try g6.xlarge (L4) or g6e.xlarge (L40S)
shift 2>/dev/null || true

# Default training args — override by passing after instance type
TRAIN_ARGS="${*:- \
  --fields 90 \
  --stage2 \
  --epochs 110 \
  --stage1-epochs 70 \
  --ramp-epochs 10 \
  --batch 256 \
}"

BUCKET="spinhance-data"
SUBNET="subnet-0096ffc9c05bebab3"
SG="sg-09d5ef7889a26f56a"
INST_PROFILE="hackathon-ec2-profile"
EICE_KEY="/tmp/spinhance-eice-key"
SSH_CFG="/tmp/spinhance-ssh.cfg"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="/home/ec2-user/spinhance"

echo "=== SpinHance EC2 training ==="
echo "  instance : $INSTANCE_TYPE"
echo "  args     : $TRAIN_ARGS"
echo ""

# ── 0. Session number (auto-increment from S3) ───────────────────────────────
LAST=$(aws s3 ls "s3://$BUCKET/training/" --profile "$PROFILE" --region "$REGION" 2>/dev/null \
  | grep -oE 'session([0-9]+)' | grep -oE '[0-9]+' | sort -n | tail -1)
SESSION=$(printf "%03d" $(( ${LAST:-0} + 1 )))
S3_CKPT_PREFIX="s3://$BUCKET/training/session$SESSION"
echo "  session  : $SESSION  ($S3_CKPT_PREFIX)"

# ── Reuse existing instance if available ──────────────────────────────────────
if [ -f /tmp/spinhance_instance_id ]; then
  EXISTING=$(cat /tmp/spinhance_instance_id)
  STATE=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" \
    --instance-ids "$EXISTING" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "gone")
  if [ "$STATE" = "running" ]; then
    echo "  reusing existing instance $EXISTING"
    INSTANCE="$EXISTING"
    REUSING=true
  else
    echo "  previous instance $EXISTING is $STATE — launching new one"
    INSTANCE=""
  fi
fi

# ── 1. Find latest Deep Learning AMI ─────────────────────────────────────────
if [ -n "${INSTANCE:-}" ]; then
  echo "[1/5] Skipping AMI lookup (reusing instance)"
  echo "[2/5] Skipping launch (reusing instance)"
else
echo "[1/5] Finding latest Deep Learning AMI..."
AMI=$(aws ec2 describe-images \
  --profile "$PROFILE" --region "$REGION" \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Amazon Linux 2023*" \
    "Name=architecture,Values=x86_64" \
    "Name=state,Values=available" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" \
  --output text)
[ -z "$AMI" ] && { echo "ERROR: no Deep Learning AMI found in $REGION"; exit 1; }
echo "  AMI: $AMI"

# ── 2. Launch spot instance (try all subnets in the VPC until one has capacity)
echo "[2/5] Launching spot instance..."
VPC_ID=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" \
  --subnet-ids "$SUBNET" --query 'Subnets[0].VpcId' --output text)
ALL_SUBNETS=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=state,Values=available" \
  --query 'Subnets[*].SubnetId' --output text | tr '\t' ' ')
# preferred subnet first, then the rest
SUBNETS_TO_TRY="$SUBNET $(echo "$ALL_SUBNETS" | sed "s/$SUBNET//g")"

_launch() {
  local subnet=$1 market=$2
  local opts=""
  [ "$market" = "spot" ] && opts='--instance-market-options {"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}'
  aws ec2 run-instances \
    --profile "$PROFILE" --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --subnet-id "$subnet" \
    --security-group-ids "$SG" \
    --iam-instance-profile "Name=$INST_PROFILE" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --metadata-options '{"HttpTokens":"required"}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spinhance-train}]' \
    --query 'Instances[0].InstanceId' \
    --output text $opts 2>/dev/null
}

INSTANCE=""
for try_subnet in $SUBNETS_TO_TRY; do
  echo "  trying spot $try_subnet..."
  INSTANCE=$(_launch "$try_subnet" spot) && [ -n "$INSTANCE" ] && break
  echo "  trying on-demand $try_subnet..."
  INSTANCE=$(_launch "$try_subnet" ondemand) && [ -n "$INSTANCE" ] && break
  INSTANCE=""
done
[ -z "$INSTANCE" ] && { echo "ERROR: no $INSTANCE_TYPE capacity in any subnet of VPC $VPC_ID"; exit 1; }
echo "  launched: $INSTANCE"
echo "$INSTANCE" > /tmp/spinhance_instance_id

echo "  waiting for running state..."
aws ec2 wait instance-running \
  --profile "$PROFILE" --region "$REGION" \
  --instance-ids "$INSTANCE"
echo "  running — waiting 60s for SSH daemon..."
sleep 60
fi  # end of new-instance block

# ── 3. SSH helpers (config file avoids ProxyCommand word-splitting) ───────────
rm -f "$EICE_KEY" "$EICE_KEY.pub"
ssh-keygen -t ed25519 -N "" -f "$EICE_KEY" -q

cat > "$SSH_CFG" <<EOF
Host $INSTANCE
  User ec2-user
  IdentityFile $EICE_KEY
  StrictHostKeyChecking no
  ServerAliveInterval 30
  ProxyCommand aws ec2-instance-connect open-tunnel --instance-id $INSTANCE --profile $PROFILE --region $REGION
EOF

_push_key() {
  aws ec2-instance-connect send-ssh-public-key \
    --profile "$PROFILE" --region "$REGION" \
    --instance-id "$INSTANCE" \
    --instance-os-user ec2-user \
    --ssh-public-key "file://$EICE_KEY.pub" > /dev/null
}

_ssh() {
  _push_key
  ssh -F "$SSH_CFG" "$INSTANCE" "$@"
}

_sync_code() {
  local archive
  archive=$(mktemp /tmp/spinhance-code-XXXXX.tar.gz)

  echo "  archiving..."
  tar czf "$archive" -C "$REPO" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='simulation/data' \
    --exclude='mol_to_spin_system/data' \
    --exclude='model/checkpoints' \
    --exclude='autoai' \
    .

  echo "  uploading ($(du -sh "$archive" | cut -f1)) to S3..."
  aws s3 cp "$archive" "s3://$BUCKET/code/spinhance-code.tar.gz" --profile "$PROFILE" --region "$REGION"

  echo "  downloading on EC2 and extracting..."
  _ssh "mkdir -p $WORKSPACE && \
    aws s3 cp s3://$BUCKET/code/spinhance-code.tar.gz /tmp/spinhance-code.tar.gz && \
    tar xzf /tmp/spinhance-code.tar.gz -C $WORKSPACE && \
    rm /tmp/spinhance-code.tar.gz"

  rm "$archive"
  echo "  done."
}

# ── 4. Sync code + download data ──────────────────────────────────────────────
if [ "${REUSING:-false}" = "true" ]; then
  echo "[3/5] Skipping code sync (reusing instance)"
  echo "[4/5] Skipping data download (reusing instance)"
else
  echo "[3/5] Syncing code..."
  _sync_code

  echo "[4/5] Downloading data from S3..."
  _ssh "
    set -e
    mkdir -p $WORKSPACE/mol_to_spin_system/data $WORKSPACE/simulation/data/spectra/90MHz

    echo '  downloading spin_systems_60k.json...'
    aws s3 cp s3://$BUCKET/spin_systems_60k.json \
      $WORKSPACE/mol_to_spin_system/data/spin_systems_60k.json

    echo '  downloading 90MHz spectra tar...'
    aws s3 cp s3://$BUCKET/spectra/90MHz/mol_all.tar.gz \
      $WORKSPACE/simulation/data/spectra/90MHz/mol_all.tar.gz

    echo '  extracting spectra...'
    tar xzf $WORKSPACE/simulation/data/spectra/90MHz/mol_all.tar.gz \
      -C $WORKSPACE/simulation/data/spectra/90MHz/
    rm $WORKSPACE/simulation/data/spectra/90MHz/mol_all.tar.gz
    echo '  done.'
  "
fi

# ── 5. Launch training + diagnostics sync sidecar ─────────────────────────────
echo "[5/5] Starting training..."
_ssh "
  cd $WORKSPACE

  # Training process
  nohup bash -c 'PYTHONPATH=. /opt/conda/bin/conda run -n pytorch python -m model.run_experiment \
    --json mol_to_spin_system/data/spin_systems_60k.json \
    --spectra simulation/data/spectra \
    --s3-ckpt-prefix $S3_CKPT_PREFIX \
    $TRAIN_ARGS' \
    > /tmp/train.log 2>&1 &
  disown

  # Sidecar: sync JSONL diagnostics (no .pt) to S3 every 30s for live dashboard
  nohup bash -c "
    while true; do
      aws s3 sync $WORKSPACE/model/runs \
        s3://$BUCKET/model/runs \
        --exclude '*.pt' --no-progress 2>>/tmp/sync.log
      sleep 30
    done
  " > /tmp/sync.log 2>&1 &
  disown

  echo 'training + sync sidecar started'
"

# ── Done ──────────────────────────────────────────────────────────────────────
cat <<EOF

=== Training launched ===
Instance : $INSTANCE
Logs     : /tmp/train.log
Checkpoint: $WORKSPACE/model/checkpoints/spinhance.pt

To get a shell:
  aws ec2-instance-connect send-ssh-public-key \\
    --profile $PROFILE --region $REGION \\
    --instance-id $INSTANCE --instance-os-user ec2-user \\
    --ssh-public-key file://$EICE_KEY.pub
  ssh -F $SSH_CFG $INSTANCE

To terminate:
  aws ec2 terminate-instances \\
    --instance-ids $INSTANCE \\
    --profile $PROFILE --region $REGION
EOF
