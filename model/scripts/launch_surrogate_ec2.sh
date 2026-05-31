#!/usr/bin/env bash
# launch_surrogate_ec2.sh — Provision a GPU instance, simulate the 90+600 MHz
# pyspin spectra for the randomized 64k ChEMBL set (~minutes), then train the
# differentiable surrogate renderer (Branch 5), syncing the run dir to S3 live.
#
# Usage:
#   bash model/scripts/launch_surrogate_ec2.sh [INSTANCE_TYPE] [--set k=v ...]
#   bash model/scripts/launch_surrogate_ec2.sh g6e.xlarge --set training.epochs=5
#
# Prereqs: aws sso login --profile hack-scripps  (randomized json already on S3)
set -euo pipefail

PROFILE="${AWS_PROFILE:-hack-scripps}"
REGION="${AWS_REGION:-us-west-2}"
INSTANCE_TYPE="${1:-g6e.xlarge}"; shift 2>/dev/null || true
EXTRA_SET="$*"
CONFIG="model/configs/surrogate.yaml"
JSON_NAME="spin_systems_chembl_8spin_randomized.json"

BUCKET="spinhance-data"
SUBNET="subnet-0096ffc9c05bebab3"
SG="sg-09d5ef7889a26f56a"
INST_PROFILE="hackathon-ec2-profile"
EICE_KEY="/tmp/spinhance-surrogate-key"
SSH_CFG="/tmp/spinhance-surrogate-ssh.cfg"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORKSPACE="/home/ec2-user/spinhance"

echo "=== SpinHance EC2 (surrogate renderer) ==="
echo "  instance : $INSTANCE_TYPE"
echo "  config   : $CONFIG   overrides: ${EXTRA_SET:-(none)}"

# ── Session number (auto-increment from S3) ───────────────────────────────────
LAST=$(aws s3 ls "s3://$BUCKET/training/" --profile "$PROFILE" --region "$REGION" 2>/dev/null \
  | grep -oE 'session[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1 || true)
SESSION=$(printf "%03d" $(( 10#${LAST:-0} + 1 )))
S3_RUN="s3://$BUCKET/training/session$SESSION"
echo "  session  : $SESSION  ($S3_RUN)"

# ── 1. Latest Deep Learning AMI ───────────────────────────────────────────────
echo "[1/6] Finding latest Deep Learning AMI..."
AMI=$(aws ec2 describe-images --profile "$PROFILE" --region "$REGION" --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Amazon Linux 2023*" \
            "Name=architecture,Values=x86_64" "Name=state,Values=available" \
  --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text)
[ -z "$AMI" ] && { echo "ERROR: no Deep Learning AMI found"; exit 1; }
echo "  AMI: $AMI"

# ── 2. Launch (spot, on-demand fallback; try subnets in the VPC) ──────────────
echo "[2/6] Launching $INSTANCE_TYPE..."
VPC_ID=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" \
  --subnet-ids "$SUBNET" --query 'Subnets[0].VpcId' --output text)
SUBNETS=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=state,Values=available" \
  --query 'Subnets[*].SubnetId' --output text | tr '\t' ' ')
SUBNETS="$SUBNET $(echo "$SUBNETS" | sed "s/$SUBNET//g")"

_launch() {
  local subnet=$1 market=$2 opts=""
  [ "$market" = "spot" ] && opts='--instance-market-options {"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}'
  aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --subnet-id "$subnet" \
    --security-group-ids "$SG" --iam-instance-profile "Name=$INST_PROFILE" \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":120,"VolumeType":"gp3"}}]' \
    --metadata-options '{"HttpTokens":"required"}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=spinhance-surrogate}]' \
    --query 'Instances[0].InstanceId' --output text $opts 2>/dev/null
}

INSTANCE=""
for sn in $SUBNETS; do
  echo "  trying spot $sn..."; INSTANCE=$(_launch "$sn" spot) && [ -n "$INSTANCE" ] && break
  echo "  trying on-demand $sn..."; INSTANCE=$(_launch "$sn" ondemand) && [ -n "$INSTANCE" ] && break
  INSTANCE=""
done
[ -z "$INSTANCE" ] && { echo "ERROR: no $INSTANCE_TYPE capacity in VPC $VPC_ID"; exit 1; }
echo "  launched: $INSTANCE"
echo "$INSTANCE" > /tmp/spinhance_surrogate_instance_id
aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE"
echo "  running — waiting 60s for SSH daemon..."; sleep 60

# ── 3. SSH helpers (EICE) ─────────────────────────────────────────────────────
rm -f "$EICE_KEY" "$EICE_KEY.pub"; ssh-keygen -t ed25519 -N "" -f "$EICE_KEY" -q
cat > "$SSH_CFG" <<EOF
Host $INSTANCE
  User ec2-user
  IdentityFile $EICE_KEY
  StrictHostKeyChecking no
  ServerAliveInterval 30
  ProxyCommand aws ec2-instance-connect open-tunnel --instance-id $INSTANCE --profile $PROFILE --region $REGION
EOF
_ssh() {
  aws ec2-instance-connect send-ssh-public-key --profile "$PROFILE" --region "$REGION" \
    --instance-id "$INSTANCE" --instance-os-user ec2-user \
    --ssh-public-key "file://$EICE_KEY.pub" >/dev/null
  ssh -F "$SSH_CFG" "$INSTANCE" "$@"
}

# ── 4. Sync code + download the spin-system dataset ───────────────────────────
echo "[3/6] Syncing code..."
ARCHIVE=$(mktemp /tmp/spinhance-code-XXXXX.tar.gz)
tar czf "$ARCHIVE" -C "$REPO" --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='simulation/data' --exclude='mol_to_spin_system/data' --exclude='generate/data' \
  --exclude='model/runs' --exclude='model/checkpoints' --exclude='docs/data' .
aws s3 cp "$ARCHIVE" "s3://$BUCKET/code/spinhance-surrogate.tar.gz" --profile "$PROFILE" --region "$REGION" --no-progress
_ssh "mkdir -p $WORKSPACE && aws s3 cp s3://$BUCKET/code/spinhance-surrogate.tar.gz /tmp/c.tar.gz && \
  tar xzf /tmp/c.tar.gz -C $WORKSPACE && rm /tmp/c.tar.gz"
rm "$ARCHIVE"

echo "[4/6] Downloading randomized dataset on instance..."
_ssh "set -e
  mkdir -p $WORKSPACE/mol_to_spin_system/data
  aws s3 cp s3://$BUCKET/$JSON_NAME $WORKSPACE/mol_to_spin_system/data/$JSON_NAME"

# ── 5. Simulate 90 + 600 MHz pyspin spectra (the surrogate's targets) ─────────
echo "[5/6] Simulating 90+600 MHz spectra (pyspin, all cores)..."
_ssh "
  cd $WORKSPACE
  PY=/opt/pytorch/bin/python; [ -x \"\$PY\" ] || PY=\"/opt/conda/bin/conda run -n pytorch python\"
  \$PY -m pip install -q pyyaml 2>/dev/null || true
  PYTHONPATH=. \$PY -m simulation.cli run \
    --graphs mol_to_spin_system/data/$JSON_NAME \
    --out_dir simulation/data --fields 90 600 --engine python --workers \$(nproc) \
    2>&1 | tail -3
"

# ── 6. Launch surrogate training (tmux) + S3 sync sidecar ─────────────────────
echo "[6/6] Starting surrogate training..."
_ssh "
  cd $WORKSPACE
  PY=/opt/pytorch/bin/python; [ -x \"\$PY\" ] || PY=\"/opt/conda/bin/conda run -n pytorch python\"
  tmux new-session -d -s surrogate \"cd $WORKSPACE && PYTHONPATH=. \$PY -m model.experiments.train_surrogate \
    --config $CONFIG \
    --set data.records=mol_to_spin_system/data/$JSON_NAME \
    --set data.spectra=simulation/data/spectra \
    --set run.name=session$SESSION $EXTRA_SET 2>&1 | tee /home/ec2-user/surrogate_$SESSION.log\"
  tmux new-session -d -s sync \"while true; do \
    aws s3 sync $WORKSPACE/model/runs $S3_RUN/runs --no-progress 2>>/home/ec2-user/sync.log; \
    sleep 60; done\"
  echo 'simulation done; surrogate training + sync started'
"

cat <<EOF

=== Launched ===
Instance : $INSTANCE   ($INSTANCE_TYPE)
Session  : session$SESSION   ->  $S3_RUN/runs
Logs     : /home/ec2-user/surrogate_$SESSION.log   (tmux: surrogate)

Monitor live (dashboard):
  AWS_PROFILE=$PROFILE streamlit run model/diagnostics/live_dashboard.py   # pick session$SESSION

Terminate when done:
  aws ec2 terminate-instances --instance-ids $INSTANCE --profile $PROFILE --region $REGION
EOF
