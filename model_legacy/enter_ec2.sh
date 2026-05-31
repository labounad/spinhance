#!/usr/bin/env bash
# ec2connect.sh — list your EC2 instances and SSH into one via EC2 Instance Connect.
#
# Usage:
#   ./ec2connect.sh                  # list running/stopped, pick a number to connect
#   ./ec2connect.sh <instance-id>    # connect directly (i-0123...)
#   ./ec2connect.sh <name>           # connect to instance whose Name tag matches (substring)
#   ./ec2connect.sh <target> -- <cmd...>   # run a remote command instead of a shell
#
# Env overrides:  PROFILE (default hack-scripps), REGION (us-west-2), SSH_USER (ec2-user)
# Requires: an EC2 Instance Connect Endpoint in the VPC (you've been using one).

set -euo pipefail

PROFILE="${PROFILE:-hack-scripps}"
REGION="${REGION:-us-west-2}"
SSH_USER="${SSH_USER:-ec2-user}"
KEY="/tmp/ec2connect-key"

aws_() { aws --profile "$PROFILE" --region "$REGION" "$@"; }

connect() {
  local iid="$1"; shift
  local state
  state=$(aws_ ec2 describe-instances --instance-ids "$iid" \
    --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo unknown)

  if [ "$state" != "running" ]; then
    echo "Instance $iid is '$state', not running." >&2
    if [ "$state" = "stopped" ]; then
      read -rp "Start it and wait? [y/N] " yn
      [[ "$yn" =~ ^[Yy]$ ]] || exit 1
      aws_ ec2 start-instances --instance-ids "$iid" >/dev/null
      echo "  starting…"; aws_ ec2 wait instance-running --instance-ids "$iid"; sleep 20
    else
      exit 1
    fi
  fi

  # Ephemeral key: the pushed public key is only valid ~60s, so push right before ssh.
  rm -f "$KEY" "$KEY.pub"
  ssh-keygen -t ed25519 -N "" -f "$KEY" -q
  aws_ ec2-instance-connect send-ssh-public-key \
    --instance-id "$iid" --instance-os-user "$SSH_USER" \
    --ssh-public-key "file://$KEY.pub" >/dev/null

  echo "Connecting to $iid as $SSH_USER …  (EICE tunnels drop at ~1h — run long jobs inside tmux)"
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -o ProxyCommand="aws ec2-instance-connect open-tunnel --instance-id $iid --profile $PROFILE --region $REGION" \
    "$SSH_USER@$iid" "$@"
}

resolve() {  # $1 = i-... or a Name substring -> prints an instance id
  local t="$1" id
  if [[ "$t" == i-* ]]; then echo "$t"; return; fi
  id=$(aws_ ec2 describe-instances \
    --filters "Name=tag:Name,Values=*${t}*" "Name=instance-state-name,Values=running,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || echo None)
  [ "$id" = "None" ] && { echo "No instance matching '$t'." >&2; exit 1; }
  echo "$id"
}

# ---- parse args (split off remote command after --) ----
ARGS=(); REMOTE=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; REMOTE=("$@"); break; fi
  ARGS+=("$1"); shift
done

# Direct target given -> connect straight away
if [ "${#ARGS[@]}" -gt 0 ]; then
  connect "$(resolve "${ARGS[0]}")" "${REMOTE[@]}"
  exit 0
fi

# ---- interactive menu ----
printf "\n  %-3s %-21s %-9s %-13s %-15s %-12s %s\n" "#" "INSTANCE ID" "STATE" "TYPE" "PRIVATE IP" "AZ" "NAME"
i=0; declare -a IDS
while IFS=$'\t' read -r id state type ip az name; do
  [ -z "$id" ] && continue
  [ "$ip" = "None" ] && ip="-"; [ "$name" = "None" ] && name="-"
  i=$((i+1)); IDS[$i]="$id"
  printf "  %-3s %-21s %-9s %-13s %-15s %-12s %s\n" "$i" "$id" "$state" "$type" "$ip" "$az" "$name"
done < <(aws_ ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running,stopped" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,InstanceType,PrivateIpAddress,Placement.AvailabilityZone,Tags[?Key==`Name`]|[0].Value]' \
  --output text)

[ "$i" -eq 0 ] && { echo "  (no running/stopped instances in $REGION)"; exit 0; }

echo
read -rp "Connect to # (or q to quit): " sel
[[ "$sel" =~ ^[0-9]+$ ]] || { echo "bye"; exit 0; }
[ -n "${IDS[$sel]:-}" ] || { echo "invalid selection"; exit 1; }
connect "${IDS[$sel]}"