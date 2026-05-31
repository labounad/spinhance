#!/usr/bin/env bash
# upload_data.sh — Upload training data to S3. Run locally before training on EC2.
#
# Usage:
#   bash modelv2/upload_data.sh

set -euo pipefail

PROFILE="${AWS_PROFILE:-hack-scripps}"
REGION="${AWS_REGION:-us-west-2}"
BUCKET="spinhance-data"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

SPIN_SYSTEMS="$REPO/mol_to_spin_system/data/buckets/spin_systems_chembl_8spin.json.gz"
SPECTRA="$REPO/simulation/data/spectra/90MHz.tar.gz"

echo "=== Uploading training data to s3://$BUCKET ==="

echo "[1/2] Uploading spin_systems_chembl_8spin.json.gz..."
aws s3 cp "$SPIN_SYSTEMS" \
  "s3://$BUCKET/mol_to_spin_system/data/buckets/spin_systems_chembl_8spin.json.gz" \
  --profile "$PROFILE" --region "$REGION"

echo "[2/2] Uploading 90MHz.tar.gz..."
aws s3 cp "$SPECTRA" \
  "s3://$BUCKET/simulation/data/spectra/90MHz.tar.gz" \
  --profile "$PROFILE" --region "$REGION"

echo ""
echo "=== Done. On EC2: ==="
echo ""
echo "  aws s3 cp s3://$BUCKET/mol_to_spin_system/data/buckets/spin_systems_chembl_8spin.json.gz ."
echo "  aws s3 cp s3://$BUCKET/simulation/data/spectra/90MHz.tar.gz ."
echo "  export SPINHANCE_OUT=s3://$BUCKET/training"
echo "  PYTHONPATH=. python -m modelv2.train \\"
echo "    --spin_systems=spin_systems_chembl_8spin.json.gz \\"
echo "    --spectra=90MHz.tar.gz"
