#!/usr/bin/env bash
# upload_data.sh — Upload training data to S3. Run locally before training on EC2.
#
# Usage:
#   bash model/upload_data.sh

set -euo pipefail

PROFILE="${AWS_PROFILE:-hack-scripps}"
REGION="${AWS_REGION:-us-west-2}"
BUCKET="spinhance-data"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Uploading training data to s3://$BUCKET ==="

echo "[1/2] Uploading spin_systems_chembl.json..."
aws s3 cp "$REPO/mol_to_spin_system/data/spin_systems_chembl.json" \
  "s3://$BUCKET/spin_systems_chembl.json" \
  --profile "$PROFILE" --region "$REGION"

echo "[2/2] Creating and uploading 90MHz spectra tar (may take a few minutes)..."
TMP=$(mktemp /tmp/spectra-90MHz-XXXXX.tar.gz)
tar czf "$TMP" -C "$REPO/simulation/data/spectra/90MHz" .
aws s3 cp "$TMP" "s3://$BUCKET/spectra/90MHz/mol_all.tar.gz" \
  --profile "$PROFILE" --region "$REGION"
rm "$TMP"

echo ""
echo "=== Done. On EC2, run: ==="
echo ""
echo "  bash ~/spinhance/model/train.sh"
