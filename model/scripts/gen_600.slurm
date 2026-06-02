#!/bin/bash
# Generate the 600 MHz / 2^16 spectra bundle as a CPU-only SLURM array (one task
# per 50k-record shard, aligned to bundle_500k_s0/records.json.gz). CPU-only, so
# it schedules on free cores without competing for the training GPUs.
#   sbatch --array=0-9 model/scripts/gen_600.slurm
#SBATCH -p shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH -o /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%A_%a.out
#SBATCH -e /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%A_%a.err

set -e
cd /gpfs/home/labounader/code/spinhance
export PYTHONPATH=.
ENVPY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python
SRC_BUNDLE="${SRC_BUNDLE:-/gpfs/home/labounader/bundle_500k_s0}"
OUT="${OUT:-/gpfs/home/labounader/bundle_600_2p16}"

echo "host $(hostname) | shard $SLURM_ARRAY_TASK_ID | src $SRC_BUNDLE | out $OUT"
"$ENVPY" -m model.scripts.gen_600_shards \
  --records "$SRC_BUNDLE/records.json.gz" \
  --out "$OUT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  --chunk 50000 --field 600 --points 65536 --linewidth 0.35 \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
