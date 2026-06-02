#!/bin/bash
# Train the differentiable SURROGATE renderer on a PREBUILT bundle (90 MHz shards).
# Mirrors train_hpc_bundle.slurm but drives model.experiments.train_surrogate.
# Submit:
#   sbatch --exclude=nodeb301 \
#     --export=ALL,CONFIG=model/configs/surrogate_pubchem_90.yaml,RUN_NAME=surrogate_pubchem_90,BUNDLE=/gpfs/home/labounader/bundle_500k_s0 \
#     -J surr90 model/scripts/train_surrogate_bundle.slurm
#SBATCH -p alphafold
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.out
#SBATCH -e /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.err

set -e
cd /gpfs/home/labounader/code/spinhance
export PYTHONPATH=.
ulimit -n 16384 2>/dev/null || true
ENVPY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python
BUNDLE="${BUNDLE:-/gpfs/home/labounader/bundle_500k_s0}"

# point the surrogate trainer at the prebuilt 90 MHz bundle (records + stacked shards)
DATA_OV="--set data.records=$BUNDLE/records.json.gz \
 --set data.parts=$BUNDLE \
 --set data.sample_n=0 \
 --set data.preload=false \
 --set training.num_workers=4"

echo "host $(hostname) | bundle $BUNDLE | config $CONFIG run $RUN_NAME"
srun "$ENVPY" -m model.experiments.train_surrogate --config "$CONFIG" \
  --set run.name="$RUN_NAME" $DATA_OV
