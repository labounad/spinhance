#!/bin/bash
# GPU training on a PREBUILT bundle (see bundle_subset.slurm). The bundle is a small
# (~33GB) contiguous stacked-shard dir, so we preload it into RAM (fast: a few big
# sequential reads, not 3200 random ones) with modest --mem — no GPFS I/O storm, no
# 210GB array. Submit:
#   sbatch --export=ALL,CONFIG=model/configs/train_500k_light_025.yaml,RUN_NAME=light_025_bundle,BUNDLE=/gpfs/group/shenvi/Users/labounader/spinhance/legacy/bundle_500k_s0 \
#          -J light025b model/scripts/train_hpc_bundle.slurm
#SBATCH -p alphafold
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.out
#SBATCH -e /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.err

set -e
cd /gpfs/home/labounader/code/spinhance
export PYTHONPATH=.
ulimit -n 16384 2>/dev/null || true
ENVPY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python
BUNDLE="${BUNDLE:-/gpfs/group/shenvi/Users/labounader/spinhance/legacy/bundle_500k_s0}"
# PRELOAD=true loads the ~33 GB (500k @ 2^14) bundle into RAM once, so training is
# immune to GPFS random-mmap contention on busy nodes (the 500k·026 crawl). Set
# --mem >= 48G at submit when using it.
PRELOAD="${PRELOAD:-false}"

# point at the prebuilt bundle; sample_n=0 (already sampled)
DATA_OV="--set data.records=$BUNDLE/records.json.gz \
 --set data.parts=$BUNDLE \
 --set data.sample_n=0 --set data.max_mol=0 \
 --set data.preload=$PRELOAD \
 --set training.batch_size=512 --set training.num_workers=4"

echo "host $(hostname) | bundle $BUNDLE | config $CONFIG run $RUN_NAME"
srun "$ENVPY" -m model.experiments.train --config "$CONFIG" \
  --set run.name="$RUN_NAME" $DATA_OV
