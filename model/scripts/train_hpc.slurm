#!/bin/bash
# Garibaldi HPC GPU training for SpinHance (PubChem 3M+, 90 MHz).
# Submit:
#   sbatch --export=ALL,CONFIG=model/configs/train_3M_spingraph_xl_025.yaml,RUN_NAME=xl025_3M_hpc \
#          -J xl025 model/scripts/train_hpc.slurm
# Data lives ON the HPC (no transfer); metrics write locally to model/runs/<id>/
# (no live S3 dashboard — pull metrics.jsonl down to view). The CONFIG's data paths
# + batch/workers are overridden below for the HPC (gpfs data, 257GB RAM box).
#SBATCH -p rtxa6000
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH -t 24:00:00
#SBATCH -o /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.out
#SBATCH -e /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.err

set -e
REPO=/gpfs/home/labounader/code/spinhance
ENVPY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python
cd "$REPO"
export PYTHONPATH=.

# HPC data + scale overrides (full 3.2M; big batch — the xl model is memory-light;
# 257GB RAM so workers are cheap). 90 MHz ONLY (model input constraint).
DATA_OV="--set data.records=/gpfs/home/yimingzhang/hackathon-shared/spin_systems_pubchem.json.gz \
 --set data.parts=/gpfs/home/yimingzhang/hackathon-shared/spectra_run/out/90MHz \
 --set data.max_mol=0 \
 --set training.batch_size=512 \
 --set training.num_workers=4"

echo "host $(hostname) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "config $CONFIG | run $RUN_NAME"
srun "$ENVPY" -m model.experiments.train --config "$CONFIG" \
  --set run.name="$RUN_NAME" $DATA_OV
