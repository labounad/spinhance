#!/bin/bash
# Multi-GPU DDP training on Garibaldi (4x A6000 across 2 nodes by default).
# A6000 nodes have 2 GPUs each -> 4 GPUs = 2 nodes. Each srun task = one rank;
# the trainer reads SLURM_PROCID/LOCALID/NTASKS + MASTER_ADDR to init NCCL.
# Effective batch = batch_size (per-GPU) x ntasks; lr is scaled for the large batch.
#
# Submit (4-GPU, 2-node, full 3M):
#   sbatch --export=ALL,CONFIG=model/configs/train_500k_025.yaml,RUN_NAME=med025_500k_ddp \
#          -J xl025ddp model/scripts/train_hpc_ddp.slurm
# 2-GPU single-node smoke (validate DDP logic first):
#   sbatch --nodes=1 --ntasks-per-node=2 --gres=gpu:rtxa6000:2 \
#          --export=ALL,CONFIG=...,RUN_NAME=ddp_smoke,MAXMOL=20000,EPOCHS=1 \
#          -J ddpsmoke model/scripts/train_hpc_ddp.slurm
#SBATCH -p rtxa6000
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:rtxa6000:2
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
ulimit -n 16384 2>/dev/null || true    # headroom for stacked-shard mmaps (belt + LRU cap)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN          # surface NCCL init/interface problems in the log

# Per-GPU batch 512 (effective = 512 x world_size); lr scaled up for the large
# effective batch (~2048-4096) with a short warmup. 90 MHz only. MAXMOL/EPOCHS
# optional overrides for the smoke test.
DATA_OV="--set data.records=/gpfs/home/yimingzhang/hackathon-shared/spin_systems_pubchem.json.gz \
 --set data.parts=/gpfs/group/shenvi/Users/labounader/spinhance/spectra_run/out/90MHz \
 --set data.max_mol=${MAXMOL:-0} \
 --set training.batch_size=512 \
 --set training.num_workers=4 \
 --set training.lr=0.0012 \
 --set training.warmup_frac=0.03"
[ -n "$EPOCHS" ] && DATA_OV="$DATA_OV --set training.epochs=$EPOCHS"

echo "nodes=$SLURM_NODELIST master=$MASTER_ADDR ntasks=$SLURM_NTASKS | config=$CONFIG run=$RUN_NAME"
srun "$ENVPY" -m model.experiments.train --config "$CONFIG" \
  --set run.name="$RUN_NAME" $DATA_OV
