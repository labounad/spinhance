#!/bin/bash
# Full 3.2M PubChem training on an alphafold A5000 (the A6000 partition is down).
# IMPORTANT tradeoffs baked in:
#   - preload=false (mmap): the 196GB preload array took down the A6000 nodes; mmap +
#     the StackedSpectra LRU fd-cap is SAFE but I/O-bound on GPFS (slow ~few-s/step).
#   - num_workers=4: the 3.2M-record list fork-copies per worker; more workers OOM.
#   - batch 256: xl (57M) fits the A5000's 24GB at 256 (512 would OOM).
# So this RUNS without crashing nodes, but is slow — it won't converge quickly; it's a
# background "let it cook" run, not a fast result.
#   sbatch --export=ALL,CONFIG=model/configs/train_3M_026.yaml,RUN_NAME=xl026_3M_a5000 -J xl026_3m model/scripts/train_hpc_3M.slurm
#SBATCH -p alphafold
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.out
#SBATCH -e /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.err

set -e
cd /gpfs/home/labounader/code/spinhance
export PYTHONPATH=.
ulimit -n 16384 2>/dev/null || true
ENVPY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python

DATA_OV="--set data.records=/gpfs/home/yimingzhang/hackathon-shared/spin_systems_pubchem.json.gz \
 --set data.parts=/gpfs/group/shenvi/Users/labounader/spinhance/spectra_run/out/90MHz \
 --set data.max_mol=0 --set data.sample_n=0 --set data.preload=false \
 --set training.batch_size=256 --set training.num_workers=4"

echo "host $(hostname) | config $CONFIG run $RUN_NAME"
srun "$ENVPY" -m model.experiments.train --config "$CONFIG" \
  --set run.name="$RUN_NAME" $DATA_OV
