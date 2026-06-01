#!/bin/bash
# Build a random 500k bundle ONCE on CPU (no GPU). Reads the 3200 shards
# sequentially a single time and writes a compact ~33GB bundle the GPU jobs reuse.
# Runs on a CPU partition so it never risks the GPU nodes; one sequential reader
# (not 4 concurrent) is gentle on GPFS.
#   sbatch model/scripts/bundle_subset.slurm
#SBATCH -J bundle500k
#SBATCH -p shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.out
#SBATCH -e /gpfs/home/labounader/code/spinhance/hpc_logs/%x_%j.err

set -e
cd /gpfs/home/labounader/code/spinhance
export PYTHONPATH=.
ulimit -n 16384 2>/dev/null || true
ENVPY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python

"$ENVPY" -m model.scripts.bundle_subset \
  --records /gpfs/home/yimingzhang/hackathon-shared/spin_systems_pubchem.json.gz \
  --parts   /gpfs/home/yimingzhang/hackathon-shared/spectra_run/out/90MHz \
  --out     /gpfs/home/labounader/bundle_500k_s0 \
  --sample_n 500000 --seed 0
