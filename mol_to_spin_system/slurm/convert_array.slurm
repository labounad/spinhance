#!/bin/bash
# Full conversion as a Slurm array: each task converts one shard of the input,
# selecting blocks where (index % num_shards == task_id). num_shards is derived
# from the array size, so keep --array and the work consistent by editing only
# the --array range below.
#
# Submit from the repo root:
#   sbatch --export=ALL,INPUT=generate/data/chembl_8spin.xyz.gz mol_to_spin_system/slurm/convert_array.slurm
# then merge once it finishes:
#   micromamba run -n spinhance python -m mol_to_spin_system.merge_shards \
#       data/processed/spin_systems.json data/processed/shards
#
#SBATCH --job-name=mol2mat
#SBATCH --partition=shared
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/processed/shards

INPUT="${INPUT:-generate/data/chembl_8spin.xyz.gz}"
MM="$HOME/.local/bin/micromamba"

eval "$("$MM" shell hook --shell bash)"
micromamba activate spinhance

python -m mol_to_spin_system.xyz \
    "$INPUT" \
    "data/processed/shards/shard_${SLURM_ARRAY_TASK_ID}.json" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" \
    --shard-index "$SLURM_ARRAY_TASK_ID"
