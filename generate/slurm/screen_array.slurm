#!/bin/bash
# Screen a compound database as a Slurm array: each task processes one shard,
# selecting records where (index % num_shards == task_id).  num_shards is
# derived from the array size, so keep the work consistent by editing only the
# --array range below.  Each task writes part_<id>.csv and part_<id>.xyz.gz.
#
# Submit from the repo root (point INPUT at the downloaded CID-SMILES.gz):
#   sbatch --export=ALL,INPUT=$HOME/pubchem/CID-SMILES.gz,SRC=pubchem \
#       generate/slurm/screen_array.slurm
# then merge once the array finishes:
#   micromamba run -n spinhance python -m generate.merge_shards \
#       data/screen/shards data/screen/pubchem_8spin.csv data/screen/pubchem_8spin.xyz.gz
#
#SBATCH --job-name=screen
#SBATCH --partition=shared
#SBATCH --array=0-63
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/screen/shards

INPUT="${INPUT:?set INPUT=/path/to/CID-SMILES.gz via --export}"
SRC="${SRC:-pubchem}"
MAXHEAVY="${MAXHEAVY:-50}"
MM="$HOME/.local/bin/micromamba"

"$MM" run -n spinhance python -m generate.cli run \
    --source "$SRC" \
    --chembl "$INPUT" \
    --output      "data/screen/shards/part_${SLURM_ARRAY_TASK_ID}.csv" \
    --xyz-output  "data/screen/shards/part_${SLURM_ARRAY_TASK_ID}.xyz.gz" \
    --max-heavy-atoms "$MAXHEAVY" \
    --num-shards  "$SLURM_ARRAY_TASK_COUNT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --workers     "$SLURM_CPUS_PER_TASK"
