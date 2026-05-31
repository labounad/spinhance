#!/bin/bash
# Screen a compound database as a Slurm array: each task processes one shard,
# selecting records where (index % num_shards == task_id).  num_shards is
# derived from the array size, so keep the work consistent by editing only the
# --array range below.  Each task writes part_<id>.csv and part_<id>.xyz.gz.
#
# Submit from the repo root (point INPUT at the downloaded CID-SMILES.gz).
#
#   # exact 8-spin screen (default):
#   sbatch --export=ALL,INPUT=$HOME/pubchem/CID-SMILES.gz,SRC=pubchem \
#       generate/slurm/screen_array.slurm
#
#   # categorising scan, buckets 1..26 in one pass:
#   sbatch --export=ALL,INPUT=$HOME/pubchem/CID-SMILES.gz,SRC=pubchem,MINGROUPS=1,MAXGROUPS=26 \
#       generate/slurm/screen_array.slurm
#
# then merge the shards into one combined file AND dedup by InChIKey in one step
# (PubChem deposits the same compound under multiple CIDs):
#   micromamba run -n spinhance python -m generate.cli merge data/screen/shards \
#       --output data/screen/pubchem_all.csv --xyz-output data/screen/pubchem_all.xyz.gz --dedup
# and (for a range scan) split the deduped combined file into per-count datasets:
#   micromamba run -n spinhance python -m generate.cli split \
#       data/screen/pubchem_all.csv data/screen/buckets \
#       --prefix pubchem --xyz data/screen/pubchem_all.xyz.gz
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
MINGROUPS="${MINGROUPS:-8}"   # default: exact 8-spin screen (MIN==MAX)
MAXGROUPS="${MAXGROUPS:-8}"   # set MINGROUPS=1,MAXGROUPS=26 for a range scan
MM="$HOME/.local/bin/micromamba"

"$MM" run -n spinhance python -m generate.cli run \
    --source "$SRC" \
    --chembl "$INPUT" \
    --output      "data/screen/shards/part_${SLURM_ARRAY_TASK_ID}.csv" \
    --xyz-output  "data/screen/shards/part_${SLURM_ARRAY_TASK_ID}.xyz.gz" \
    --min-groups "$MINGROUPS" \
    --max-groups "$MAXGROUPS" \
    --max-heavy-atoms "$MAXHEAVY" \
    --num-shards  "$SLURM_ARRAY_TASK_COUNT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --workers     "$SLURM_CPUS_PER_TASK"
