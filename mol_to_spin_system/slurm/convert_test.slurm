#!/bin/bash
# Small Slurm test: convert ~100 molecules on one shared-partition node.
# Submit from the repo root:  sbatch mol_to_spin_system/slurm/convert_test.slurm
#SBATCH --job-name=mol2mat-test
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/processed

# java (OpenJDK 17) is on the default PATH; rdkit/numpy come from the env.
MM="$HOME/.local/bin/micromamba"

"$MM" run -n spinhance python -m mol_to_spin_system.xyz \
    generate/data/chembl_8spin.xyz.gz \
    data/processed/test_100.json \
    --workers "$SLURM_CPUS_PER_TASK" \
    --limit 100
