#!/bin/bash
# refresh_website_data.sh — one command to regenerate all MODEL-viewer JSON for the
# website from the corrected-data REBUILD fleet, once checkpoints exist. Run on the
# HPC on a GPU node (the held-out eval wants CUDA); then pull docs/data/*.json locally.
#
#   srun -p rtxa6000 --gres=gpu:1 --mem=48G -t 2:00:00 \
#        bash docs/refresh_website_data.sh
#
# It is turnkey + incremental: each tier is included only once its best.pt exists, so
# you can run it repeatedly as 64k -> 500k -> 3M finish and the viewers gain tiers.
set -e
C=/gpfs/home/labounader/code/spinhance
REB=/gpfs/group/shenvi/Users/labounader/spinhance/rebuild3M
CONSOL_TEST=/gpfs/group/shenvi/Users/labounader/spinhance/consolidated_test  # contiguous held-out shards (fast eval)
PY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python
cd "$C"; export PYTHONPATH=.
RUNS="$C/model/runs"

# The full fleet: 64k + 500k recipe sweeps (025-030) and the 3M tiers (026/030).
# Exact names skip the cancelled rebuild_3M_xl / rebuild_500k_030_sym(legacy). Finished-only guard below.
FLEET_NAMES="rebuild_64k_025_sym rebuild_64k_026_sym rebuild_64k_027_sym rebuild_64k_028_sym rebuild_64k_029_sym rebuild_64k_030_sym \
rebuild_500k_025_sym rebuild_500k_026_sym rebuild_500k_027_sym rebuild_500k_028_sym rebuild_500k_029_sym rebuild_500k_030_sym \
rebuild_3M_026_sym rebuild_3M_030_sym"

# 1. standardized held-out eval on the leakage-controlled 10% PubChem test split.
#    Every run with a best.pt, in ONE process (the held-out spectra are preloaded ONCE and
#    shared); writes <run_dir>/heldout_eval.json that gen_viewer_data.py reads.
DIRS=""   # collected for a SINGLE --run-dir flag (it is nargs='+'; repeating the flag keeps only the last)
echo "run dirs (finished only — avoids reading a checkpoint mid-write):"
for nm in $FLEET_NAMES; do
  d=$(ls -dt $RUNS/*_${nm}_* 2>/dev/null | head -1)
  [ -n "$d" ] && [ -f "$d/checkpoints/best.pt" ] || continue
  st=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('state',''))" "$d/status.json" 2>/dev/null)
  if [ "$st" = "finished" ] || [ "$st" = "completed" ]; then printf '  %s (%s)\n' "$d" "$st"; DIRS="$DIRS $d"; fi
done
if [ -n "$DIRS" ]; then
  echo "== held-out eval =="
  $PY -m model.experiments.eval_heldout --run-dir $DIRS \
      --test-records "$CONSOL_TEST/records_test_consol.json.gz" --parts "$CONSOL_TEST/parts" \
      --device "${DEVICE:-cuda}" --limit "${LIMIT:-50000}"
else
  echo "WARN: no best.pt checkpoints yet — skipping held-out eval"
fi

# 2. learning_curves.json + test_eval.json (val curves + held-out val/test)
echo "== learning curves + test eval =="
$PY docs/gen_viewer_data.py

# 3. test_explorer.json (held-out molecules + per-model predictions/rendered spectra)
echo "== test-molecule explorer =="
$PY docs/export_test_explorer.py docs/data/test_explorer.json "${EXPLORER_N:-80}"

echo "== DONE =="
ls -l docs/data/learning_curves.json docs/data/test_eval.json docs/data/test_explorer.json
echo "Now pull docs/data/*.json to the local repo and commit."
