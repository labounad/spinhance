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
REB=/gpfs/home/labounader/rebuild3M
PY=/gpfs/home/labounader/micromamba/envs/spinhance/bin/python
cd "$C"; export PYTHONPATH=.
RUNS="$C/model/runs"

# latest run dir per tier (exact names skip the cancelled rebuild_3M_xl)
d64=$(ls -dt $RUNS/*_rebuild_64k_026_* 2>/dev/null | head -1)
d500=$(ls -dt $RUNS/*_rebuild_500k_xl_026_* 2>/dev/null | head -1)
d3m=$(ls -dt $RUNS/*_rebuild_3M_xxl_026_* 2>/dev/null | head -1)
echo "run dirs:"; printf '  %s\n' "$d64" "$d500" "$d3m"

# 1. standardized held-out eval on the leakage-controlled 10% PubChem test split.
#    All available tiers in one process (the held-out spectra are preloaded ONCE and
#    shared); writes <run_dir>/heldout_eval.json that gen_viewer_data.py reads.
ARGS=""
for d in "$d64" "$d500" "$d3m"; do
  [ -n "$d" ] && [ -f "$d/checkpoints/best.pt" ] && ARGS="$ARGS --run-dir $d"
done
if [ -n "$ARGS" ]; then
  echo "== held-out eval =="
  $PY -m model.experiments.eval_heldout $ARGS \
      --test-records "$REB/records_3M_test.json.gz" --parts "$REB/parts" \
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
