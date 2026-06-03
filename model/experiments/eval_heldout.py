"""
model.experiments.eval_heldout
==============================
Standardized evaluation on the GLOBAL held-out test pool — the shared 10% that
NO model trained on (``records_3M_test.json.gz`` + the full stacked parts). This
is independent of any within-run split, so every model is scored on identical
molecules → directly comparable. Writes ``<run_dir>/heldout_eval.json`` for the
live dashboard's standardized-eval panel.

The test records carry an explicit ``row`` (→ full parts) and are pre-sorted by
``test_rank``, so ``--limit N`` evaluates the first N = a nested subset shared
across model tiers (64k ⊂ 500k ⊂ 3M).

    python -m model.experiments.eval_heldout --run-dir model/runs/<id> [more...] \
        --test-records /gpfs/.../rebuild3M/records_3M_test.json.gz \
        --parts /gpfs/.../rebuild3M/parts --device cuda [--limit 50000]

Pass several ``--run-dir`` to score the whole fleet in one process: the spectra
are model-independent, so the held-out rows are ``preload``-ed into RAM ONCE
(sequential shard reads — fast on GPFS) and shared across every model; only the
per-model standardizer/target differs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.architectures import build_architecture
from model.data.collate import collate_spin_batch
from model.data.dataset import SpectrumMatrixDataset
from model.data.records import load_pubchem_records
from model.data.stacked_spectra import StackedSpectra
from model.data.standardization import DegeneracyVocab, Standardizer
from model.evaluation.metrics import evaluate_output


def _score_one(run_dir, checkpoint, recs, src, vocab, field, device, batch_size,
               region_tokens, out, test_records):
    ckpt_path = checkpoint or str(Path(run_dir) / "checkpoints" / "best.pt")
    out_path = out or (str(Path(run_dir) / "heldout_eval.json") if run_dir else "heldout_eval.json")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    std = Standardizer().load_state_dict(ck["standardizer"])
    mcfg = dict(ck["cfg"]["model"]); name = mcfg.pop("name")
    model = build_architecture(name, n_deg_classes=len(vocab), **mcfg).to(device).eval()
    model.load_state_dict(ck["model"], strict=False)

    # the dataset standardizes targets with THIS model's std; spectra come from the
    # shared, already-preloaded src (no re-read across models).
    ds = SpectrumMatrixDataset(recs, vocab, std, spectrum_field=f"spec{field}",
                               augment=False, region_tokens=region_tokens,
                               region_max=48, spectra_source=src)
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_spin_batch, num_workers=4)

    agg, n = {}, 0
    for batch in dl:
        batch = batch.to(device)
        with torch.no_grad():
            out_ = model(batch)
        for k, v in evaluate_output(out_, batch, std, vocab).items():
            if isinstance(v, (int, float)) and v == v:        # skip NaN
                agg[k] = agg.get(k, 0.0) + v
        n += 1
    metrics = {k: agg[k] / max(1, n) for k in agg}

    res = {
        "run_id": Path(run_dir).name if run_dir else name,
        "n_test": len(recs),
        "checkpoint": ckpt_path,
        "test_records": test_records,
        "metrics": metrics,
        "time": time.time(),
    }
    Path(out_path).write_text(json.dumps(res, indent=2))
    head = "  ".join(f"{k}={metrics[k]:.4f}" for k in
                     ("shift_mae_ppm", "j_mae_hz", "presence_f1", "deg_acc_balanced") if k in metrics)
    print(f"[{Path(run_dir).name if run_dir else name}] held-out ({len(recs)} mols): {head}\n  -> {out_path}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", nargs="+", required=True,
                    help="one or more run dirs; each uses checkpoints/best.pt and writes its heldout_eval.json")
    ap.add_argument("--checkpoint", help="explicit checkpoint (only valid with a single --run-dir)")
    ap.add_argument("--test-records", required=True, help="records_3M_test.json.gz (held-out, ranked)")
    ap.add_argument("--parts", required=True, help="full stacked parts dir (records carry global row)")
    ap.add_argument("--field", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0, help="first-N by test_rank (nested subset); 0 = all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--region-tokens", action="store_true")
    ap.add_argument("--out", help="output path (only valid with a single --run-dir)")
    a = ap.parse_args()
    if (a.checkpoint or a.out) and len(a.run_dir) != 1:
        ap.error("--checkpoint/--out require exactly one --run-dir")

    # test records: explicit row -> full parts; pre-sorted by test_rank, so [:limit] is a nested subset
    recs = load_pubchem_records(a.test_records)
    if a.limit > 0:
        recs = recs[:a.limit]
    src = StackedSpectra(a.parts)
    src.preload([r["row"] for r in recs])     # one sequential pass; shared across all models

    for rd in a.run_dir:
        try:
            _score_one(rd, a.checkpoint, recs, src, DegeneracyVocab(), a.field,
                       a.device, a.batch_size, a.region_tokens, a.out, a.test_records)
        except Exception as e:
            print(f"[{Path(rd).name}] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
