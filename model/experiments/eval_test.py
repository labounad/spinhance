"""
model.experiments.eval_test
===========================
Headless TEST-split evaluation of a trained matrix-model checkpoint on the STORED
(training-distribution) 90 MHz spectra — the faithful answer to "is the model
actually good on held-out test molecules?", independent of the GUI.

Rebuilds the same molecule-level test fold (identical seed), runs the model on
the stored .npy spectra, and reports mean canonical + Hungarian-matched metrics
via the shared evaluate_output. Use --region-tokens to feed support-region tokens
(required to fairly evaluate a model TRAINED with them).

    python -m model.experiments.eval_test --checkpoint best.pt \
        --records mol_to_spin_system/data/spin_systems_chembl.json \
        --spectra simulation/data/spectra --device cuda
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.architectures import build_architecture
from model.data.collate import collate_spin_batch
from model.data.dataset import SpectrumMatrixDataset
from model.data.records import load_records
from model.data.splits import make_splits
from model.data.standardization import DegeneracyVocab, Standardizer
from model.evaluation.metrics import evaluate_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--spectra", required=True)
    ap.add_argument("--field", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--region-tokens", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = DegeneracyVocab()
    std = Standardizer().load_state_dict(ckpt["standardizer"])
    mcfg = dict(ckpt["cfg"]["model"]); name = mcfg.pop("name")
    model = build_architecture(name, n_deg_classes=len(vocab), **mcfg).to(args.device).eval()
    model.load_state_dict(ckpt["model"])

    recs = load_records(args.records, args.spectra, fields=(args.field,), require_spectra=True)
    assign, _ = make_splits(recs, seed=args.seed, compute_scaffold=False)
    test = [r for r in recs if assign.get(r["mol_id"]) == "test"]
    ds = SpectrumMatrixDataset(test, vocab, std, spectrum_field=f"spec{args.field}",
                               augment=False, region_tokens=args.region_tokens, region_max=48)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_spin_batch, num_workers=4)

    agg, n = {}, 0
    for batch in dl:
        batch = batch.to(args.device)
        with torch.no_grad():
            out = model(batch)
        for k, v in evaluate_output(out, batch, std, vocab).items():
            if isinstance(v, (int, float)) and v == v:        # skip NaN
                agg[k] = agg.get(k, 0.0) + v
        n += 1

    print(f"\n===== TEST-SET ({len(test)} molecules) | {name} | "
          f"region_tokens={args.region_tokens} =====")
    for k in ("shift_mae_ppm", "j_mae_hz", "presence_f1", "presence_precision",
              "presence_recall", "deg_acc", "deg_acc_balanced",
              "h_shift_mae_ppm", "h_j_mae_hz"):
        if k in agg:
            print(f"  {k:22s} {agg[k]/max(1,n):.4f}")


if __name__ == "__main__":
    main()
