"""
model.run_experiment
====================
End-to-end Task 4 entry point on the preliminary dataset: adapter -> splits ->
fit. Stage-1 (matrix loss) by default; --stage2 enables the curriculum blend.

Examples
--------
# validate the whole non-torch data path (no training, no torch needed):
PYTHONPATH=. python3 -m model.run_experiment --dry-run --no-scaffold

# Stage-1 training on the 1072-mol set (small model for small data):
PYTHONPATH=. python3 -m model.run_experiment --small --epochs 60 --batch 64

# add Stage-2 spectral consistency (curriculum blend) after epoch 40:
PYTHONPATH=. python3 -m model.run_experiment --small --epochs 80 \
    --stage1-epochs 40 --ramp-epochs 10 --stage2 --batch 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model.data_adapter import load_records, renderable_mask
from model.splits import make_splits
from model.targets import DegeneracyVocab, Standardizer, encode_target

REPO = Path(__file__).resolve().parents[1]
DEF_JSON = REPO / "mol_to_spin_system/data/spin_systems.json"
DEF_SPECTRA = REPO / "simulation/data/spectra"


def _load(args):
    fields = tuple(int(f) for f in args.fields.split(","))
    recs = load_records(args.json, args.spectra, fields=fields)
    if args.max_mol:
        recs = recs[:args.max_mol]
    assignment, report = make_splits(recs, ratios=(0.7, 0.2, 0.1), seed=args.seed,
                                     compute_scaffold=args.scaffold)
    return recs, assignment, report


def dry_run(args):
    """Exercise adapter -> splits -> standardizer -> target encoding, all
    torch-free, to confirm the data path before launching training."""
    recs, assignment, report = _load(args)
    print(f"records: {len(recs)} | split {report['counts']} "
          f"ratios {{'train': {report['ratios']['train']:.2f}, "
          f"'val': {report['ratios']['val']:.2f}, 'test': {report['ratios']['test']:.2f}}}")
    print(f"leakage scaffold={report['scaffold_leaks']} "
          f"dup_matrix={report['dup_matrix_leaks']} | groups={report['n_groups']}")

    vocab = DegeneracyVocab()
    train = [r for r in recs if assignment[r["mol_id"]] == "train"]
    std = Standardizer().fit(train, vocab)
    print(f"standardizer(train): shift {std.shift_mean:.2f}±{std.shift_std:.2f} ppm | "
          f"J {std.j_mean:.2f}±{std.j_std:.2f} Hz")

    n_bad = 0
    for r in recs:
        try:
            encode_target(r["shifts"], r["couplings"], r["degeneracy"], vocab)
        except KeyError:
            n_bad += 1
    print(f"target encoding: {len(recs)-n_bad}/{len(recs)} OK (vocab miss: {n_bad})")

    rmask = renderable_mask(recs, max_block=args.max_block)
    print(f"Stage-2 renderable (max Mz-block<={args.max_block}): "
          f"{sum(rmask)}/{len(recs)} ({100*np.mean(rmask):.0f}%)")
    print("DRY RUN OK — data path ready; run without --dry-run to train.")


def full_run(args):
    import torch
    from model.model import SpinHanceModel, ResNet1DEncoder
    from model.train import TrainConfig, fit

    recs, assignment, report = _load(args)
    print(f"records: {len(recs)} | split {report['counts']} | "
          f"leakage scaffold={report['scaffold_leaks']} dup={report['dup_matrix_leaks']}")

    vocab = DegeneracyVocab()
    enc = None
    if args.small:                       # lighter encoder for the small prelim set
        enc = ResNet1DEncoder(stem_channels=24, stage_channels=(32, 64, 128, 192),
                              blocks_per_stage=(1, 1, 1, 1))
    model = SpinHanceModel(n_groups=8, n_deg_classes=len(vocab), encoder=enc,
                           head_hidden=256 if args.small else 512,
                           dropout=0.2 if args.small else 0.1)

    cfg = TrainConfig(
        batch_size=args.batch, lr=args.lr, epochs=args.epochs,
        stage1_epochs=args.epochs if not args.stage2 else args.stage1_epochs,
        ramp_epochs=args.ramp_epochs, render_subset_frac=args.render_frac,
        weight_decay=args.weight_decay, patience=args.patience, seed=args.seed,
        device=args.device, amp_dtype=args.amp, ckpt_path=args.ckpt,
        s3_ckpt_prefix=args.s3_ckpt_prefix)
    print(f"config: {cfg}")
    npar = sum(p.numel() for p in model.parameters())
    print(f"model params: {npar/1e6:.2f}M | device {cfg.device} | "
          f"stage2 {'ON' if args.stage2 else 'OFF'}")

    fit(recs, assignment, cfg, model=model)
    print("TRAINING COMPLETE — best checkpoint at", cfg.ckpt_path)


def build_parser():
    p = argparse.ArgumentParser(description="SpinHance Task 4 experiment runner")
    p.add_argument("--json", default=str(DEF_JSON))
    p.add_argument("--spectra", default=str(DEF_SPECTRA))
    p.add_argument("--dry-run", action="store_true", help="validate data path only (no torch)")
    p.add_argument("--scaffold", action="store_true", help="enable Bemis-Murcko scaffold split (requires RDKit)")
    p.add_argument("--fields", default="90,600", help="comma-separated MHz fields to require, e.g. 90 or 90,600")
    p.add_argument("--max-mol", type=int, default=0, help="subset N molecules (smoke)")
    p.add_argument("--small", action="store_true", help="lighter encoder for small data")
    p.add_argument("--stage2", action="store_true", help="enable Stage-2 spectral loss")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--stage1-epochs", type=int, default=40)
    p.add_argument("--ramp-epochs", type=int, default=10)
    p.add_argument("--render-frac", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--max-block", type=int, default=2048)
    p.add_argument("--device", default=None)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--ckpt", default="model/checkpoints/spinhance.pt")
    p.add_argument("--s3-ckpt-prefix", default="", help="S3 prefix for per-epoch checkpoints, e.g. s3://bucket/training/session001")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    if args.dry_run:
        dry_run(args)
    else:
        full_run(args)
