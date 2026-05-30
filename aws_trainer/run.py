"""
aws_trainer.run
================
Entry point for single-GPU and multi-GPU (torchrun) training.

Examples
--------
# Validate data path (no torch needed):
    PYTHONPATH=. python -m aws_trainer.run --dry-run

# Single GPU, medium model:
    PYTHONPATH=. python -m aws_trainer.run --model-size medium

# 4-GPU DDP:
    PYTHONPATH=. torchrun --nproc_per_node=4 -m aws_trainer.run --model-size medium

# From a saved config (with optional overrides):
    PYTHONPATH=. python -m aws_trainer.run --config aws_trainer/configs/medium_s2.json \
        --epochs 120 --batch 512

# Stage 1 only (fast sanity check):
    PYTHONPATH=. python -m aws_trainer.run --small --no-stage2 --epochs 30 --batch 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.data_adapter import load_records, renderable_mask
from model.splits import make_splits
from model.targets import DegeneracyVocab, Standardizer, encode_target
from aws_trainer.config import VAWSConfig


# ── Dry-run (torch-free validation) ──────────────────────────────────────────

def dry_run(cfg: VAWSConfig) -> None:
    recs = load_records(cfg.json_path, cfg.spectra_root, fields=(90,))
    print(f"records loaded: {len(recs)}", flush=True)
    if not recs:
        print("ERROR: 0 records — check json_path and spectra_root", flush=True)
        sys.exit(1)

    assignment, report = make_splits(
        recs, ratios=cfg.split_ratios, seed=cfg.split_seed,
        compute_scaffold=cfg.scaffold_split)
    print(f"split: {report['counts']} (ratios "
          f"tr={report['ratios']['train']:.2f} "
          f"va={report['ratios']['val']:.2f} "
          f"te={report['ratios']['test']:.2f})")
    print(f"leakage: scaffold={report['scaffold_leaks']} "
          f"dup_matrix={report['dup_matrix_leaks']} | groups={report['n_groups']}")

    vocab = DegeneracyVocab()
    train = [r for r in recs if assignment[r["mol_id"]] == "train"]
    std = Standardizer().fit(train, vocab)
    print(f"standardizer(train): shift {std.shift_mean:.2f}±{std.shift_std:.2f} ppm | "
          f"J {std.j_mean:.2f}±{std.j_std:.2f} Hz")

    n_bad = sum(1 for r in recs if _bad_target(r, vocab))
    print(f"target encoding: {len(recs) - n_bad}/{len(recs)} OK (vocab miss: {n_bad})")

    rmask = renderable_mask(recs, max_block=2048)
    print(f"Stage-2 renderable: {sum(rmask)}/{len(recs)} ({100 * np.mean(rmask):.0f}%)")
    print("DRY RUN OK")


def _bad_target(r: dict, vocab: DegeneracyVocab) -> bool:
    try:
        encode_target(r["shifts"], r["couplings"], r["degeneracy"], vocab)
        return False
    except KeyError:
        return True


# ── Full training run ─────────────────────────────────────────────────────────

def full_run(cfg: VAWSConfig) -> None:
    from model.data_adapter import load_records
    from model.splits import make_splits
    from aws_trainer.train import fit

    recs = load_records(cfg.json_path, cfg.spectra_root, fields=(90,))
    if not recs:
        print("ERROR: 0 records — run with --dry-run to diagnose", flush=True)
        sys.exit(1)

    assignment, report = make_splits(
        recs, ratios=cfg.split_ratios, seed=cfg.split_seed,
        compute_scaffold=cfg.scaffold_split)

    from aws_trainer.train import _is_main
    if _is_main():
        print(f"records: {len(recs)} | split {report['counts']} | "
              f"leakage scaffold={report['scaffold_leaks']} "
              f"dup={report['dup_matrix_leaks']}", flush=True)
        cfg.to_json(Path(cfg.ckpt_dir) / "config.json")

    fit(recs, assignment, cfg)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="aws_trainer — production SpinHance trainer")
    p.add_argument("--config",         default=None,        help="JSON config file (base)")
    p.add_argument("--dry-run",        action="store_true", help="validate data path, no training")

    # Data
    p.add_argument("--json",     default=None, dest="json",    help="spin_systems.json path")
    p.add_argument("--spectra",  default=None,                 help="spectra root dir")
    p.add_argument("--no-scaffold", action="store_true",       help="skip RDKit scaffold split")
    p.add_argument("--no-preload",  action="store_true",       help="disable RAM preload")
    p.add_argument("--workers",  type=int, default=None,       help="DataLoader num_workers")

    # Model
    p.add_argument("--model-size", default=None,
                   choices=["tiny", "small", "medium", "large",
                             "medium-attn", "large-attn"])
    p.add_argument("--small",      action="store_true",    help="shorthand for --model-size tiny")

    # Training
    p.add_argument("--batch",          type=int,   default=None)
    p.add_argument("--accum",          type=int,   default=None, help="gradient accumulation steps")
    p.add_argument("--lr",             type=float, default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--stage1-epochs",  type=int,   default=None)
    p.add_argument("--ramp-epochs",    type=int,   default=None)
    p.add_argument("--render-frac",    type=float, default=None)
    p.add_argument("--no-stage2",      action="store_true", help="disable Stage-2 spectral loss")
    p.add_argument("--amp",            default=None, choices=["bf16", "fp16", "none"])
    p.add_argument("--compile",        action="store_true", help="enable torch.compile")

    # Logging / checkpointing
    p.add_argument("--ckpt-dir",       default=None)
    p.add_argument("--run-name",       default=None)
    p.add_argument("--wandb-project",  default=None)

    # Misc
    p.add_argument("--seed",           type=int, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # --small is a shorthand
    if args.small and args.model_size is None:
        args.model_size = "tiny"

    cfg = VAWSConfig.from_args(args)

    if args.dry_run:
        dry_run(cfg)
    else:
        full_run(cfg)


if __name__ == "__main__":
    main()
