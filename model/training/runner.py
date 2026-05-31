"""
model.training.runner
====================
Glue from a Config to a finished run: load records, split, train.
"""
from __future__ import annotations

from pathlib import Path

from model.data.records import load_records
from model.data.splits import make_splits
from model.training.config import Config
from model.training.trainer import Trainer

REPO = Path(__file__).resolve().parents[2]


def run_from_config(cfg: Config):
    recs = load_records(cfg.data.records, cfg.data.spectra, fields=(cfg.data.field,))
    if cfg.data.max_mol:
        recs = recs[: cfg.data.max_mol]
    assignment, report = make_splits(
        recs, seed=cfg.training.seed,
        compute_scaffold=(cfg.data.split == "scaffold"))
    print(f"[runner] {len(recs)} records | split {report['counts']} | "
          f"scaffold_leaks={report['scaffold_leaks']} dup_leaks={report['dup_matrix_leaks']}")
    return Trainer(cfg, recs, assignment).fit()
