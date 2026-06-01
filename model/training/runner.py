"""
model.training.runner
====================
Glue from a Config to a finished run: load records, split, train.
"""
from __future__ import annotations

from pathlib import Path

from model.data.records import load_records, load_pubchem_records
from model.data.splits import make_splits
from model.training.config import Config
from model.training.trainer import Trainer

REPO = Path(__file__).resolve().parents[2]


def run_from_config(cfg: Config):
    spectra_source = None
    if cfg.data.parts:                       # PubChem 3M+: stacked part_<k>.npy shards
        from model.data.stacked_spectra import StackedSpectra
        recs = load_pubchem_records(cfg.data.records, max_mol=cfg.data.max_mol,
                                    sample_n=cfg.data.sample_n, sample_seed=cfg.data.sample_seed)
        spectra_source = StackedSpectra(cfg.data.parts)
        # records may be fewer than spectra (out-of-vocab molecules filtered), but every
        # record's global row must index into the stacked spectra — catch real mismatch.
        max_row = max((r["row"] for r in recs), default=-1)
        if max_row >= len(spectra_source):
            raise ValueError(f"record row {max_row} >= stacked spectra "
                             f"({len(spectra_source)}) — order/count mismatch")
    else:
        recs = load_records(cfg.data.records, cfg.data.spectra, fields=(cfg.data.field,))
        if cfg.data.max_mol:
            recs = recs[: cfg.data.max_mol]
    assignment, report = make_splits(
        recs, seed=cfg.training.seed,
        compute_scaffold=(cfg.data.split == "scaffold"))
    print(f"[runner] {len(recs)} records | split {report['counts']} | "
          f"scaffold_leaks={report['scaffold_leaks']} dup_leaks={report['dup_matrix_leaks']}")
    return Trainer(cfg, recs, assignment, spectra_source=spectra_source).fit()
