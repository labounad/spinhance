"""
model.experiments.train_surrogate
=================================
Config-driven entrypoint for the surrogate matrix->spectrum renderer (Branch 5).

    PYTHONPATH=. python -m model.experiments.train_surrogate --config model/configs/surrogate.yaml
    PYTHONPATH=. python -m model.experiments.train_surrogate --config model/configs/surrogate.yaml \
        --set training.epochs=2 --set data.max_mol=256 --set run.name=smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from model.training.config import _apply_override
from model.training.surrogate import SurrogateTrainer


def main():
    ap = argparse.ArgumentParser(description="SpinHance surrogate renderer trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=value")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    for ov in args.overrides:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got '{ov}'")
        k, v = ov.split("=", 1)
        _apply_override(cfg, k.strip(), v.strip())
    out = SurrogateTrainer(cfg).fit()
    print("SURROGATE TRAINING COMPLETE —", out["run_dir"])


if __name__ == "__main__":
    main()
