"""
model.experiments.train
=======================
Config-driven training entrypoint.

    PYTHONPATH=. python -m model.experiments.train --config model/configs/baseline_matrix.yaml
    PYTHONPATH=. python -m model.experiments.train --config model/configs/baseline_matrix.yaml \
        --set training.epochs=2 --set run.name=smoke
"""
from __future__ import annotations

import argparse

from model.training.config import load_config
from model.training.runner import run_from_config


def main():
    ap = argparse.ArgumentParser(description="SpinHance config-driven trainer")
    ap.add_argument("--config", required=True, help="path to a YAML run config")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="key=value", help="dotted config override (repeatable)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = run_from_config(cfg)
    print("TRAINING COMPLETE —", out["run_dir"])


if __name__ == "__main__":
    main()
