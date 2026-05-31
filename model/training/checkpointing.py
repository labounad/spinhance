"""Checkpoint save/load. A checkpoint is self-describing: weights, standardizer
stats, config, epoch, and metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model, standardizer, cfg_dict: dict[str, Any],
                    epoch: int, metrics: dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "standardizer": standardizer.state_dict(),
        "cfg": cfg_dict,
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
