from pathlib import Path

import torch

from model.train import _do_checkpoint


def test_checkpoint_worker_creates_missing_parent_dirs(tmp_path):
    ckpt = {"value": torch.tensor([1.0])}

    last_path = tmp_path / "canonical" / "checkpoints" / "last.pt"
    best_path = tmp_path / "canonical" / "checkpoints" / "best.pt"
    legacy_path = tmp_path / "legacy" / "missing" / "spinhance.pt"

    _do_checkpoint(
        ckpt,
        str(last_path),
        "",
        str(best_path),
        str(legacy_path),
    )

    assert last_path.exists()
    assert best_path.exists()
    assert legacy_path.exists()

    loaded = torch.load(legacy_path, map_location="cpu")
    assert torch.equal(loaded["value"], ckpt["value"])
