"""
End-to-end smoke: build a tiny Config, train 2 epochs on synthetic in-memory
records, and assert the canonical run-directory artifact contract. Also covers
config loading + dotted overrides and the diagnostics writer contract.
CPU only, no data files.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from model.training.config import config_from_dict, load_config
from model.training.trainer import Trainer
from model.data.splits import make_splits

G, P = 8, 512


def _records(n=96, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 10))
        recs.append(dict(
            mol_id=f"m{i}", smiles="C", scaffold=f"s{i % 12}",
            shifts=rng.uniform(0.5, 9, G), couplings=c,
            degeneracy=rng.choice([1, 2, 3], size=G).astype(int),
            spec90=rng.random(P).astype(np.float32),
        ))
    return recs


def _tiny_config(tmp_path) -> "Config":
    return config_from_dict({
        "run": {"name": "smoke", "output_dir": str(tmp_path / "runs")},
        "data": {"records": "", "spectra": "", "field": 90, "split": "none"},
        "model": {"name": "resnet1d", "size": "tiny", "dropout": 0.0},
        "loss": {"name": "composite", "terms": [{"name": "matrix", "weight": 1.0}]},
        "training": {"epochs": 2, "batch_size": 16, "lr": 1e-3, "amp": "none",
                     "warmup_frac": 0.1, "patience": 5, "num_workers": 0, "device": "cpu"},
        "diagnostics": {"enabled": True, "log_every_steps": 1,
                        "probe_every_epochs": 1, "probe_count": 4},
    })


def test_smoke_train_produces_canonical_run_dir(tmp_path):
    recs = _records()
    assignment, _ = make_splits(recs, seed=0, compute_scaffold=False)
    cfg = _tiny_config(tmp_path)
    out = Trainer(cfg, recs, assignment).fit()

    run_dir = Path(out["run_dir"])
    assert run_dir.exists()
    for name in ("config.json", "status.json", "metrics.jsonl", "events.jsonl", "summary.json"):
        assert (run_dir / name).exists(), f"missing {name}"
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "last.pt").exists()
    # per-epoch snapshots (save_every default = 1) so the viewer can load any epoch
    assert (run_dir / "checkpoints" / "epoch_0000.pt").exists()
    assert (run_dir / "checkpoints" / "epoch_0001.pt").exists()


def test_smoke_status_and_summary_contents(tmp_path):
    recs = _records()
    assignment, _ = make_splits(recs, seed=0, compute_scaffold=False)
    out = Trainer(_tiny_config(tmp_path), recs, assignment).fit()
    run_dir = Path(out["run_dir"])

    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "finished"
    assert status["epochs"] == 2

    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["state"] == "finished"
    assert "best_metrics" in summary
    # val metrics include the Hungarian-matched fields
    assert "h_shift_mae_ppm" in summary["best_metrics"]


def test_smoke_metrics_jsonl_has_train_and_val_rows(tmp_path):
    recs = _records()
    assignment, _ = make_splits(recs, seed=0, compute_scaffold=False)
    out = Trainer(_tiny_config(tmp_path), recs, assignment).fit()
    rows = [json.loads(l) for l in (Path(out["run_dir"]) / "metrics.jsonl").read_text().splitlines()]
    splits = {r["split"] for r in rows}
    assert {"train_step", "train", "val"} <= splits


def test_config_load_and_override(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "run:\n  name: base\ndata:\n  records: r\n  spectra: s\n"
        "training:\n  epochs: 80\n  batch_size: 128\n")
    cfg = load_config(cfg_path, ["training.epochs=3", "run.name=smoke", "training.amp=none"])
    assert cfg.training.epochs == 3
    assert cfg.run.name == "smoke"
    assert cfg.training.amp == "none"
    assert cfg.data.records == "r"


def test_config_rejects_unknown_key(tmp_path):
    with pytest.raises(ValueError):
        config_from_dict({"training": {"epochsss": 5}})
