"""
Surrogate training smoke: dataset/collate, and a tiny end-to-end SurrogateTrainer
run on synthetic in-memory (matrix, spectrum) pairs — asserting the canonical run
dir + that the W1 loss decreases. No pyspin simulation or data files needed
(records carry in-memory spec90/spec600 arrays).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.data.surrogate_dataset import SurrogateSpectrumDataset, make_surrogate_collate
from model.training.surrogate import SurrogateTrainer

G, P = 8, 1024


def _synthetic_records(n=96, seed=0):
    """Records with a physical matrix + an in-memory 'spectrum' that's a smooth
    function of the matrix (so the surrogate has a learnable signal)."""
    rng = np.random.default_rng(seed)
    ppm = np.linspace(0, 12, P)
    recs = []
    for i in range(n):
        shifts = rng.uniform(0.5, 9, G)
        deg = rng.choice([1, 2, 3], size=G).astype(int)
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.3:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 8))
        # toy target: sum of Lorentzians at the shifts, area ~ degeneracy
        def spec(field):
            y = np.zeros(P)
            hw = (1.0 / 2.0) / field * 50  # wide-ish so the toy is smooth
            for s, d in zip(shifts, deg):
                y += d / (1 + ((ppm - s) / max(hw, 1e-3)) ** 2)
            y = np.clip(y, 0, None)
            return (y / (y.sum() * (12.0 / P) + 1e-12)).astype(np.float32)
        recs.append(dict(mol_id=f"m{i}", smiles="C", scaffold=f"s{i%8}",
                         shifts=shifts, couplings=c, degeneracy=deg,
                         spec90=spec(90.0), spec600=spec(600.0)))
    return recs


def test_dataset_and_collate():
    recs = _synthetic_records(8)
    ds = SurrogateSpectrumDataset(recs, fields=(90, 600))
    item = ds[0]
    assert item["shifts"].shape == (G,) and item["couplings"].shape == (G, G)
    assert item["spec90"].shape == (P,) and item["spec600"].shape == (P,)
    dl = DataLoader(ds, batch_size=4, collate_fn=make_surrogate_collate((90, 600)))
    b = next(iter(dl))
    assert b["shifts"].shape == (4, G) and b["spec600"].shape == (4, P)


def test_surrogate_train_smoke(tmp_path, monkeypatch):
    # records via in-memory arrays; bypass load_records by injecting the split data.
    recs = _synthetic_records(96)
    cfg = {
        "run": {"name": "surr_smoke", "output_dir": str(tmp_path / "runs"),
                "dir": str(tmp_path / "runs" / "surr_smoke")},
        "data": {"records": "", "spectra": "", "fields": [90, 600], "split": "none"},
        "model": {"name": "surrogate", "dim": 32, "depth": 2, "heads": 2,
                  "sticks_per_group": 16, "points": P},
        "loss": {"mse_weight": 0.3},
        "training": {"epochs": 3, "batch_size": 16, "lr": 1e-3, "amp": "none",
                     "warmup_frac": 0.1, "num_workers": 0, "device": "cpu", "save_every": 1},
        "diagnostics": {"enabled": True, "log_every_steps": 1},
    }
    tr = SurrogateTrainer(cfg)
    # inject synthetic records instead of load_records (which needs files)
    from model.data import surrogate_dataset as SD
    from model.data.splits import make_splits

    def fake_build():
        assignment, report = make_splits(recs, seed=0, compute_scaffold=False)
        by = {"train": [], "val": []}
        for r in recs:
            f = assignment.get(r["mol_id"])
            if f in by:
                by[f].append(r)
        ds = {k: SD.SurrogateSpectrumDataset(v, fields=(90, 600)) for k, v in by.items()}
        from model.renderers import build_renderer
        model = build_renderer("surrogate", dim=32, depth=2, heads=2,
                               sticks_per_group=16, points=P)
        return ds, model, report

    monkeypatch.setattr(tr, "_build", fake_build)
    out = tr.fit()

    run_dir = Path(out["run_dir"])
    for f in ("config.json", "status.json", "metrics.jsonl", "summary.json"):
        assert (run_dir / f).exists(), f"missing {f}"
    assert (run_dir / "checkpoints" / "best.pt").exists()
    assert (run_dir / "checkpoints" / "epoch_0000.pt").exists()

    # W1 loss should drop from first to last logged train_step
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    steps = [r["metrics"]["w1"] for r in rows if r.get("split") == "train_step"]
    assert steps[-1] < steps[0], f"W1 did not improve: {steps[0]:.3f} -> {steps[-1]:.3f}"

    # validation metrics present for both fields
    val = [r for r in rows if r.get("split") == "val"]
    assert val and "w1_90" in val[-1]["metrics"] and "w1_600" in val[-1]["metrics"]
