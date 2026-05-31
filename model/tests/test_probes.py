"""
Probe + failure-analysis tests: a tiny trainer run produces the probe artifacts
the dashboards read, plus unit checks on selection and failure tagging.
"""
import json
from pathlib import Path

import numpy as np

from model.evaluation.probes import _probe_indices
from model.evaluation.failure_analysis import _tag_failure, save_failure_cases
from model.training.config import config_from_dict
from model.training.trainer import Trainer
from model.data.splits import make_splits

G, P = 8, 512


def _records(n=80, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 10))
        recs.append(dict(mol_id=f"m{i}", smiles="C", scaffold=f"s{i % 10}",
                         shifts=rng.uniform(0.5, 9, G), couplings=c,
                         degeneracy=rng.choice([1, 2, 3], size=G).astype(int),
                         spec90=rng.random(P).astype(np.float32)))
    return recs


# ── unit ──────────────────────────────────────────────────────────────────────

def test_probe_indices_count_and_unique():
    recs = _records(60)
    idxs = _probe_indices(recs, 12)
    assert len(idxs) == 12 and len(set(idxs)) == 12


def test_tag_failure_priority():
    assert _tag_failure({"shift_mae_ppm": 0.4})["failure_type"] == "large_shift_error"
    assert _tag_failure({"shift_mae_ppm": 0.1, "presence_f1": 0.2,
                         "presence_recall": 0.2})["failure_type"] == "false_negative_couplings"
    assert _tag_failure({"shift_mae_ppm": 0.1})["failure_type"] == "ok"


def test_save_failure_cases_writes_tables(tmp_path):
    rng = np.random.default_rng(0)
    results = [{"mol_id": f"m{i}", "shift_mae_ppm": float(rng.uniform(0, .5)),
                "j_mae_hz": float(rng.uniform(0, 5)), "presence_f1": float(rng.uniform(0, 1)),
                "deg_acc": float(rng.uniform(.5, 1))} for i in range(30)]
    summary = save_failure_cases(results, tmp_path / "run", epoch=3)
    ep = tmp_path / "run" / "probes" / "epoch_0003"
    for f in ("worst_shift_cases.json", "worst_j_cases.json", "failure_summary.json"):
        assert (ep / f).exists()
    assert summary["n_molecules"] == 30


# ── integration: trainer writes probe artifacts ───────────────────────────────

def test_trainer_emits_probe_artifacts(tmp_path):
    recs = _records()
    assignment, _ = make_splits(recs, seed=0, compute_scaffold=False)
    cfg = config_from_dict({
        "run": {"name": "probe_smoke", "output_dir": str(tmp_path / "runs")},
        "data": {"records": "", "spectra": "", "field": 90, "split": "none"},
        "model": {"name": "resnet1d", "size": "tiny", "dropout": 0.0},
        "loss": {"name": "composite", "terms": [{"name": "matrix", "weight": 1.0}]},
        "training": {"epochs": 2, "batch_size": 16, "lr": 1e-3, "amp": "none",
                     "warmup_frac": 0.1, "num_workers": 0, "device": "cpu"},
        "diagnostics": {"enabled": True, "log_every_steps": 5,
                        "probe_every_epochs": 1, "probe_count": 4},
    })
    out = Trainer(cfg, recs, assignment).fit()
    run_dir = Path(out["run_dir"])

    probe_epoch = run_dir / "probes" / "epoch_0000"
    assert probe_epoch.exists(), "no probe epoch dir written"
    for f in ("predictions.json", "probe_metrics.json", "worst_cases.json", "failure_summary.json"):
        assert (probe_epoch / f).exists(), f"missing probe artifact {f}"

    # predictions carry per-molecule true/pred matrices for the inspector
    preds = json.loads((probe_epoch / "predictions.json").read_text())
    assert preds and "true_couplings" in preds[0] and "pred_shifts" in preds[0]

    # summary.json now carries the failure summary + recommendation
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "failure_summary" in summary and "recommendation" in summary

    # a probe metrics row was logged
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any(r["split"] == "probe" for r in rows)
