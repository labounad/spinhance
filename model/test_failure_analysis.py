"""
Tests for model.failure_analysis.  Torch-free; uses only stdlib + numpy.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from model.failure_analysis import _tag_failure, save_failure_cases


# ── _tag_failure ───────────────────────────────────────────────────────────────

def test_tag_large_shift():
    r = _tag_failure({"shift_mae_ppm": 0.3, "j_mae_hz": 1.0,
                      "presence_f1": 0.9, "deg_acc": 0.9})
    assert r["failure_type"] == "large_shift_error"


def test_tag_false_negative_couplings():
    r = _tag_failure({"shift_mae_ppm": 0.1, "j_mae_hz": 1.0,
                      "presence_f1": 0.3, "presence_recall": 0.2, "deg_acc": 0.9})
    assert r["failure_type"] == "false_negative_couplings"


def test_tag_false_positive_couplings():
    # low F1 but high recall means model predicts many couplings that don't exist
    r = _tag_failure({"shift_mae_ppm": 0.1, "j_mae_hz": 1.0,
                      "presence_f1": 0.3, "presence_recall": 0.95, "deg_acc": 0.9})
    assert r["failure_type"] == "false_positive_couplings"


def test_tag_bad_j_magnitude():
    r = _tag_failure({"shift_mae_ppm": 0.1, "j_mae_hz": 4.0,
                      "presence_f1": 0.85, "deg_acc": 0.9})
    assert r["failure_type"] == "bad_j_magnitude"


def test_tag_wrong_degeneracy():
    r = _tag_failure({"shift_mae_ppm": 0.1, "j_mae_hz": 1.0,
                      "presence_f1": 0.85, "deg_acc": 0.5})
    assert r["failure_type"] == "wrong_degeneracy"


def test_tag_ok():
    r = _tag_failure({"shift_mae_ppm": 0.1, "j_mae_hz": 1.0,
                      "presence_f1": 0.85, "deg_acc": 0.9})
    assert r["failure_type"] == "ok"


def test_tag_preserves_all_input_fields():
    inp = {"shift_mae_ppm": 0.4, "mol_id": "mol_001", "smiles": "CCO",
           "deg_acc": 0.9, "j_mae_hz": 1.0, "presence_f1": 0.9}
    r = _tag_failure(inp)
    for k in inp:
        assert k in r, f"field '{k}' was dropped by _tag_failure"


def test_tag_shift_takes_priority_over_deg():
    # shift > 0.25 threshold fires before wrong_degeneracy
    r = _tag_failure({"shift_mae_ppm": 0.3, "j_mae_hz": 1.0,
                      "presence_f1": 0.9, "deg_acc": 0.4})
    assert r["failure_type"] == "large_shift_error"


def test_tag_missing_fields_defaults():
    # Graceful with partial input — defaults should not crash
    r = _tag_failure({"shift_mae_ppm": 0.1})
    assert "failure_type" in r


# ── save_failure_cases ─────────────────────────────────────────────────────────

def _make_results(n: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {
            "mol_id":         f"mol_{i:06d}",
            "smiles":         "CCO",
            "shift_mae_ppm":  float(rng.uniform(0.03, 0.5)),
            "j_mae_hz":       float(rng.uniform(0.3, 6.0)),
            "presence_f1":    float(rng.uniform(0.2, 1.0)),
            "presence_recall": float(rng.uniform(0.2, 1.0)),
            "deg_acc":        float(rng.uniform(0.4, 1.0)),
        }
        for i in range(n)
    ]


def test_creates_all_expected_files(tmp_path):
    run_dir = tmp_path / "run"
    save_failure_cases(_make_results(), run_dir, epoch=7, n_worst=10)
    ep = run_dir / "probes" / "epoch_0007"
    for fname in (
        "worst_shift_cases.json", "worst_j_cases.json",
        "worst_deg_cases.json", "worst_presence_cases.json",
        "failure_summary.json",
    ):
        assert (ep / fname).exists(), f"missing {fname}"


def test_worst_shift_sorted_descending(tmp_path):
    save_failure_cases(_make_results(50), tmp_path / "run", epoch=0, n_worst=20)
    worst = json.loads(
        (tmp_path / "run" / "probes" / "epoch_0000" / "worst_shift_cases.json").read_text()
    )
    shifts = [r["shift_mae_ppm"] for r in worst]
    assert shifts == sorted(shifts, reverse=True)


def test_worst_deg_sorted_ascending(tmp_path):
    # Lower deg_acc is worse; worst_deg_cases should be sorted ascending
    save_failure_cases(_make_results(40), tmp_path / "run", epoch=0)
    worst = json.loads(
        (tmp_path / "run" / "probes" / "epoch_0000" / "worst_deg_cases.json").read_text()
    )
    accs = [r["deg_acc"] for r in worst]
    assert accs == sorted(accs)


def test_worst_presence_sorted_ascending(tmp_path):
    save_failure_cases(_make_results(40), tmp_path / "run", epoch=0)
    worst = json.loads(
        (tmp_path / "run" / "probes" / "epoch_0000" / "worst_presence_cases.json").read_text()
    )
    f1s = [r["presence_f1"] for r in worst]
    assert f1s == sorted(f1s)


def test_failure_summary_structure(tmp_path):
    results = _make_results(40)
    summary = save_failure_cases(results, tmp_path / "run", epoch=0)
    assert "dominant_failure" in summary
    assert "failure_distribution" in summary
    assert "n_molecules" in summary
    assert summary["n_molecules"] == 40


def test_failure_distribution_sums_to_n(tmp_path):
    n = 35
    summary = save_failure_cases(_make_results(n), tmp_path / "run", epoch=0)
    total = sum(summary["failure_distribution"].values())
    assert total == n


def test_n_worst_is_capped(tmp_path):
    save_failure_cases(_make_results(10), tmp_path / "run", epoch=0, n_worst=32)
    worst = json.loads(
        (tmp_path / "run" / "probes" / "epoch_0000" / "worst_shift_cases.json").read_text()
    )
    assert len(worst) == 10   # only 10 records total


def test_epoch_dir_naming(tmp_path):
    save_failure_cases(_make_results(), tmp_path / "run", epoch=42)
    assert (tmp_path / "run" / "probes" / "epoch_0042").exists()


def test_failure_tags_present_in_worst_cases(tmp_path):
    save_failure_cases(_make_results(30), tmp_path / "run", epoch=0)
    worst = json.loads(
        (tmp_path / "run" / "probes" / "epoch_0000" / "worst_shift_cases.json").read_text()
    )
    for entry in worst:
        assert "failure_type" in entry


def test_empty_results_does_not_crash(tmp_path):
    # Edge case: empty val set
    summary = save_failure_cases([], tmp_path / "run", epoch=0)
    assert summary["n_molecules"] == 0
