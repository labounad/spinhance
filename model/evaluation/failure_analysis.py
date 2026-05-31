"""
model.evaluation.failure_analysis
=================================
Per-sample validation evaluation + worst-case failure tables (ported, adapted to
the typed contract). Saved alongside probe artifacts so the dashboard and AutoAI
can diagnose WHY a run underperforms, not just HOW MUCH. Filesystem-only.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.data.collate import collate_spin_batch
from model.evaluation.metrics import compute_metrics, _np_pred, _np_target

__all__ = ["per_sample_evaluate", "save_failure_cases"]


# ── Failure tagging ────────────────────────────────────────────────────────────

def _tag_failure(r: dict) -> dict:
    shift = r.get("shift_mae_ppm", 0.0)
    j = r.get("j_mae_hz", 0.0)
    f1 = r.get("presence_f1", 1.0)
    rec = r.get("presence_recall", f1 + 1e-9)
    deg = r.get("deg_acc", 1.0)
    if shift > 0.25:
        tag = "large_shift_error"
    elif f1 < 0.4 and rec < 0.5:
        tag = "false_negative_couplings"
    elif f1 < 0.4:
        tag = "false_positive_couplings"
    elif j > 3.5:
        tag = "bad_j_magnitude"
    elif deg < 0.75:
        tag = "wrong_degeneracy"
    else:
        tag = "ok"
    return {**r, "failure_type": tag}


# ── Per-sample evaluation ──────────────────────────────────────────────────────

def per_sample_evaluate(model, val_records, val_dataset, batch_size, std, vocab,
                        device, amp_ctx) -> list[dict]:
    """Run the model over the val set; one metric dict per sample (in order)."""
    dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_spin_batch, num_workers=0,
                    pin_memory=(device != "cpu"))
    model.eval()
    results: list[dict] = []
    idx = 0
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device, non_blocking=True)
            with amp_ctx():
                out = model(batch)
            pred_np = _np_pred(out)
            tgt_np = _np_target(batch)
            B = pred_np["shifts"].shape[0]
            for b in range(B):
                if idx >= len(val_records):
                    break
                met = compute_metrics({k: pred_np[k][b:b + 1] for k in pred_np},
                                      {k: tgt_np[k][b:b + 1] for k in tgt_np},
                                      std, vocab)
                rec = val_records[idx]
                met["mol_id"] = rec.get("mol_id", f"mol_{idx:06d}")
                met["smiles"] = rec.get("smiles", "")
                results.append(met)
                idx += 1
    return results


# ── Failure tables ─────────────────────────────────────────────────────────────

def save_failure_cases(per_sample_results, run_dir, epoch: int, n_worst: int = 32) -> dict:
    """Write worst-case JSON tables + failure_summary.json; return the summary."""
    epoch_dir = Path(run_dir) / "probes" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    tagged = [_tag_failure(r) for r in per_sample_results]

    for metric, fname, descending in [
        ("shift_mae_ppm", "worst_shift_cases.json", True),
        ("j_mae_hz", "worst_j_cases.json", True),
        ("deg_acc", "worst_deg_cases.json", False),
        ("presence_f1", "worst_presence_cases.json", False),
    ]:
        worst = sorted(tagged, key=lambda r: r.get(metric, 0.0), reverse=descending)[:n_worst]
        (epoch_dir / fname).write_text(json.dumps(worst, indent=2))

    counts = Counter(r.get("failure_type", "unknown") for r in tagged)
    summary = {
        "dominant_failure": counts.most_common(1)[0][0] if counts else "none",
        "n_ok": counts.get("ok", 0),
        "failure_distribution": dict(counts.most_common()),
        "n_molecules": len(tagged),
    }
    (epoch_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
