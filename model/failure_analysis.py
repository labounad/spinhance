"""
model.failure_analysis
======================
Per-sample validation evaluation and worst-case failure tables.
Saved alongside probe artifacts so the live dashboard and autoai/ can
diagnose WHY a run is underperforming, not just HOW MUCH.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset import collate_fn
from model.metrics import compute_metrics


# ── Failure tagging ────────────────────────────────────────────────────────────

def _tag_failure(r: dict) -> dict:
    """Assign a primary failure tag based on per-sample metrics."""
    shift = r.get("shift_mae_ppm", 0.0)
    j     = r.get("j_mae_hz", 0.0)
    f1    = r.get("presence_f1", 1.0)
    rec   = r.get("presence_recall", 1.0) if "presence_recall" in r else (f1 + 1e-9)
    deg   = r.get("deg_acc", 1.0)

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

def per_sample_evaluate(
    model,
    val_records: list[dict],
    val_dataset,
    cfg,
    std,
    vocab,
    device: str,
    amp_ctx,
    balance: dict,
) -> list[dict]:
    """Run the model on the val set; return one metric dict per sample."""
    dl = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=(device != "cpu"),
    )
    model.eval()
    results: list[dict] = []
    sample_idx = 0

    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with amp_ctx():
                pred = model(batch["spectrum"])
            B = batch["spectrum"].shape[0]
            pred_np = {k: pred[k].float().cpu().numpy() for k in pred}
            tgt_np  = {k: batch[k].cpu().numpy()
                       for k in ("shifts", "j_mag", "j_presence", "deg_class")}
            for b in range(B):
                if sample_idx >= len(val_records):
                    break
                met = compute_metrics(
                    {k: pred_np[k][b:b+1] for k in pred_np},
                    {k: tgt_np[k][b:b+1]  for k in tgt_np},
                    std, vocab,
                )
                rec = val_records[sample_idx]
                met["mol_id"] = rec.get("mol_id", f"mol_{sample_idx:06d}")
                met["smiles"] = rec.get("smiles", "")
                results.append(met)
                sample_idx += 1

    return results


# ── Failure tables ─────────────────────────────────────────────────────────────

def save_failure_cases(
    per_sample_results: list[dict],
    run_dir: Path,
    epoch: int,
    n_worst: int = 32,
) -> dict:
    """Save worst-case JSON tables and a failure_summary.json."""
    epoch_dir = run_dir / "probes" / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    tagged = [_tag_failure(r) for r in per_sample_results]

    for metric, fname, descending in [
        ("shift_mae_ppm", "worst_shift_cases.json",    True),
        ("j_mae_hz",      "worst_j_cases.json",        True),
        ("deg_acc",       "worst_deg_cases.json",      False),
        ("presence_f1",   "worst_presence_cases.json", False),
    ]:
        worst = sorted(tagged, key=lambda r: r.get(metric, 0.0), reverse=descending)[:n_worst]
        (epoch_dir / fname).write_text(json.dumps(worst, indent=2))

    counts   = Counter(r.get("failure_type", "unknown") for r in tagged)
    dominant = counts.most_common(1)[0][0] if counts else "none"
    summary  = {
        "dominant_failure":     dominant,
        "n_ok":                 counts.get("ok", 0),
        "failure_distribution": dict(counts.most_common()),
        "n_molecules":          len(tagged),
    }
    (epoch_dir / "failure_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
