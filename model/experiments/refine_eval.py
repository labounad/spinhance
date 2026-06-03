"""
model.experiments.refine_eval
=============================
Prototype harness for test-time spectral refinement (model.inference.refine).

Runs a checkpoint on the first N held-out molecules, then refines the predicted
shifts against each molecule's own (input) spectrum, and reports shift-MAE and
spectral cosine BEFORE vs AFTER refinement — plus a per-molecule dump for plots.

    python -m model.experiments.refine_eval --run-dir model/runs/<64k run> \
        --test-records $REB/records_3M_test.json.gz --parts $REB/parts \
        --n 40 --device cpu --out refine_eval.json
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from model.architectures import build_architecture
from model.data.records import load_pubchem_records
from model.data.stacked_spectra import StackedSpectra
from model.data.standardization import DegeneracyVocab, Standardizer
from model.evaluation.metrics import decode, _np_pred
from model.renderers._torch_exact import simulate
from model.evaluation.spectral_metrics import cosine_similarity
from model.inference.refine import refine_shifts

G, P = 8, 16384


def shift_mae(a, b):
    """Canonical shift-MAE: compare the two shift vectors sorted descending."""
    return float(np.mean(np.abs(np.sort(a)[::-1] - np.sort(b)[::-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir"); ap.add_argument("--checkpoint")
    ap.add_argument("--test-records", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--field", type=float, default=90.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--trust", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--reg", type=float, default=2.0)
    ap.add_argument("--out", default="refine_eval.json")
    a = ap.parse_args()

    ckpt = a.checkpoint or str(Path(a.run_dir) / "checkpoints" / "best.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    vocab = DegeneracyVocab(); std = Standardizer().load_state_dict(ck["standardizer"])
    mc = dict(ck["cfg"]["model"]); name = mc.pop("name")
    model = build_architecture(name, n_deg_classes=len(vocab), **mc).to(a.device).eval()
    model.load_state_dict(ck["model"], strict=False)

    recs = load_pubchem_records(a.test_records)[:a.n]
    src = StackedSpectra(a.parts)
    print(f"refining {len(recs)} held-out molecules from {Path(ckpt).parent.parent.name}", flush=True)

    rows, t0 = [], time.time()
    sane = None
    for i, r in enumerate(recs):
        spec = np.asarray(src[r["row"]], np.float32)
        t_sh = np.asarray(r["shifts"], float); t_cp = np.asarray(r["couplings"], float)
        t_dg = np.asarray(r["degeneracy"], int)
        with torch.no_grad():
            out = model(torch.tensor(spec, dtype=torch.float32, device=a.device)[None])
        pr = _np_pred(out); pr.pop("soft_equiv", None); dec = decode(pr, std, vocab)
        p_sh, p_cp, p_dg = dec["shifts"][0], dec["couplings"][0], dec["degeneracy"][0]

        # sanity (first mol): torch-exact sim of the TRUE graph should match the target
        if sane is None:
            with torch.no_grad():
                _, sp_true = simulate(torch.tensor(t_sh), torch.tensor(t_cp),
                                      torch.tensor(t_dg), a.field, P)
                sane = cosine_similarity(sp_true[None].double(),
                                         torch.tensor(spec)[None].double()).item()
            print(f"  [sanity] cos(sim(true), target) = {sane:.4f} (≈1 ⇒ refine target consistent)", flush=True)

        refined, info = refine_shifts(p_sh, p_cp, p_dg, spec, field_mhz=a.field,
                                      n_steps=a.steps, trust=a.trust, reg=a.reg)
        rows.append({
            "id": r.get("mol_id"), "smiles": r.get("smiles"),
            "shift_mae_pred": shift_mae(p_sh, t_sh),
            "shift_mae_refined": shift_mae(refined, t_sh),
            "cos0": info["cos0"], "cos1": info["cos1"], "reverted": info["reverted"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(recs)} ({time.time()-t0:.0f}s)", flush=True)

    sm0 = float(np.mean([x["shift_mae_pred"] for x in rows]))
    sm1 = float(np.mean([x["shift_mae_refined"] for x in rows]))
    c0 = float(np.mean([x["cos0"] for x in rows])); c1 = float(np.mean([x["cos1"] for x in rows]))
    improved = sum(x["shift_mae_refined"] < x["shift_mae_pred"] - 1e-4 for x in rows)
    worse = sum(x["shift_mae_refined"] > x["shift_mae_pred"] + 1e-4 for x in rows)
    summary = {"n": len(rows), "shift_mae_pred": round(sm0, 4), "shift_mae_refined": round(sm1, 4),
               "cos_pred": round(c0, 4), "cos_refined": round(c1, 4),
               "n_improved": improved, "n_worse": worse, "n_reverted": sum(x["reverted"] for x in rows),
               "sanity_cos_true": sane}
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=2)
    print("\n=== REFINEMENT (shifts only) ===")
    print(f"  shift MAE: {sm0:.4f} -> {sm1:.4f} ppm  ({100*(sm0-sm1)/sm0:+.1f}%)")
    print(f"  spectral cosine: {c0:.4f} -> {c1:.4f}")
    print(f"  improved {improved}/{len(rows)} | worse {worse} | reverted {summary['n_reverted']}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
