"""
model.experiments.eval_surrogate
================================
Headless test-set evaluation for the trained differentiable surrogate renderer
(Branch 5). Rebuilds the SAME held-out test fold the surrogate trained against
(molecule-level split, identical seed), simulates the pyspin ground-truth
spectra on the fly (parallel), runs the frozen surrogate, and reports the mean
fidelity across all test molecules — W1 and cosine, per field and overall.

Metrics match the trainer exactly: both spectra are unit-integral, W1 uses
dx = (PPM_TO - PPM_FROM) / points (model resolution).

    python -m model.experiments.eval_surrogate \
        --checkpoint /tmp/spinhance_eval/session012_best.pt \
        --records mol_to_spin_system/data/spin_systems_chembl_8spin_randomized.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from model.data.splits import make_splits
from model.evaluation.spectral_metrics import wasserstein1, cosine_similarity
from model.renderers import build_renderer
from model.schemas.constants import PPM_FROM, PPM_TO
from simulation.graph_io import read_spin_systems, record_to_arrays


# ── ground truth (worker) ──────────────────────────────────────────────────────

_POINTS = 16384


def _init_worker(points):
    global _POINTS
    _POINTS = points


def _simulate(args):
    """Simulate unit-integral 90+600 MHz spectra for one molecule."""
    shifts, couplings, deg = args
    from simulation.pyspin.composite import simulate_spectrum_composite
    out = {}
    for field in (90, 600):
        _, y = simulate_spectrum_composite(np.asarray(shifts), np.asarray(couplings),
                                           list(deg), float(field), points=_POINTS)
        out[field] = np.asarray(y, dtype=np.float32)
    return out


# ── eval ───────────────────────────────────────────────────────────────────────

def load_test_records(records_json, seed):
    recs = []
    for idx, rec in read_spin_systems(records_json):
        _, shifts, couplings, deg = record_to_arrays(rec)
        recs.append({"mol_id": f"mol_{idx:06d}",
                     "shifts": np.asarray(shifts, float),
                     "couplings": np.asarray(couplings, float),
                     "degeneracy": np.asarray(deg, int)})
    assignment, report = make_splits(recs, seed=seed, compute_scaffold=False)
    test = [r for r in recs if assignment.get(r["mol_id"]) == "test"]
    return test, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--limit", type=int, default=0, help="cap test molecules (0 = all)")
    ap.add_argument("--out", default="", help="optional json path for the report")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    seed = int((cfg.get("training", {}) or {}).get("seed", 0))
    mcfg = {k: v for k, v in (cfg.get("model", {}) or {}).items() if k != "name"}
    model = build_renderer("surrogate", **mcfg).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    points = int(getattr(model, "points", _POINTS))
    dx = (PPM_TO - PPM_FROM) / points

    test, report = load_test_records(args.records, seed)
    if args.limit:
        test = test[: args.limit]
    print(f"[eval] test molecules: {len(test):,}  (split seed={seed}, "
          f"counts={report['counts']})  points={points}  device={args.device}")

    sims = [(r["shifts"], r["couplings"], r["degeneracy"]) for r in test]
    nw = args.workers or None
    gts = [None] * len(sims)
    print("[eval] simulating pyspin ground truth (90+600 MHz)…")
    with ProcessPoolExecutor(max_workers=nw, initializer=_init_worker,
                             initargs=(points,)) as ex:
        for i, gt in enumerate(ex.map(_simulate, sims, chunksize=8)):
            gts[i] = gt
            if (i + 1) % 500 == 0:
                print(f"   {i + 1}/{len(sims)}")

    # surrogate inference + metrics, batched, per field
    agg = {f: {"w1": [], "cos": []} for f in (90, 600)}
    bs = 256
    for f in (90, 600):
        for s0 in range(0, len(test), bs):
            chunk = test[s0:s0 + bs]
            shifts = torch.tensor(np.stack([r["shifts"] for r in chunk]), dtype=torch.float32, device=args.device)
            cpl = torch.tensor(np.stack([r["couplings"] for r in chunk]), dtype=torch.float32, device=args.device)
            deg = torch.tensor(np.stack([r["degeneracy"] for r in chunk]), dtype=torch.float32, device=args.device)
            tgt = torch.tensor(np.stack([gts[s0 + j][f] for j in range(len(chunk))]),
                               dtype=torch.float32, device=args.device)
            with torch.no_grad():
                pred = model(shifts, cpl, deg, float(f))
            agg[f]["w1"].extend(wasserstein1(pred, tgt, dx=dx).cpu().tolist())
            agg[f]["cos"].extend(cosine_similarity(pred, tgt).cpu().tolist())

    def stats(x):
        a = np.asarray(x)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}

    out = {"checkpoint": args.checkpoint, "n_test": len(test), "seed": seed,
           "points": points, "per_field": {}, "split_counts": report["counts"]}
    w1_means, cos_means = [], []
    for f in (90, 600):
        w1s, css = stats(agg[f]["w1"]), stats(agg[f]["cos"])
        out["per_field"][f] = {"w1": w1s, "cosine": css}
        w1_means.append(w1s["mean"]); cos_means.append(css["mean"])
    out["overall"] = {"w1_mean": float(np.mean(w1_means)),
                      "cosine_mean": float(np.mean(cos_means))}

    print("\n========== SURROGATE TEST-SET FIDELITY ==========")
    print(f"  test molecules : {len(test):,}")
    for f in (90, 600):
        w, c = out["per_field"][f]["w1"], out["per_field"][f]["cosine"]
        print(f"  {f:>3} MHz : W1 {w['mean']:.4f} ± {w['std']:.4f} "
              f"(median {w['median']:.4f}, p90 {w['p90']:.4f})  |  "
              f"cosine {c['mean']:.4f} ± {c['std']:.4f}")
    print(f"  OVERALL : mean W1 {out['overall']['w1_mean']:.4f}  |  "
          f"mean cosine {out['overall']['cosine_mean']:.4f}")
    print("=================================================")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[eval] report -> {args.out}")


if __name__ == "__main__":
    main()
