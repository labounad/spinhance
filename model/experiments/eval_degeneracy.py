"""Per-degeneracy-class accuracy breakdown for a trained checkpoint.

Aggregate deg-acc is >99% but dominated by d=1; this splits it by true class
(1H, 2H, 3H, 4H, 6H, 9H, 12H, 18H) — recall per class + the confusion (what each
true class gets predicted as). Runs on the bundle's val fold (seed-0 split),
aligning predicted vs true groups by descending shift (same canonical order as
compute_metrics).

    PYTHONPATH=. python model/experiments/eval_degeneracy.py \
        --ckpt <best.pt> --records <bundle>/records.json.gz --parts <bundle> [--sample_n 60000]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from model.architectures import build_architecture
from model.data.records import load_pubchem_records
from model.data.splits import make_splits
from model.data.stacked_spectra import StackedSpectra
from model.data.standardization import DegeneracyVocab, Standardizer
from model.evaluation.metrics import decode, _np_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--sample_n", type=int, default=60000, help="val molecules to evaluate (0=all val)")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    vocab = DegeneracyVocab()
    std = Standardizer().load_state_dict(ck["standardizer"])
    mc = dict(ck["cfg"]["model"]); nm = mc.pop("name")
    model = build_architecture(nm, n_deg_classes=len(vocab), **mc).eval().to(dev)
    model.load_state_dict(ck["model"], strict=False)
    print(f"loaded {nm} on {dev}", flush=True)

    recs = load_pubchem_records(a.records)
    assign, _ = make_splits(recs, seed=0, compute_scaffold=False)
    val = [r for r in recs if assign.get(r["mol_id"]) == "val"]
    if a.sample_n and a.sample_n < len(val):
        val = val[: a.sample_n]
    src = StackedSpectra(a.parts)
    print(f"val molecules: {len(val)}", flush=True)

    counts = defaultdict(int)            # true value -> n
    correct = defaultdict(int)           # true value -> n correct
    conf = defaultdict(lambda: defaultdict(int))   # true -> {pred: n}

    for s in range(0, len(val), a.batch):
        chunk = val[s:s + a.batch]
        spec = torch.tensor(np.stack([np.asarray(src[r["row"]], np.float32) for r in chunk]), device=dev)
        with torch.no_grad():
            out = model(spec)
        dec = decode(_np_pred(out), std, vocab)
        psh, pdg = dec["shifts"], dec["degeneracy"]          # (B,G)
        for bi, r in enumerate(chunk):
            tsh = np.asarray(r["shifts"], float); tdg = np.asarray(r["degeneracy"], int)
            to = np.argsort(-tsh); po = np.argsort(-psh[bi])
            tv = tdg[to]; pv = np.rint(pdg[bi][po]).astype(int)
            for t, p in zip(tv, pv):
                counts[int(t)] += 1; conf[int(t)][int(p)] += 1
                if int(t) == int(p):
                    correct[int(t)] += 1
        if s % (a.batch * 40) == 0:
            print(f"  {s + len(chunk)}/{len(val)}", flush=True)

    print("\n=== per-true-class degeneracy recall ===")
    print(f"{'true':>5} {'n':>8} {'recall':>8}   top confusions (pred:count)")
    tot = sum(counts.values()); bal = []
    for v in sorted(counts):
        n = counts[v]; acc = correct[v] / n
        bal.append(acc)
        top = sorted(conf[v].items(), key=lambda kv: -kv[1])[:4]
        topstr = "  ".join(f"{p}:{c}" for p, c in top)
        print(f"{v:>5} {n:>8} {acc:>8.3f}   {topstr}")
    raw = sum(correct.values()) / tot
    print(f"\nraw acc = {raw:.4f}  | balanced (mean per-class recall) = {np.mean(bal):.4f}  | classes seen = {len(counts)}")


if __name__ == "__main__":
    main()
