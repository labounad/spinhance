"""Generate docs/data/{learning_curves,test_eval}.json from the v2 fleet run dirs.
Reads metrics.jsonl (val rows) + heldout_eval.json via autoai.run_reader."""
import glob, json, os, re
from autoai import run_reader as rr

RUNS = "model/runs"
CFGS = ["025", "026", "027", "028", "029"]
TIERS = ["64k", "500k", "3M"]

def latest_run(cfg, tier):
    pat = f"{RUNS}/*_v2_{cfg}_{tier}_*"
    ds = sorted(glob.glob(pat))
    return ds[-1] if ds else None

def val_series(d):
    out = []
    for r in rr.read_metrics(d):
        if r.get("split") != "val": continue
        m = r.get("metrics", {})
        if "shift_mae_ppm" not in m: continue
        out.append({"epoch": r.get("epoch", len(out)),
                    "shift": round(m["shift_mae_ppm"], 4),
                    "j": round(m.get("j_mae_hz", 0), 3),
                    "f1": round(m.get("presence_f1", 0), 4),
                    "deg": round(m.get("deg_acc_balanced", 0), 4)})
    return out

def model_size(d):
    try:
        return json.load(open(os.path.join(d, "config.json")))["model"].get("size", "")
    except Exception:
        return ""

# ---- learning_curves.json : v2 runs with >=1 val epoch.  For 500k/3M only the
#      current (xl) models count — the cancelled medium runs must not show. ----
lc = {}
for tier in TIERS:
    for cfg in CFGS:
        d = latest_run(cfg, tier)
        if not d: continue
        if tier != "64k" and model_size(d) != "xl":   # skip stale medium 500k/3M dirs
            continue
        s = val_series(d)
        if not s: continue
        state = rr.read_status(d).get("state", "")
        training = state not in ("finished", "completed") or len(s) < 5
        lc[f"{tier}_{cfg}"] = {
            "label": f"{tier} · {cfg}" + (" (training)" if training and tier != "64k" else ""),
            "series": s,
        }
json.dump(lc, open("docs/data/learning_curves.json", "w"))
print("learning_curves:", {k: len(v["series"]) for k, v in lc.items()})

# ---- test_eval.json : 64k held-out vs val (the finished tier) ----
te = {}
for cfg in CFGS:
    d = latest_run(cfg, "64k")
    if not d: continue
    h = rr.read_heldout_eval(d)
    if not (h and h.get("metrics")): continue
    a = rr.analyze_run(d); bm = a.get("best_metrics", {}) or {}
    m = h["metrics"]
    keys = ["shift_mae_ppm", "j_mae_hz", "presence_f1", "deg_acc_balanced"]
    te[cfg] = {
        "val":  {k: round(bm.get(k, 0), 4) for k in keys},
        "test": {k: round(m.get(k, 0), 4) for k in keys},
    }
    n_test = h.get("n_test", 0)
te["_meta"] = {
    "split": "leakage-controlled global 10% PubChem held-out",
    "test_n_total": 315629, "test_n_eval": n_test if te else 0,
    "note": "Held-out on a leakage-controlled global 10% PubChem split (union-find on matrix fingerprint + InChIKey).",
}
json.dump(te, open("docs/data/test_eval.json", "w"))
print("test_eval models:", [k for k in te if k != "_meta"])
