"""Generate docs/data/{learning_curves,test_eval}.json from the corrected-data
REBUILD fleet (Audit-2 regeneration). One run per data tier:

    64k  -> rebuild_64k_026       medium  ~10M
    500k -> rebuild_500k_xl_026   xl      ~57M
    3M   -> rebuild_3M_xxl_026    xxl     ~137M

Reads val rows from metrics.jsonl + the standardized heldout_eval.json (written by
model.experiments.eval_heldout on the leakage-controlled global 10% PubChem test
split) via autoai.run_reader. Run on the HPC where the run dirs live, then pull the
two JSON files into docs/data/.

    PYTHONPATH=. python docs/gen_viewer_data.py
"""
import glob
import json

from autoai import run_reader as rr

RUNS = "model/runs"

# tier key -> (run-name substring, display label, params). The substring matches the
# run dir <date>_<time>_<name>_<hash>; exact names avoid the cancelled rebuild_3M_xl.
FLEET = [
    ("64k",  "rebuild_64k_026",     "64k · medium",  "10M"),
    ("500k", "rebuild_500k_xl_026", "500k · xl",     "57M"),
    ("3M",   "rebuild_3M_xxl_026",  "3M · xxl",      "137M"),
]
KEYS = ["shift_mae_ppm", "j_mae_hz", "presence_f1", "deg_acc_balanced"]


def latest_run(name):
    ds = sorted(glob.glob(f"{RUNS}/*_{name}_*"))
    return ds[-1] if ds else None


def val_series(d):
    out = []
    for r in rr.read_metrics(d):
        if r.get("split") != "val":
            continue
        m = r.get("metrics", {})
        if "shift_mae_ppm" not in m:
            continue
        out.append({"epoch": r.get("epoch", len(out)),
                    "shift": round(m["shift_mae_ppm"], 4),
                    "j": round(m.get("j_mae_hz", 0), 3),
                    "f1": round(m.get("presence_f1", 0), 4),
                    "deg": round(m.get("deg_acc_balanced", 0), 4)})
    return out


# ---- learning_curves.json : every fleet tier with >=1 val epoch ------------------
lc = {}
for key, name, lab, params in FLEET:
    d = latest_run(name)
    if not d:
        continue
    s = val_series(d)
    if not s:
        continue
    state = rr.read_status(d).get("state", "")
    training = state not in ("finished", "completed")
    lc[key] = {"label": f"{lab} ({params})" + (" · training" if training else ""),
               "series": s}
json.dump(lc, open("docs/data/learning_curves.json", "w"))
print("learning_curves:", {k: len(v["series"]) for k, v in lc.items()})

# ---- test_eval.json : standardized held-out (val vs test) per tier ---------------
te = {}
n_eval = 0
for key, name, lab, params in FLEET:
    d = latest_run(name)
    if not d:
        continue
    h = rr.read_heldout_eval(d)
    if not (h and h.get("metrics")):
        continue
    bm = (rr.analyze_run(d).get("best_metrics", {}) or {})
    m = h["metrics"]
    te[key] = {"val":  {k: round(bm.get(k, 0), 4) for k in KEYS},
               "test": {k: round(m.get(k, 0), 4) for k in KEYS}}
    n_eval = h.get("n_test", n_eval)
te["_meta"] = {
    "split": "leakage-controlled global 10% PubChem held-out",
    "test_n_total": 312682, "test_n_eval": n_eval,
    "note": "Held-out on a leakage-controlled global 10% PubChem split (union-find on "
            "matrix fingerprint + InChIKey); every tier scored on identical molecules.",
}
json.dump(te, open("docs/data/test_eval.json", "w"))
print("test_eval models:", [k for k in te if k != "_meta"])
