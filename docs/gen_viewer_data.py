"""Generate docs/data/{learning_curves,test_eval}.json from the corrected-data
REBUILD fleet (Audit-2). Two comparison axes the website toggles between:

    architecture sweep @ 64k :  025 / 026 / 027 / 028 / 029   (fixed size, vary recipe)
    size sweep @ 026         :  64k(10M) / 500k(57M) / 3M(137M) (fixed recipe, vary size)

Each model carries `tier` + `recipe` + `params` so the viewers can build both views.
Only FINISHED runs get held-out numbers; a still-training run is emitted as a 'training'
stub (so its column shows but reads "—/training"). Reads metrics.jsonl + the standardized
heldout_eval.json via model.diagnostics.run_reader. Run on the HPC, then pull the JSON.

    PYTHONPATH=. python docs/gen_viewer_data.py
"""
import glob
import json

from model.diagnostics import run_reader as rr

RUNS = "model/runs"

# key -> (run-name substring, label, tier, recipe, params). The substring matches the run
# dir <date>_<time>_<name>_<hash>; exact names avoid the cancelled rebuild_3M_xl / 500k_030.
FLEET = [
    ("64k_025",  "rebuild_64k_025",     "64k · 025", "64k",  "025", "10M"),
    ("64k_026",  "rebuild_64k_026",     "64k · 026", "64k",  "026", "10M"),
    ("64k_027",  "rebuild_64k_027",     "64k · 027", "64k",  "027", "10M"),
    ("64k_028",  "rebuild_64k_028",     "64k · 028", "64k",  "028", "10M"),
    ("64k_029",  "rebuild_64k_029",     "64k · 029", "64k",  "029", "10M"),
    ("500k_026", "rebuild_500k_xl_026", "500k · 026", "500k", "026", "57M"),
    ("3M_026",   "rebuild_3M_xxl_026",  "3M · 026",   "3M",   "026", "137M"),
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


lc, te, n_eval = {}, {}, 0
for key, name, label, tier, recipe, params in FLEET:
    d = latest_run(name)
    if not d:
        continue
    finished = rr.read_status(d).get("state", "") in ("finished", "completed")
    meta = {"tier": tier, "recipe": recipe, "params": params}

    s = val_series(d)
    if s:
        lc[key] = {"label": label + ("" if finished else " · training"),
                   "series": s, **meta}

    h = rr.read_heldout_eval(d)
    if finished and h and h.get("metrics"):
        bm = (rr.analyze_run(d).get("best_metrics", {}) or {})
        m = h["metrics"]
        te[key] = {**meta, "state": "finished",
                   "val":  {k: round(bm.get(k, 0), 4) for k in KEYS},
                   "test": {k: round(m.get(k, 0), 4) for k in KEYS}}
        n_eval = h.get("n_test", n_eval)
    else:                                          # known fleet member, not yet evaluable
        te[key] = {**meta, "state": "training"}

te["_meta"] = {
    "split": "leakage-controlled global 10% PubChem held-out",
    "test_n_total": 312682, "test_n_eval": n_eval,
    "note": "Held-out on a leakage-controlled global 10% PubChem split (union-find on "
            "matrix fingerprint + InChIKey); every model scored on identical molecules.",
}
json.dump(lc, open("docs/data/learning_curves.json", "w"))
json.dump(te, open("docs/data/test_eval.json", "w"))
print("learning_curves:", {k: len(v["series"]) for k, v in lc.items()})
print("test_eval:", {k: te[k].get("state") for k in te if k != "_meta"})
