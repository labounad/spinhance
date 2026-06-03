"""Add ONE model's predictions to docs/data/test_explorer.json in place — without
re-running the other models or re-embedding 3D coords. Each molecule's spin system
is reconstructed from the stored true_shift/true_deg/true_J, its 90 MHz input is
re-simulated, the checkpoint is run, and preds[KEY] is appended (+ KEY added to
test_of for molecules of --test-src, and to the models list). Run on the HPC:

    PYTHONPATH=. python docs/augment_explorer_model.py \
        --ckpt <best.pt> --key light027 --label "500k·027 · focal (running)" \
        --test-src pubchem --json docs/data/test_explorer.json
"""
import argparse
import json

import numpy as np
import torch

from simulation.pyspin.composite import simulate_spectrum_composite
from model.data.standardization import DegeneracyVocab, Standardizer
from model.architectures import build_architecture
from model.evaluation.metrics import decode, _np_pred

G, P, DS = 8, 16384, 16
iu = np.triu_indices(G, 1)


def dsamp(y):
    n = (len(y) // DS) * DS
    return y[:n].reshape(-1, DS).max(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--test-src", default=None, help="add KEY to test_of for molecules with this src")
    ap.add_argument("--json", default="docs/data/test_explorer.json")
    ap.add_argument("--insert-after", default="light026", help="model key to insert the new model after")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    vocab = DegeneracyVocab(); std = Standardizer().load_state_dict(ck["standardizer"])
    mc = dict(ck["cfg"]["model"]); nm = mc.pop("name")
    m = build_architecture(nm, n_deg_classes=len(vocab), **mc).eval()
    m.load_state_dict(ck["model"], strict=False)
    print(f"loaded {nm} as '{a.key}'", flush=True)

    d = json.load(open(a.json))
    for n, mol in enumerate(d["molecules"]):
        tsh = np.asarray(mol["true_shift"], float)           # already shift-desc sorted
        tdg = np.asarray(mol["true_deg"], int)
        tJ = np.asarray(mol["true_J"], float)
        _, spec = simulate_spectrum_composite(tsh, tJ, tdg, 90.0, points=P)
        spec = np.asarray(spec, np.float32); sc = float(spec.max()) or 1.0
        with torch.no_grad():
            o = m(torch.tensor(spec, dtype=torch.float32)[None])
        pr = _np_pred(o); pr.pop("soft_equiv", None); dec = decode(pr, std, vocab)
        psh, pcp, pdg = dec["shifts"][0], dec["couplings"][0], dec["degeneracy"][0]
        po = np.argsort(-psh); psh2, pdg2 = psh[po], pdg[po]; pJ = pcp[np.ix_(po, po)]
        _, rspec = simulate_spectrum_composite(psh, pcp, pdg, 90.0, points=P)
        mm = np.abs(tJ[iu]) > 0.5
        mol["preds"][a.key] = {
            "pred_shift": [round(float(x), 3) for x in psh2], "pred_deg": [int(x) for x in pdg2],
            "pred_J": [[round(float(pJ[i, j]), 2) for j in range(G)] for i in range(G)],
            "rendered": [round(float(v / sc), 4) for v in dsamp(np.asarray(rspec, np.float32))],
            "shift_mae": round(float(np.mean(np.abs(tsh - psh2))), 4),
            "j_mae": round(float(np.mean(np.abs(tJ[iu][mm] - pJ[iu][mm]))) if mm.any() else 0.0, 3)}
        if a.test_src and mol.get("src") == a.test_src and a.key not in mol["test_of"]:
            mol["test_of"].append(a.key)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(d['molecules'])}", flush=True)

    if not any(mm["key"] == a.key for mm in d["models"]):
        entry = {"key": a.key, "label": a.label}
        idx = next((i for i, mm in enumerate(d["models"]) if mm["key"] == a.insert_after), len(d["models"]) - 1)
        d["models"].insert(idx + 1, entry)
    json.dump(d, open(a.json, "w"), separators=(",", ":"))
    import os
    print(f"wrote {a.json} ({os.path.getsize(a.json)/1024:.1f} KB) — models: {[mm['key'] for mm in d['models']]}", flush=True)


if __name__ == "__main__":
    main()
