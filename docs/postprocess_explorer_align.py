"""One-shot: relabel the stored predictions in test_explorer.json by the exact-orbit
alignment permutation, so the DISPLAYED J matrix matches the reported (already-aligned)
J-MAE. Equivalent to regenerating with the fixed export, but instant (no model reload).

Applies align_pred_permutation to (pred_shift, pred_deg, pred_J) together per molecule/model.
Verifies the reported j_mae is unchanged (this only relabels; it never alters the metric).

    PYTHONPATH=. python3 docs/postprocess_explorer_align.py
"""
import json, sys
import numpy as np
from model.evaluation.symmetry import align_pred_permutation

PATH = "docs/data/test_explorer.json"
G = 8
iu = np.triu_indices(G, 1)
D = json.load(open(PATH))
changed = 0; total = 0; max_jmae_drift = 0.0
for mol in D["molecules"]:
    tsh = np.array(mol["true_shift"], float); tdg = np.array(mol["true_deg"], int)
    tJ = np.array(mol["true_J"], float); tmask = (np.abs(tJ) > 0.5).astype(float)
    mm = np.abs(tJ[iu]) > 0.5
    for k, p in mol["preds"].items():
        total += 1
        psh = np.array(p["pred_shift"], float); pdg = np.array(p["pred_deg"], int)
        pJ = np.array(p["pred_J"], float)
        pal = align_pred_permutation(pJ, tJ, tmask, tsh, tdg)
        if not np.array_equal(pal, np.arange(G)):
            changed += 1
        psh, pdg, pJ = psh[pal], pdg[pal], pJ[np.ix_(pal, pal)]
        new_jmae = round(float(np.mean(np.abs(tJ[iu][mm] - pJ[iu][mm]))) if mm.any() else 0.0, 3)
        if p.get("j_mae") is not None:
            max_jmae_drift = max(max_jmae_drift, abs(new_jmae - p["j_mae"]))
        p["pred_shift"] = [round(float(x), 3) for x in psh]
        p["pred_deg"] = [int(x) for x in pdg]
        p["pred_J"] = [[round(float(pJ[i, j]), 2) for j in range(G)] for i in range(G)]
        p["j_mae"] = new_jmae

json.dump(D, open(PATH, "w"), separators=(",", ":"))
print("relabeled %d/%d (molecule,model) predictions; displays changed by alignment: %d" % (total, total, changed))
print("max |j_mae drift| vs stored: %.4f Hz (should be ~0 — this only relabels)" % max_jmae_drift)
