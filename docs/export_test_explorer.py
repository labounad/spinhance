"""Build docs/data/test_explorer.json — the held-out test-molecule explorer.

Samples molecules from the global leakage-controlled held-out split
(records_3M_test.json.gz, pre-sorted by test_rank) using the NESTED-SUBSET scheme
(first N by rank), fetches their 90 MHz spectra from the stacked parts, and runs
each model — the CNN baseline + the 64k spingraph_decoder configs — producing a
prediction matrix + a rendered spectrum per model. Every model is scored on the
SAME held-out PubChem molecules, so there is no per-model test fold.

    PYTHONPATH=. python docs/export_test_explorer.py [OUT.json] [N]

Runs on the HPC where the checkpoints + rebuild parts live (CPU is fine — inference).
"""
import glob
import json
import os
import sys

import numpy as np
import torch

from simulation.pyspin.composite import simulate_spectrum_composite
from model.data.records import load_pubchem_records
from model.data.stacked_spectra import StackedSpectra
from model.data.standardization import DegeneracyVocab, Standardizer
from model.architectures import build_architecture
from model.evaluation.metrics import decode, _np_pred

REB = "/gpfs/home/labounader/rebuild3M"
RUNS = "/gpfs/home/labounader/code/spinhance/model/runs"
TEST = REB + "/records_3M_test.json.gz"
PARTS = REB + "/parts"
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/data/test_explorer.json"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 80
G, P = 8, 16384


def best_ckpt(name):
    # newest FINISHED run for this tier (skip still-training runs so the explorer never
    # shows undertrained predictions, e.g. a 500k at epoch 3).
    for d in sorted(glob.glob(f"{RUNS}/*_{name}_*"), reverse=True):
        p = os.path.join(d, "checkpoints", "best.pt")
        try:
            st = json.load(open(os.path.join(d, "status.json"))).get("state")
        except Exception:
            st = None
        if st in ("finished", "completed") and os.path.exists(p):
            return p
    return None

# The corrected-data REBUILD fleet (one run per tier) + the CNN baseline. Each model
# is included only once its best.pt exists, so this is turnkey: re-run as checkpoints
# land and the explorer gains tiers without edits.
FLEET = [("64k",  "rebuild_64k_026",     "64k · medium (10M)"),
         ("500k", "rebuild_500k_xl_026", "500k · xl (57M)"),
         ("3M",   "rebuild_3M_xxl_026",  "3M · xxl (137M)")]
MODELS, MODEL_LABELS = {}, []
# CNN baseline (trained on ChEMBL, but handles PubChem fine) — reference floor.
BASELINE = "/gpfs/home/labounader/ckpts/baseline_best.pt"
if os.path.exists(BASELINE):
    MODELS["baseline"] = BASELINE
    MODEL_LABELS.append(("baseline", "CNN baseline · ResNet-1D"))
for key, name, label in FLEET:
    p = best_ckpt(name)
    if p:
        MODELS[key] = p
        MODEL_LABELS.append((key, label))

vocab = DegeneracyVocab()
print("loading models...", flush=True)
loaded = {}
for k, p in MODELS.items():
    ck = torch.load(p, map_location="cpu", weights_only=False)
    std = Standardizer().load_state_dict(ck["standardizer"])
    mc = dict(ck["cfg"]["model"]); nm = mc.pop("name")
    m = build_architecture(nm, n_deg_classes=len(vocab), **mc).eval()
    m.load_state_dict(ck["model"], strict=False)
    loaded[k] = (m, std); print(" loaded", k, flush=True)


PPM_MAX = 12.0
MARGIN = 0.4        # ppm padding on each side of the active region
MIN_WIN = 1.5       # don't over-zoom narrow spectra
N_OUT = 1024        # output points across the (smaller) window -> higher effective resolution


def active_window(shifts):
    """[lo, hi] ppm window covering the spin groups (+margin), like the hero zoom."""
    lo = max(0.0, float(np.min(shifts)) - MARGIN)
    hi = min(PPM_MAX, float(np.max(shifts)) + MARGIN)
    if hi - lo < MIN_WIN:                       # widen tight windows, keep centred + clamped
        c = (lo + hi) / 2.0
        lo = max(0.0, c - MIN_WIN / 2.0); hi = min(PPM_MAX, lo + MIN_WIN)
        lo = max(0.0, hi - MIN_WIN)
    return lo, hi


def window_pool(y, lo, hi):
    """Crop a full 0-PPM_MAX spectrum to [lo, hi] and max-pool to ~N_OUT points.
    Sampling stays bounded (<=~2*N_OUT) but the window is smaller -> finer ppm resolution."""
    i0 = max(0, int(round(lo / PPM_MAX * P)))
    i1 = min(P, int(round(hi / PPM_MAX * P)))
    seg = y[i0:i1]
    if seg.size == 0:
        return np.zeros(1, np.float32)
    k = max(1, seg.size // N_OUT)
    m = (seg.size // k) * k
    return seg[:m].reshape(-1, k).max(1)


def xyz_of(smi):
    if not smi:
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = Chem.AddHs(m)
        if AllChem.EmbedMolecule(m, AllChem.ETKDGv3()) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=400)
        except Exception:
            pass
        c = m.GetConformer(); L = [str(m.GetNumAtoms()), smi]
        for a in m.GetAtoms():
            pt = c.GetAtomPosition(a.GetIdx())
            L.append(f"{a.GetSymbol()} {pt.x:.4f} {pt.y:.4f} {pt.z:.4f}")
        return "\n".join(L)
    except Exception:
        return None


iu = np.triu_indices(G, 1)


def emit(rid, smi, sh, cp, dg, spec):
    sc = float(spec.max()) or 1.0
    lo, hi = active_window(sh)                  # zoom to the molecule's actual peaks (+margin)
    to = np.argsort(-sh); tsh, tdg = sh[to], dg[to]; tJ = cp[np.ix_(to, to)]
    preds = {}
    for k, (m, std) in loaded.items():
        with torch.no_grad():
            o = m(torch.tensor(spec, dtype=torch.float32)[None])
        pr = _np_pred(o); pr.pop("soft_equiv", None); dec = decode(pr, std, vocab)
        psh, pcp, pdg = dec["shifts"][0], dec["couplings"][0], dec["degeneracy"][0]
        po = np.argsort(-psh); psh2, pdg2 = psh[po], pdg[po]; pJ = pcp[np.ix_(po, po)]
        _, rspec = simulate_spectrum_composite(psh, pcp, pdg, 90.0, points=P)
        mm = np.abs(tJ[iu]) > 0.5
        preds[k] = {"pred_shift": [round(float(x), 3) for x in psh2], "pred_deg": [int(x) for x in pdg2],
                    "pred_J": [[round(float(pJ[i, j]), 2) for j in range(G)] for i in range(G)],
                    "rendered": [round(float(v / sc), 4) for v in window_pool(rspec, lo, hi)],
                    "shift_mae": round(float(np.mean(np.abs(tsh - psh2))), 4),
                    "j_mae": round(float(np.mean(np.abs(tJ[iu][mm] - pJ[iu][mm]))) if mm.any() else 0.0, 3)}
    return {"id": rid, "smiles": smi or "", "n_spins": int(np.sum(dg)), "src": "pubchem",
            "x0": round(lo, 3), "x1": round(hi, 3),
            "input": [round(float(v / sc), 4) for v in window_pool(spec, lo, hi)],
            "true_shift": [round(float(x), 3) for x in tsh], "true_deg": [int(x) for x in tdg],
            "true_J": [[round(float(tJ[i, j]), 2) for j in range(G)] for i in range(G)],
            "xyz": xyz_of(smi), "preds": preds}


print(f"held-out nested subset: first {N} by test_rank", flush=True)
recs = load_pubchem_records(TEST)[:N]           # pre-sorted by test_rank -> nested subset
src = StackedSpectra(PARTS)
mols = []
for r in recs:
    sh = np.array(r["shifts"], float); cp = np.array(r["couplings"], float); dg = np.array(r["degeneracy"], int)
    spec = np.asarray(src[r["row"]], np.float32)
    mols.append(emit(r["mol_id"], r.get("smiles"), sh, cp, dg, spec))
    if len(mols) % 20 == 0:
        print(" ", len(mols), "/", len(recs), flush=True)

# Each molecule carries its own [x0, x1] ppm window; the viewer derives a per-molecule
# ppm axis from it (no shared global axis), so spectra are stored at the window's resolution.
out = {"models": [{"key": k, "label": l} for k, l in MODEL_LABELS], "molecules": mols}
json.dump(out, open(OUT, "w"), separators=(",", ":"))
print("WROTE", OUT, round(os.path.getsize(OUT) / 1024, 1), "KB |", len(mols), "mols", flush=True)
