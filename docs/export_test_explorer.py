"""Build docs/data/test_explorer.json — the held-out test-molecule explorer.

Per-model held-out test folds (N each) with each model's prediction + a rendered
spectrum, for: CNN baseline, 022, 025, 026 (64k ChEMBL test fold) and light-025
(PRODUCTION) + light-026 (500k-PubChem test fold). Runs on the HPC where the
checkpoints + bundle live:

    PYTHONPATH=. python docs/export_test_explorer.py [OUT.json]

Spectra are simulated with the project-default pseudo-Voigt lineshape; at the 1024
downsampled display resolution the lineshape is indistinguishable from Lorentzian
(cos > 0.999), so the Lorentzian-trained models see in-distribution inputs.
"""
import glob
import json
import sys

import numpy as np
import torch

from simulation.pyspin.composite import simulate_spectrum_composite
from model.data.records import load_pubchem_records, load_records
from model.data.stacked_spectra import StackedSpectra
from model.data.splits import make_splits
from model.data.standardization import DegeneracyVocab, Standardizer
from model.architectures import build_architecture
from model.evaluation.metrics import decode, _np_pred

B = "/gpfs/home/labounader/bundle_500k_s0"
RUNS = "/gpfs/home/labounader/code/spinhance/model/runs"
CHEMBL = "/gpfs/home/labounader/ckpts/spin_systems_chembl.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/data/test_explorer.json"
MODELS = {
    "baseline": "/gpfs/home/labounader/ckpts/baseline_best.pt",
    "022": "/gpfs/home/labounader/ckpts/session022_best.pt",
    "025": "/gpfs/home/labounader/ckpts/session025_best.pt",
    "026": "/gpfs/home/labounader/ckpts/session026_best.pt",
    # light-025 is the PRODUCTION model (500k PubChem, e80fv run, best ep43)
    "light025": sorted(glob.glob(RUNS + "/*light_025_e80fv*/checkpoints/best.pt"))[-1],
    "light026": sorted(glob.glob(RUNS + "/*light_026_e80fv*/checkpoints/best.pt"))[-1],
}
MODEL_LABELS = [("baseline", "CNN baseline · ResNet-1D"), ("022", "022 · spingraph + spectral"),
                ("025", "025 · 64k ChEMBL"), ("026", "026 · 64k ChEMBL"),
                ("light025", "light·025 · 500k PubChem (production)"),
                ("light026", "light·026 · 500k PubChem")]
CHEMBL_MODELS = ["baseline", "022", "025", "026"]
PUBCHEM_MODELS = ["light025", "light026"]
N_EACH = 50
DS = 16
G = 8
P = 16384
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


def dsamp(y):
    n = (len(y) // DS) * DS
    return y[:n].reshape(-1, DS).max(1)


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


def emit(rid, smi, sh, cp, dg, spec, test_of, src):
    sc = float(spec.max()) or 1.0
    to = np.argsort(-sh); tsh, tdg = sh[to], dg[to]; tJ = cp[np.ix_(to, to)]
    preds = {}
    for k, (m, std) in loaded.items():
        with torch.no_grad():
            o = m(torch.tensor(spec, dtype=torch.float32)[None])
        pr = _np_pred(o); pr.pop("soft_equiv", None); dec = decode(pr, std, vocab)
        psh, pcp, pdg = dec["shifts"][0], dec["couplings"][0], dec["degeneracy"][0]
        po = np.argsort(-psh); psh2, pdg2 = psh[po], pdg[po]; pJ = pcp[np.ix_(po, po)]
        _, rspec = simulate_spectrum_composite(psh, pcp, pdg, 90.0, points=P); mm = np.abs(tJ[iu]) > 0.5
        preds[k] = {"pred_shift": [round(float(x), 3) for x in psh2], "pred_deg": [int(x) for x in pdg2],
                    "pred_J": [[round(float(pJ[i, j]), 2) for j in range(G)] for i in range(G)],
                    "rendered": [round(float(v / sc), 4) for v in dsamp(rspec)],
                    "shift_mae": round(float(np.mean(np.abs(tsh - psh2))), 4),
                    "j_mae": round(float(np.mean(np.abs(tJ[iu][mm] - pJ[iu][mm]))) if mm.any() else 0.0, 3)}
    return {"id": rid, "smiles": smi or "", "n_spins": int(np.sum(dg)), "src": src, "test_of": test_of,
            "input": [round(float(v / sc), 4) for v in dsamp(spec)],
            "true_shift": [round(float(x), 3) for x in tsh], "true_deg": [int(x) for x in tdg],
            "true_J": [[round(float(tJ[i, j]), 2) for j in range(G)] for i in range(G)],
            "xyz": xyz_of(smi), "preds": preds}


mols = []
print("chembl test fold...", flush=True)
crecs = load_records(CHEMBL, "simulation/data/spectra", fields=(90,), require_spectra=False)
ca, _ = make_splits(crecs, seed=0, compute_scaffold=False)
ctest = [r for r in crecs if ca.get(r["mol_id"]) == "test"]; ctest.sort(key=lambda r: int(np.sum(r["degeneracy"])))
for r in [ctest[i] for i in np.linspace(0, len(ctest) - 1, N_EACH).astype(int)]:
    sh, cp, dg = np.array(r["shifts"], float), np.array(r["couplings"], float), np.array(r["degeneracy"], int)
    _, spec = simulate_spectrum_composite(sh, cp, dg, 90.0, points=P)
    mols.append(emit(r["mol_id"], r.get("smiles"), sh, cp, dg, np.asarray(spec, np.float32), CHEMBL_MODELS, "chembl"))
print(" chembl done", len(mols), flush=True)
precs = load_pubchem_records(B + "/records.json.gz", max_mol=0); src = StackedSpectra(B)
pa, _ = make_splits(precs, seed=0, compute_scaffold=False)
ptest = [r for r in precs if pa.get(r["mol_id"]) == "test"]; ptest.sort(key=lambda r: int(np.sum(r["degeneracy"])))
for r in [ptest[i] for i in np.linspace(0, len(ptest) - 1, N_EACH).astype(int)]:
    sh, cp, dg = np.array(r["shifts"], float), np.array(r["couplings"], float), np.array(r["degeneracy"], int)
    spec = np.asarray(src[r["row"]], np.float32)
    mols.append(emit(r["mol_id"], r.get("smiles"), sh, cp, dg, spec, PUBCHEM_MODELS, "pubchem"))
print(" pubchem done", len(mols), flush=True)
out = {"ppm": [round(float(x), 3) for x in np.linspace(0, 12, P // DS)],
       "models": [{"key": k, "label": l} for k, l in MODEL_LABELS], "molecules": mols}
json.dump(out, open(OUT, "w"), separators=(",", ":"))
import os
print("WROTE", OUT, round(os.path.getsize(OUT) / 1024, 1), "KB |", len(mols), "mols", flush=True)
