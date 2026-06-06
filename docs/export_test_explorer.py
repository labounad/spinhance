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
from model.evaluation.symmetry import align_pred_couplings
from model.inference.refine import refine_system

REB = "/gpfs/group/shenvi/Users/labounader/spinhance/rebuild3M"
RUNS = "/gpfs/group/shenvi/Users/labounader/spinhance/runs"
TEST = "/gpfs/group/shenvi/Users/labounader/spinhance/consolidated_test/records_test_consol.json.gz"   # consolidated (contiguous) held-out
PARTS = "/gpfs/group/shenvi/Users/labounader/spinhance/consolidated_test/parts"
# CLI:  OUT.json [N]               -> render all N molecules, write OUT (serial)
#       --only IDX FRAGDIR [N]     -> render ONE molecule -> FRAGDIR/mol_IDX.json (SLURM array task)
#       --combine FRAGDIR OUT [N]  -> merge per-molecule fragments -> OUT (no model load)
_a = sys.argv[1:]
MODE, ONLY_IDX, FRAG_DIR = "all", None, None
OUT = "docs/data/test_explorer.json"; N = 80
if _a and _a[0] == "--only":
    MODE = "only"; ONLY_IDX = int(_a[1]); FRAG_DIR = _a[2]
    if len(_a) > 3: N = int(_a[3])
elif _a and _a[0] == "--combine":
    MODE = "combine"; FRAG_DIR = _a[1]; OUT = _a[2]
    if len(_a) > 3: N = int(_a[3])
else:
    if _a: OUT = _a[0]
    if len(_a) > 1: N = int(_a[1])
G, P = 8, 16384


def best_ckpt(name):
    # newest FINISHED run for this name. If the NEWEST run is still training, the model
    # isn't ready -> return None rather than falling back to a SUPERSEDED older finished
    # run (e.g. a prematurely early-stopped run that is being re-trained).
    for d in sorted(glob.glob(f"{RUNS}/*_{name}_*"), reverse=True):
        p = os.path.join(d, "checkpoints", "best.pt")
        try:
            st = json.load(open(os.path.join(d, "status.json"))).get("state")
        except Exception:
            st = None
        if st == "running":
            return None
        if st in ("finished", "completed") and os.path.exists(p):
            return p
    return None

# The corrected-data REBUILD fleet (one run per tier) + the CNN baseline. Each model
# is included only once its best.pt exists, so this is turnkey: re-run as checkpoints
# land and the explorer gains tiers without edits.
FLEET = [("64k_025", "rebuild_64k_025_sym",     "64k · 025 (matrix)"),
         ("64k_026", "rebuild_64k_026_sym",     "64k · 026 (peak+soft-eq)"),
         ("64k_027", "rebuild_64k_027_sym",     "64k · 027 (focal)"),
         ("64k_028", "rebuild_64k_028_sym",     "64k · 028 (cum-integral)"),
         ("64k_029", "rebuild_64k_029_sym",     "64k · 029 (026+focal)"),
         ("64k_030", "rebuild_64k_030_sym",     "64k · 030 (super)"),
         ("500k_025", "rebuild_500k_025_sym", "500k · 025 (matrix, 57M)"),
         ("500k_026", "rebuild_500k_026_sym", "500k · 026 (peak+soft-eq, 57M)"),
         ("500k_027", "rebuild_500k_027_sym", "500k · 027 (focal, 57M)"),
         ("500k_028", "rebuild_500k_028_sym", "500k · 028 (cum-integral, 57M)"),
         ("500k_029", "rebuild_500k_029_sym", "500k · 029 (026+focal, 57M, best)"),
         ("500k_030", "rebuild_500k_030_sym", "500k · 030 (super, 57M)"),
         ("3M",      "rebuild_3M_026_sym",  "3M · 026 (137M)")]
MODELS, MODEL_LABELS = {}, []
# CNN baseline (trained on ChEMBL, but handles PubChem fine) — reference floor.
BASELINE = "/gpfs/group/shenvi/Users/labounader/spinhance/ckpts/baseline_best.pt"
if os.path.exists(BASELINE):
    MODELS["baseline"] = BASELINE
    MODEL_LABELS.append(("baseline", "CNN baseline · ResNet-1D"))
for key, name, label in FLEET:
    p = best_ckpt(name)
    if p:
        MODELS[key] = p
        MODEL_LABELS.append((key, label))

# optional tier filter (env EXPLORER_TIERS="64k" -> baseline + 64k models only): focuses
# the explorer + cuts per-task work. baseline is always kept as the reference floor.
_tiers = os.environ.get("EXPLORER_TIERS")
if _tiers:
    keep = set(_tiers.split(","))
    MODEL_LABELS = [(k, l) for k, l in MODEL_LABELS if k == "baseline" or k.split("_")[0] in keep]
    MODELS = {k: v for k, v in MODELS.items() if k == "baseline" or k.split("_")[0] in keep}

vocab = DegeneracyVocab()
loaded = {}
if MODE != "combine":                       # --combine just merges fragments; no model load
    print("loading models...", flush=True)
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

# --- adaptive peak mesh (Ramer-Douglas-Peucker) -------------------------------
# A ¹H spectrum is ~95% flat baseline + a handful of sharp peaks; a uniform grid wastes
# nearly all its points on the baseline. Every peak has the same shape (pseudo-Voigt,
# eta=0.8), so we render each curve at full resolution, then keep only the points needed
# to reproduce it by LINEAR INTERPOLATION to within a vertical tolerance EPS (RDP with a
# vertical-distance metric == bounding the interpolation error). Flat baseline collapses
# to its endpoints; points cluster around peaks where the curve actually bends. Each
# spectrum carries its own mesh (x, y), so the plotter just interpolates between them.
PPM_FULL = np.linspace(0.0, PPM_MAX, P)          # full-res ppm grid (0..12), the sample source
EPS = 0.002                                      # max linear-interp error, fraction of input peak
SNAP = 7.5e-4                                    # values below this (normalized) snap to 0 (cheap baseline)


def rdp_curve(y_full, lo, hi, sc):
    """Sparse polyline (xs ppm, ys) reproducing the full-res curve y_full on [lo, hi] to
    within EPS vertical error after /sc normalization. Returns two equal-length lists."""
    i0 = max(0, int(round(lo / PPM_MAX * P)))
    i1 = min(P - 1, int(round(hi / PPM_MAX * P)))
    y = (y_full[i0:i1 + 1].astype(float)) / sc
    n = y.size
    if n < 2:
        return [round(lo, 4)], [0.0]
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:                                  # iterative RDP, vertical (interp-error) metric
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = np.arange(a + 1, b)
        line = y[a] + (y[b] - y[a]) * (seg - a) / (b - a)
        d = np.abs(y[a + 1:b] - line)
        k = int(d.argmax())
        if d[k] > EPS:
            mid = a + 1 + k; keep[mid] = True
            stack.append((a, mid)); stack.append((mid, b))
    idx = np.nonzero(keep)[0]
    xs = [round(float(v), 4) for v in PPM_FULL[i0 + idx]]
    ys = [(0.0 if v < SNAP else round(float(v), 4)) for v in y[idx]]
    return xs, ys


def active_window(shifts):
    """[lo, hi] ppm window covering the spin groups (+margin), like the hero zoom."""
    lo = max(0.0, float(np.min(shifts)) - MARGIN)
    hi = min(PPM_MAX, float(np.max(shifts)) + MARGIN)
    if hi - lo < MIN_WIN:                       # widen tight windows, keep centred + clamped
        c = (lo + hi) / 2.0
        lo = max(0.0, c - MIN_WIN / 2.0); hi = min(PPM_MAX, lo + MIN_WIN)
        lo = max(0.0, hi - MIN_WIN)
    return lo, hi


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
        rx, ry = rdp_curve(rspec, lo, hi, sc)              # this model's adaptive mesh
        # symmetry-aware J: align the predicted couplings to the target under the equal-shift
        # tie-break group so swapping chemically-equivalent groups isn't penalized (model.evaluation.symmetry).
        tmask = (np.abs(tJ) > 0.5).astype(float)
        pJ_al = align_pred_couplings(pJ, tJ, tmask, tsh, tdg)
        mm = np.abs(tJ[iu]) > 0.5
        # test-time refinement (analysis-by-synthesis): polish the predicted shifts to
        # match the INPUT spectrum (couplings/degeneracy fixed) — a "legal" correction
        # that uses only the 90 MHz input. Render + mesh the post-corrected spectrum.
        # test-time refinement (joint shift+J: graduated non-convexity + centroid coarse-fix).
        # The refined overlay reflects BOTH corrected shifts AND corrected couplings.
        refined, ref_cp, rinfo = refine_system(psh, pcp, pdg, spec, field_mhz=90.0, max_cost=5e10)  # raise guard above the explorer set's max eigh_cost (1.37e10; renders <=2.2s) so all refine
        rsh = np.sort(refined)[::-1]
        _, fspec = simulate_spectrum_composite(refined, ref_cp, pdg, 90.0, points=P)
        fx, fy = rdp_curve(fspec, lo, hi, sc)
        rpo = np.argsort(-refined)
        rpJ_al = align_pred_couplings(ref_cp[np.ix_(rpo, rpo)], tJ, tmask, tsh, tdg)
        preds[k] = {"pred_shift": [round(float(x), 3) for x in psh2], "pred_deg": [int(x) for x in pdg2],
                    "pred_J": [[round(float(pJ[i, j]), 2) for j in range(G)] for i in range(G)],
                    "rx": rx, "rendered": ry,
                    "ref_shift": [round(float(x), 3) for x in rsh], "fx": fx, "refined": fy,
                    "ref_skipped": bool(rinfo.get("skipped", False)),
                    "shift_mae": round(float(np.mean(np.abs(tsh - psh2))), 4),
                    "ref_shift_mae": round(float(np.mean(np.abs(tsh - rsh))), 4),
                    "j_mae": round(float(np.mean(np.abs(tJ[iu][mm] - pJ_al[iu][mm]))) if mm.any() else 0.0, 3),
                    "ref_j_mae": round(float(np.mean(np.abs(tJ[iu][mm] - rpJ_al[iu][mm]))) if mm.any() else 0.0, 3)}
    ix, iy = rdp_curve(spec, lo, hi, sc)                   # the input (target) adaptive mesh
    return {"id": rid, "smiles": smi or "", "n_spins": int(np.sum(dg)), "src": "pubchem",
            "x0": round(lo, 3), "x1": round(hi, 3),
            "ix": ix, "input": iy,
            "true_shift": [round(float(x), 3) for x in tsh], "true_deg": [int(x) for x in tdg],
            "true_J": [[round(float(tJ[i, j]), 2) for j in range(G)] for i in range(G)],
            "xyz": xyz_of(smi), "preds": preds}


def _models_block():
    return [{"key": k, "label": l} for k, l in MODEL_LABELS]

# --combine: merge per-molecule fragments (from the array tasks) into the final JSON.
if MODE == "combine":
    frags = sorted(glob.glob(os.path.join(FRAG_DIR, "mol_*.json")))
    mols = [json.load(open(f)) for f in frags]
    out = {"models": _models_block(), "molecules": mols}
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print("COMBINED", OUT, round(os.path.getsize(OUT) / 1024, 1), "KB |", len(mols), "mols", flush=True)
    sys.exit(0)

print(f"held-out nested subset: first {N} by test_rank", flush=True)
recs = load_pubchem_records(TEST, max_mol=N)[:N]   # pre-sorted by test_rank -> nested subset
src = StackedSpectra(PARTS)

def _emit_one(r):
    sh = np.array(r["shifts"], float); cp = np.array(r["couplings"], float); dg = np.array(r["degeneracy"], int)
    spec = np.asarray(src[r["row"]], np.float32)
    return emit(r["mol_id"], r.get("smiles"), sh, cp, dg, spec)

# --only IDX: render ONE molecule to a fragment file (a SLURM array task; CPU-only, so it
# backfills across the whole cluster and the 80 molecules finish in parallel in minutes).
if MODE == "only":
    os.makedirs(FRAG_DIR, exist_ok=True)
    if ONLY_IDX >= len(recs):
        print(f"idx {ONLY_IDX} >= {len(recs)} valid recs — nothing to do"); sys.exit(0)
    mol = _emit_one(recs[ONLY_IDX])
    fp = os.path.join(FRAG_DIR, f"mol_{ONLY_IDX:05d}.json")
    json.dump(mol, open(fp, "w"), separators=(",", ":"))
    print("WROTE FRAG", fp, "| mol", ONLY_IDX, mol.get("id"), flush=True)
    sys.exit(0)

# MODE == "all": serial render of every molecule (the original turnkey path).
mols = []
for r in recs:
    mols.append(_emit_one(r))
    if len(mols) % 20 == 0:
        print(" ", len(mols), "/", len(recs), flush=True)
out = {"models": _models_block(), "molecules": mols}
json.dump(out, open(OUT, "w"), separators=(",", ":"))
print("WROTE", OUT, round(os.path.getsize(OUT) / 1024, 1), "KB |", len(mols), "mols", flush=True)
