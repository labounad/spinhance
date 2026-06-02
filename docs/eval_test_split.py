"""Held-out TEST-split metrics for 025 + 026 (vs the val numbers). Simulates a
random sample of the test fold once, runs each checkpoint, aggregates metrics."""
import json, numpy as np, torch
from simulation.pyspin.composite import simulate_spectrum_composite
from model.data.records import load_records
from model.data.splits import make_splits, canonical_order
from model.data.standardization import DegeneracyVocab, Standardizer
from model.data.transforms import encode_target
from model.architectures import build_architecture
from model.evaluation.metrics import compute_metrics, _np_pred
from model.schemas.constants import N_POINTS

RECORDS="mol_to_spin_system/data/spin_systems_chembl.json"; N=800; G=8; B=64
CKPTS={"025":"model_artifacts/session025_best.pt","026":"model_artifacts/session026_best.pt"}
# val numbers (best-epoch, from S3 metrics.jsonl) for the side-by-side
VAL={"025":{"shift_mae_ppm":0.037,"j_mae_hz":0.59,"presence_f1":0.940,"deg_acc_balanced":0.945},
     "026":{"shift_mae_ppm":0.0361,"j_mae_hz":0.644,"presence_f1":0.941,"deg_acc_balanced":0.950}}

recs=load_records(RECORDS,"simulation/data/spectra",fields=(90,),require_spectra=False)
assign,_=make_splits(recs,seed=0,compute_scaffold=False)
test=[r for r in recs if assign.get(r["mol_id"])=="test"]
rng=np.random.default_rng(0); sample=[test[i] for i in rng.choice(len(test),min(N,len(test)),replace=False)]
print(f"test fold {len(test)} mols; evaluating {len(sample)}")
# simulate once
specs=[]
for r in sample:
    _,s=simulate_spectrum_composite(np.array(r["shifts"],float),np.array(r["couplings"],float),np.array(r["degeneracy"],int),90.0,points=N_POINTS)
    specs.append(s.astype(np.float32))
specs=np.stack(specs)
# targets (standardized, canonical order) — std comes from each ckpt
def targets(std,vocab):
    sh=[];jm=[];jp=[];dc=[]
    for r in sample:
        t=std.transform(encode_target(r["shifts"],r["couplings"],r["degeneracy"],vocab,
            order=canonical_order(r["shifts"],r["couplings"],r["degeneracy"])))
        sh.append(t["shifts"]);jm.append(t["j_mag"]);jp.append(t["j_presence"]);dc.append(t["deg_class"])
    return {"shifts":np.stack(sh),"j_mag":np.stack(jm),"j_presence":np.stack(jp),"deg_class":np.stack(dc)}

out={}
for tag,path in CKPTS.items():
    c=torch.load(path,map_location="cpu",weights_only=False)
    vocab=DegeneracyVocab(); std=Standardizer().load_state_dict(c["standardizer"])
    mcfg=dict(c["cfg"]["model"]); name=mcfg.pop("name")
    m=build_architecture(name,n_deg_classes=len(vocab),**mcfg).eval(); m.load_state_dict(c["model"],strict=False)
    P={"shifts":[],"j_mag":[],"j_presence":[],"deg_logits":[]}
    for st in range(0,len(specs),B):
        with torch.no_grad(): o=m(torch.tensor(specs[st:st+B]))
        p=_np_pred(o)
        for k in P: P[k].append(p[k])
    pred={k:np.concatenate(v) for k,v in P.items()}
    met=compute_metrics(pred,targets(std,vocab),std,vocab)
    out[tag]={"test":{k:round(float(met[k]),4) for k in ("shift_mae_ppm","j_mae_hz","presence_f1","deg_acc_balanced")},
              "val":VAL[tag]}
    print(tag,"TEST:",out[tag]["test"])
out["_meta"]={"split":"random 70/20/10 seed 0","test_n_total":len(test),"test_n_eval":len(sample),
              "note":"test fold never used in training or model selection; val drove early-stopping"}
json.dump(out,open("docs/data/test_eval.json","w"),indent=1)
print("wrote docs/data/test_eval.json")
