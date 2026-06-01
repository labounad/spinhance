"""Export held-out test molecules + session026 predictions for the web explorer."""
import json, numpy as np, torch
from pathlib import Path
from simulation.pyspin.composite import simulate_spectrum_composite
from model.data.records import load_records
from model.data.splits import make_splits, canonical_order
from model.data.standardization import DegeneracyVocab, Standardizer
from model.architectures import build_architecture
from model.evaluation.metrics import decode, _np_pred
from model.schemas.constants import N_POINTS

CKPT="model_artifacts/session026_best.pt"; RECORDS="mol_to_spin_system/data/spin_systems_chembl.json"
N_MOL=100; DS=16; G=8

def smiles_to_xyz(smi):
    """3D coords from SMILES (ETKDG embed + MMFF) as an XYZ block for 3Dmol.js; None on failure."""
    if not smi: return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0: return None
        try: AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
        except Exception: pass
        conf = mol.GetConformer()
        lines = [str(mol.GetNumAtoms()), smi]
        for a in mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            lines.append(f"{a.GetSymbol()} {p.x:.4f} {p.y:.4f} {p.z:.4f}")
        return "\n".join(lines)
    except Exception:
        return None

ckpt=torch.load(CKPT,map_location="cpu",weights_only=False)
vocab=DegeneracyVocab(); std=Standardizer().load_state_dict(ckpt["standardizer"])
mcfg=dict(ckpt["cfg"]["model"]); name=mcfg.pop("name")
model=build_architecture(name,n_deg_classes=len(vocab),**mcfg).eval(); model.load_state_dict(ckpt["model"], strict=False)
print("loaded",name,"params",sum(p.numel() for p in model.parameters())//1000,"k")

recs=load_records(RECORDS,"simulation/data/spectra",fields=(90,),require_spectra=False)
assign,_=make_splits(recs,seed=0,compute_scaffold=False)
test=[r for r in recs if assign.get(r["mol_id"])=="test"]
test.sort(key=lambda r:int(np.sum(r["degeneracy"])))
pick=[test[i] for i in np.linspace(0,len(test)-1,N_MOL).astype(int)]
print(f"{len(test)} test mols; picked {len(pick)}")

def ds(y):
    n=(len(y)//DS)*DS; return y[:n].reshape(-1,DS).max(1)
ppm=ds(np.linspace(0,12,N_POINTS)); iu=np.triu_indices(G,1)
out={"ppm":[round(float(x),3) for x in np.linspace(0,12,N_POINTS//DS)],"molecules":[]}
for r in pick:
    sh,cp,dg=np.array(r["shifts"],float),np.array(r["couplings"],float),np.array(r["degeneracy"],int)
    _,spec=simulate_spectrum_composite(sh,cp,dg,90.0,points=N_POINTS)
    with torch.no_grad(): o=model(torch.tensor(spec,dtype=torch.float32)[None])
    dec=decode(_np_pred(o),std,vocab)
    psh,pcp,pdg=dec["shifts"][0],dec["couplings"][0],dec["degeneracy"][0]
    _,rspec=simulate_spectrum_composite(psh,pcp,pdg,90.0,points=N_POINTS)
    # align: sort true by canonical, pred by predicted shift desc
    to=list(canonical_order(sh,cp,dg)); po=list(np.argsort(-psh))
    tsh,tdg=sh[to],dg[to]; tJ=cp[np.ix_(to,to)]
    psh2,pdg2=psh[po],pdg[po]; pJ=pcp[np.ix_(po,po)]
    sc=float(np.max(spec)) or 1.0
    shift_mae=float(np.mean(np.abs(tsh-psh2)))
    m=np.abs(tJ[iu])>0.5; jmae=float(np.mean(np.abs(tJ[iu][m]-pJ[iu][m]))) if m.any() else 0.0
    out["molecules"].append({
        "id":r["mol_id"],"smiles":r.get("smiles") or "","n_spins":int(np.sum(dg)),
        "input":[round(float(v/sc),4) for v in ds(spec)],
        "rendered":[round(float(v/sc),4) for v in ds(rspec)],
        "true_shift":[round(float(x),3) for x in tsh],"pred_shift":[round(float(x),3) for x in psh2],
        "true_deg":[int(x) for x in tdg],"pred_deg":[int(x) for x in pdg2],
        "true_J":[[round(float(tJ[i,j]),2) for j in range(G)] for i in range(G)],
        "pred_J":[[round(float(pJ[i,j]),2) for j in range(G)] for i in range(G)],
        "shift_mae":round(shift_mae,4),"j_mae":round(jmae,3),
        "xyz":smiles_to_xyz(r.get("smiles"))})
json.dump(out,open("docs/data/test_explorer.json","w"))
print("wrote docs/data/test_explorer.json",round(Path("docs/data/test_explorer.json").stat().st_size/1024,1),"KB")
