from __future__ import annotations

"""Validation gate for the Pretsch (2009) ¹H shift engine.

Run:  PYTHONPATH=. micromamba run -n spinhance python -m mol_to_spin_system.shifts_pretsch_validate

Prints:
  1. ANCHOR TABLE — textbook anchors + the book's own worked examples, with
     predicted vs expected and Δ.
  2. COVERAGE — over a sample of real drug-like SMILES (local dataset if present,
     else a built-in list): fraction of protium H assigned by a *real* path
     (not fallback), the per-path breakdown, and the predicted-δ distribution.

This is the human-reviewed gate before the engine may replace the Java HOSE
predictor; the Java predictor is NOT available locally, so we compare to anchors
and report coverage rather than head-to-head against HOSE.
"""

import statistics
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem

from mol_to_spin_system.shifts_pretsch import predict_shifts_pretsch_verbose


def _embed(smiles: str, seed: int = 0xF00D) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # fall back to 2-D (alkene cis/trans then assumes cis — documented)
        AllChem.Compute2DCoords(mol)
    else:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    return mol


# (name, SMILES, {description: (atom-selector, expected δ)})
# Selector: a function name we evaluate below, kept simple via SMARTS heads.
ANCHORS = [
    # name, SMILES, list of (label, SMARTS-for-H-parent, expected_delta)
    ("benzene",       "c1ccccc1",   [("ArH", "[cH]", 7.26)]),
    ("toluene",       "Cc1ccccc1",  [("CH3", "[CH3]", 2.34), ("ArH(ortho)", "c1ccccc1", 7.17)]),
    ("ethanol",       "CCO",        [("CH3", "[CH3]", 1.20), ("CH2", "[CH2]O", 3.70)]),
    ("acetic acid",   "CC(=O)O",    [("CH3", "[CH3]", 2.10), ("COOH", "[OX2H1]", 11.5)]),
    ("acetaldehyde",  "CC=O",       [("CHO", "[CX3H1]=O", 9.7)]),
    ("benzaldehyde",  "O=Cc1ccccc1",[("CHO", "[CX3H1]=O", 9.7), ("ArH", "[cH]", 7.5)]),
    ("nitrobenzene",  "O=[N+]([O-])c1ccccc1", [("ArH(ortho)", "[cH]", 8.0)]),
    ("anisole",       "COc1ccccc1", [("OCH3", "[CH3]O", 3.78), ("ArH(para)", "[cH]", 6.9)]),
    ("phenol",        "Oc1ccccc1",  [("ArH(ortho)", "[cH]", 6.8)]),
    ("aniline",       "Nc1ccccc1",  [("ArH(para)", "[cH]", 6.7)]),
    ("furan",         "c1ccoc1",    [("H2", "[cH]o", 7.42), ("H3", "[cH][cH]", 6.38)]),
    ("pyridine",      "c1ccncc1",   [("H2", "[cH]n", 8.59)]),
    ("thiophene",     "c1ccsc1",    [("H2", "[cH]s", 7.31)]),
    ("ethylene",      "C=C",        [("=CH2", "[CH2]=C", 5.25)]),
    ("styrene",       "C=Cc1ccccc1",[("vinyl", "[CH2]=C", 5.4)]),
    ("chloroethane",  "CCCl",       [("CH2Cl", "[CH2]Cl", 3.47), ("CH3", "[CH3]", 1.48)]),
    ("isopropanol",   "CC(O)C",     [("CH", "[CH1]O", 4.0)]),
    # Book worked examples — aromatic margin examples on p178 (≈6.94 / 6.61).
    # p-cresol-like / methoxy-amino patterns reproduced as the engine sees them.
    ("p178-ex1(p-Br-anisole H near OMe)", "COc1ccc(Br)cc1",
        [("ArH ortho to OMe", "[cH]c-O", 6.78)]),
]


def run_anchors() -> None:
    print("=" * 78)
    print("ANCHOR TABLE  (predicted vs expected, Δ in ppm)")
    print("=" * 78)
    print(f"{'compound':<34}{'site':<22}{'pred':>7}{'exp':>7}{'Δ':>7}")
    print("-" * 78)
    for name, smi, sites in ANCHORS:
        mol = _embed(smi)
        if mol is None:
            print(f"{name:<34}  (parse/embed failed)")
            continue
        verbose = predict_shifts_pretsch_verbose(mol)
        for label, smarts, exp in sites:
            patt = Chem.MolFromSmarts(smarts)
            preds = []
            if patt is not None:
                for match in mol.GetSubstructMatches(patt):
                    head = mol.GetAtomWithIdx(match[0])
                    for nb in head.GetNeighbors():
                        if nb.GetAtomicNum() == 1 and nb.GetIsotope() in (0, 1):
                            if nb.GetIdx() in verbose:
                                preds.append(verbose[nb.GetIdx()][0])
                    # head itself may be the H-bearing carbon already covered;
                    # also allow the head to be an H (e.g. OH/COOH selectors)
                    if head.GetAtomicNum() == 1 and head.GetIdx() in verbose:
                        preds.append(verbose[head.GetIdx()][0])
                    if head.GetAtomicNum() in (7, 8) and head.GetTotalNumHs() > 0:
                        # heteroatom H (OH/NH): find its H index
                        for nb in head.GetNeighbors():
                            if nb.GetAtomicNum() == 1 and nb.GetIdx() in verbose:
                                preds.append(verbose[nb.GetIdx()][0])
            if not preds:
                print(f"{name:<34}{label:<22}{'n/a':>7}{exp:>7.2f}{'--':>7}")
                continue
            pred = statistics.mean(preds)
            d = pred - exp
            flag = "" if abs(d) <= 0.5 else "  <-- off"
            print(f"{name:<34}{label:<22}{pred:>7.2f}{exp:>7.2f}{d:>+7.2f}{flag}")
    print()


BUILTIN_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",       # ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",     # caffeine
    "CC(=O)Nc1ccc(O)cc1",               # paracetamol
    "Clc1ccccc1",                       # chlorobenzene
    "COc1ccc(CC(C)N)cc1",               #
    "c1ccc2ccccc2c1",                   # naphthalene
    "O=C(O)c1ccccc1O",                  # salicylic acid
    "CCOC(=O)c1ccccc1",                 # ethyl benzoate
    "c1ccncc1",                         # pyridine
    "c1ccoc1",                          # furan
    "c1cc[nH]c1",                       # pyrrole
    "c1ccsc1",                          # thiophene
    "c1ccc2[nH]ccc2c1",                 # indole
    "c1ccc2occc2c1",                    # benzofuran
    "Cc1ccccc1C",                       # o-xylene
    "OCc1ccccc1",                       # benzyl alcohol
    "CC(=O)c1ccccc1",                   # acetophenone
    "N#Cc1ccccc1",                      # benzonitrile
    "CCN(CC)CC",                        # triethylamine
    "OCCO",                             # ethylene glycol
    "CC(C)O",                           # isopropanol
    "FC(F)(F)c1ccccc1",                 # benzotrifluoride
    "O=Cc1ccc(O)cc1",                   # 4-hydroxybenzaldehyde
    "Cc1ccc(N)cc1",                     # p-toluidine
    "CC=CC",                            # 2-butene
    "C=CC=C",                           # butadiene
    "c1ccc(-c2ccccc2)cc1",              # biphenyl
    "NS(=O)(=O)c1ccccc1",               # benzenesulfonamide
    "COc1cc(C=O)ccc1O",                 # vanillin
]


def load_sample(limit: int = 200) -> list[tuple[str, str]]:
    """Return [(id, smiles)] from a local dataset if available, else builtin."""
    try:
        from simulation.graph_io import read_spin_systems, molecule_id
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        candidates = [
            repo / "mol_to_spin_system" / "data" / "spin_systems.json",
        ]
        for p in candidates:
            if p.exists():
                out = []
                for _idx, r in read_spin_systems(p):   # yields (index, record)
                    smi = r.get("smiles")
                    if smi:
                        out.append((molecule_id(r) or smi, smi))
                    if len(out) >= limit:
                        break
                if out:
                    return out
    except Exception as e:  # pragma: no cover
        print(f"(dataset load failed: {e}; using builtin sample)")
    return [(s, s) for s in BUILTIN_SMILES]


def run_coverage(limit: int = 200) -> None:
    sample = load_sample(limit)
    print("=" * 78)
    print(f"COVERAGE  (sample n={len(sample)} molecules)")
    print("=" * 78)
    path_counter: Counter[str] = Counter()
    all_shifts: list[float] = []
    n_mol_ok = 0
    n_embed_fail = 0
    for _id, smi in sample:
        mol = _embed(smi)
        if mol is None:
            n_embed_fail += 1
            continue
        n_mol_ok += 1
        verbose = predict_shifts_pretsch_verbose(mol)
        for _h, (delta, path) in verbose.items():
            # bucket flagged hetero/fused as "real but uncertain"
            base = path.split(":")[0]
            path_counter[base] += 1
            all_shifts.append(delta)

    total_h = sum(path_counter.values())
    fallback_h = sum(v for k, v in path_counter.items() if k.startswith("fallback"))
    real_h = total_h - fallback_h
    print(f"molecules embedded ok : {n_mol_ok}  (embed failed: {n_embed_fail})")
    print(f"protium H total       : {total_h}")
    if total_h:
        print(f"covered by real path  : {real_h}  ({100*real_h/total_h:.1f}%)")
        print(f"fallback (uncovered)  : {fallback_h}  ({100*fallback_h/total_h:.1f}%)")
    print("\nper-path breakdown:")
    for path, n in path_counter.most_common():
        print(f"  {path:<16}{n:>7}  ({100*n/total_h:.1f}%)")
    if all_shifts:
        print("\nshift distribution (ppm):")
        print(f"  min {min(all_shifts):.2f}  p25 {_pct(all_shifts,25):.2f}  "
              f"median {statistics.median(all_shifts):.2f}  "
              f"p75 {_pct(all_shifts,75):.2f}  max {max(all_shifts):.2f}")
        print(f"  mean {statistics.mean(all_shifts):.2f}  stdev {statistics.pstdev(all_shifts):.2f}")
    print()


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


if __name__ == "__main__":
    run_anchors()
    run_coverage()
