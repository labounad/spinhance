"""Tests for the Pretsch (2009) ¹H chemical-shift engine.

These guard the validation gate: anchor compounds must predict within tolerance,
coverage on a small real-ish sample must stay high, and the engine must obey its
contract (protium-only, value shape, path precedence).

Run:  PYTHONPATH=. micromamba run -n spinhance python -m pytest tests/test_shifts_pretsch.py -v
"""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from mol_to_spin_system.shifts_pretsch import (
    AROM_INCR,
    predict_shifts_pretsch,
    predict_shifts_pretsch_verbose,
)


def _embed(smiles: str, seed: int = 0xF00D) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad SMILES {smiles!r}"
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        AllChem.Compute2DCoords(mol)
    else:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    return mol


def _mean_for_smarts(mol: Chem.Mol, smarts: str, h_query: int = 0) -> float:
    """Mean predicted δ over all protons on the SMARTS query atom `h_query`."""
    preds = predict_shifts_pretsch(mol)
    patt = Chem.MolFromSmarts(smarts)
    vals = []
    for match in mol.GetSubstructMatches(patt):
        head = mol.GetAtomWithIdx(match[h_query])
        for nb in head.GetNeighbors():
            if nb.GetAtomicNum() == 1 and nb.GetIdx() in preds:
                vals.append(preds[nb.GetIdx()])
    assert vals, f"no protons matched {smarts!r} in {Chem.MolToSmiles(mol)}"
    return sum(vals) / len(vals)


# (name, SMILES, SMARTS-for-H-parent, h_query_idx, expected_ppm, tol)
ANCHOR_CASES = [
    ("benzene_ArH",     "c1ccccc1",            "[cH]",        0, 7.26, 0.3),
    ("toluene_CH3",     "Cc1ccccc1",           "[CH3]",       0, 2.34, 0.3),
    ("ethanol_CH3",     "CCO",                 "[CH3]",       0, 1.20, 0.3),
    ("ethanol_CH2",     "CCO",                 "[CH2]O",      0, 3.70, 0.3),
    ("aceticacid_CH3",  "CC(=O)O",             "[CH3]",       0, 2.10, 0.3),
    ("aceticacid_COOH", "CC(=O)O",             "[CX3](=O)[OX2H1]", 2, 11.5, 1.0),
    ("acetaldehyde_CHO","CC=O",                "[CX3H1]=O",   0, 9.7,  0.5),
    ("benzaldehyde_CHO","O=Cc1ccccc1",         "[CX3H1]=O",   0, 9.7,  0.5),
    ("anisole_OCH3",    "COc1ccccc1",          "[CH3]O",      0, 3.78, 0.3),
    ("furan_H2",        "c1ccoc1",             "[cH]o",       0, 7.42, 0.3),
    ("furan_H3",        "c1ccoc1",             "[cH][cH]",    0, 6.38, 0.3),
    ("pyridine_H2",     "c1ccncc1",            "[cH]n",       0, 8.59, 0.3),
    ("thiophene_H2",    "c1ccsc1",             "[cH]s",       0, 7.31, 0.3),
    ("ethylene",        "C=C",                 "[CH2]=C",     0, 5.25, 0.3),
    ("chloroethane_CH2","CCCl",                "[CH2]Cl",     0, 3.47, 0.4),
    ("chloroethane_CH3","CCCl",                "[CH3]",       0, 1.48, 0.3),
]


@pytest.mark.parametrize("name,smi,smarts,hq,exp,tol", ANCHOR_CASES,
                         ids=[c[0] for c in ANCHOR_CASES])
def test_anchor_within_tolerance(name, smi, smarts, hq, exp, tol):
    mol = _embed(smi)
    pred = _mean_for_smarts(mol, smarts, hq)
    assert abs(pred - exp) <= tol, f"{name}: predicted {pred:.2f}, expected {exp:.2f} (tol {tol})"


def test_protium_only_and_value_shape():
    """Only protium H's appear; values are floats; deuterium is skipped."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    # mark one H as deuterium
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            atom.SetIsotope(2)
            d_idx = atom.GetIdx()
            break
    AllChem.Compute2DCoords(mol)
    out = predict_shifts_pretsch(mol)
    assert d_idx not in out, "deuterium must be skipped"
    assert all(isinstance(v, float) for v in out.values())
    # every reported key is a protium H
    for idx in out:
        a = mol.GetAtomWithIdx(idx)
        assert a.GetAtomicNum() == 1 and a.GetIsotope() in (0, 1)


def test_drop_in_shape_matches_legacy_keys():
    """predict_shifts_pretsch keys are RDKit H atom indices (drop-in mapping)."""
    mol = _embed("CCO")
    out = predict_shifts_pretsch(mol)
    n_h = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 1)
    assert len(out) == n_h
    assert set(out) == {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1}


def test_special_group_precedence():
    """Aldehyde / COOH H's come from the special path, not fallback/alkane."""
    mol = _embed("O=Cc1ccccc1")
    verbose = predict_shifts_pretsch_verbose(mol)
    cho = [p for d, p in verbose.values() if d > 9.0]
    assert any(p == "special" for p in cho), "aldehyde CHO should be a special path"


def test_coverage_on_sample_is_high():
    """At least 80% of protium H on a drug-like sample get a real (non-fallback)
    path — the engine's headline coverage metric for the gate."""
    sample = [
        "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen
        "CC(=O)Nc1ccc(O)cc1",             # paracetamol
        "COc1ccc(CC(C)N)cc1",
        "CCOC(=O)c1ccccc1",               # ethyl benzoate
        "c1ccncc1", "c1ccoc1", "c1ccsc1",
        "OCc1ccccc1", "CC(=O)c1ccccc1",
        "N#Cc1ccccc1", "OCCO", "CC(C)O",
        "Cc1ccc(N)cc1", "CC=CC",
    ]
    total = real = 0
    for smi in sample:
        mol = _embed(smi)
        for _h, (_d, path) in predict_shifts_pretsch_verbose(mol).items():
            total += 1
            if not path.startswith("fallback"):
                real += 1
    assert total > 0
    frac = real / total
    assert frac >= 0.80, f"real-path coverage only {frac:.1%} (< 80%)"


def test_aromatic_increment_additivity():
    """A disubstituted benzene shift is the base plus both substituents'
    position-resolved increments (sanity on the additive core)."""
    # p-nitrotoluene: ring H ortho to NO2 (and meta to CH3) should be ~8.0
    mol = _embed("Cc1ccc([N+](=O)[O-])cc1")
    preds = predict_shifts_pretsch(mol)
    arom = [v for i, v in preds.items()
            if mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIsAromatic()]
    # two pairs of aromatic H; the NO2-adjacent pair must be downfield (>7.7)
    assert max(arom) > 7.7, f"expected a downfield NO2-ortho H, got {sorted(arom)}"
    # CH3-ortho / NO2-meta H: 7.34 - 0.17 (CH3 ortho) + 0.26 (NO2 meta) = 7.43
    assert min(arom) < 7.5, f"expected an upfield CH3-ortho H, got {sorted(arom)}"


# ─── audit PIPELINE_AUDIT_2 §5-B regression guards ──────────────────────────

def test_pyrazine_base_shift_b2():
    """B2: pyrazine's 8.63 base must be reachable (the `_NAME` key
    `(6,((1,7),(4,7)))` was missing → it fell to the 7.30 aromatic default)."""
    mol = _embed("c1cnccn1")
    verbose = predict_shifts_pretsch_verbose(mol)
    arom = [(d, p) for _h, (d, p) in verbose.items()
            if mol.GetAtomWithIdx(_h).GetNeighbors()[0].GetIsAromatic()]
    assert arom, "no aromatic protons found for pyrazine"
    # all four equivalent ring H -> ~8.63, on the (unflagged) hetero path
    for d, p in arom:
        assert abs(d - 8.63) <= 0.1, f"pyrazine H predicted {d}, expected ~8.63"
        assert p == "hetero", f"pyrazine should hit the hetero path, got {p!r}"


def test_furfural_not_equal_2_methylfuran_b1():
    """B1: a substituent anywhere on a hetero ring (furfural 2-CHO) must change
    the predicted ring shifts — previously the substituent-blind flag returned
    the bare furan-parent shifts identical to 2-methylfuran."""
    def arom_shifts(smi):
        mol = _embed(smi)
        v = predict_shifts_pretsch_verbose(mol)
        return sorted(round(d, 3) for _h, (d, p) in v.items()
                      if mol.GetAtomWithIdx(_h).GetNeighbors()[0].GetIsAromatic())

    furfural = arom_shifts("O=Cc1ccco1")
    methylfuran = arom_shifts("Cc1ccco1")
    assert furfural != methylfuran, (
        f"furfural {furfural} must differ from 2-methylfuran {methylfuran}")
    # CHO is strongly deshielding; furfural's most-downfield ring H must exceed
    # the parent furan H2 (7.42) and 2-methylfuran's most-downfield ring H.
    assert max(furfural) > max(methylfuran)


def test_substituted_hetero_ring_is_flagged_b1():
    """B1: any substituted supported hetero ring is flagged `hetero?`, while the
    bare parent stays the confident `hetero` path."""
    mol = _embed("O=Cc1ccco1")  # furfural
    v = predict_shifts_pretsch_verbose(mol)
    ring_paths = [p for _h, (d, p) in v.items()
                  if mol.GetAtomWithIdx(_h).GetNeighbors()[0].GetIsAromatic()]
    assert ring_paths and all(p == "hetero?" for p in ring_paths), ring_paths

    bare = _embed("c1ccoc1")  # furan
    vb = predict_shifts_pretsch_verbose(bare)
    bare_paths = [p for _h, (d, p) in vb.items()
                  if bare.GetAtomWithIdx(_h).GetNeighbors()[0].GetIsAromatic()]
    assert bare_paths and all(p == "hetero" for p in bare_paths), bare_paths


def test_corrected_arom_incr_rows_b8():
    """B8: the placeholder AROM_INCR rows for vinyl and nitrile must hold the
    real Pretsch increments, not copies of CH2Cl / NCS."""
    assert AROM_INCR["CH=CH2"] == (0.06, -0.03, -0.10)
    assert AROM_INCR["C#N"] == (0.36, 0.18, 0.28)
    # and they must no longer equal the rows they were copied from
    assert AROM_INCR["CH=CH2"] != AROM_INCR["CH2Cl"]
    assert AROM_INCR["C#N"] != AROM_INCR["NCS"]


def test_benzonitrile_uses_nitrile_increment_b8():
    """B8: benzonitrile ortho H uses the corrected C#N ortho increment (+0.36),
    so ortho H ≈ 7.34 + 0.36 = 7.70 (not the old NCS-copy 7.66)."""
    mol = _embed("N#Cc1ccccc1")
    preds = predict_shifts_pretsch(mol)
    arom = sorted(v for i, v in preds.items()
                  if mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIsAromatic())
    # ortho pair is the most downfield: 7.34 + 0.36 = 7.70
    assert abs(max(arom) - 7.70) <= 0.02, f"benzonitrile ortho H {arom}"
