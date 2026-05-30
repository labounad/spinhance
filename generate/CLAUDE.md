# generate/ — AI context for Claude

## What this module does

Task 1 of SpinHance: screen ChEMBL (~2.3 M compounds) down to molecules
with **exactly 8 magnetically distinct ¹H spin groups**.  These molecules
feed Task 2 (shift + J matrix generation) and ultimately Task 3 (MNova
spin simulation).

The number 8 lives in **`config.py → N_SPIN_GROUPS`**.  Everything else
derives from it automatically.

---

## Pipeline stages

```
ChEMBL chemreps.txt  (~738 MB)
        │
        ▼  generate/chembl_filter.py
        │  Heuristic: n_proton_bearing_c ≤ 8  AND  n_ch_protons ≥ 8
        │  Fast O(n) stream; ~10 min on full ChEMBL
        ▼
8spin.csv (intermediate not kept)  (~1 M rows)
        │
        ▼  generate/screen.py  (calls spin_equivalence.py)
        │  Exact 3-D deuterium test; ~100–500 ms/mol
        ▼
8spin.csv  (~4 K rows — the final dataset)
        │
        ▼  generate/viewer.py  (triage GUI)
```

---

## The deuterium substitution test

### Invariant

Two C-H protons belong to the **same spin group** iff and only iff
replacing either with deuterium (isotope = 2) produces **identical
canonical isomeric SMILES**.  Every other pair — enantiotopic,
diastereotopic — belongs to **separate spin groups**.

### Why 3-D embedding is non-negotiable

In 2-D SMILES, a methylene CH₂ adjacent to a defined stereocentre creates
a new CHD stereocentre whose configuration cannot be determined from the
molecular graph: both Ha→D and Hb→D produce the same canonical SMILES,
collapsing two diastereotopic protons into one group (false negative).

The fix: embed the molecule in 3-D first, then call
**`Chem.AssignStereochemistryFrom3D`** (not `AssignStereochemistry`) after
each H→D substitution.  This reads the atom coordinates to assign the
chiral tag at the new CHD centre, producing distinct SMILES for the two
diastereomers.

### The critical call

```python
# In spin_equivalence._assign_stereo():
Chem.AssignStereochemistryFrom3D(mol, confId=conf_id, replaceExistingTags=True)
Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
```

`AssignStereochemistry` alone (the old path) silently fails here.
Do **not** revert to `AssignStereochemistry(cleanIt=True)` — that erases
the existing `@`/`@@` tags before reassigning, and without 3-D coordinates
the new CHD centre comes back as "unspecified".

### Exchangeable protons

Strip all H atoms bonded to N, O, or S **before** the test.  They exchange
rapidly in solution and must not be counted as spin groups.  Retaining them
also allows heteroatom geometry to perturb stereocentre assignment.

---

## Enantiotopic protons: design decision

By user requirement, enantiotopic protons are counted as **separate** spin
groups.  This is correct for the second-order coupling regime (90 MHz, low
field) targeted by this project: magnetically inequivalent protons in the
same chemical environment produce observable strong-coupling effects.

Do NOT merge enantiomeric SMILES (i.e., do NOT set `merge_enantiotopic=True`
as in the reference implementation in the pasted equivalence module).

---

## Known failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `embed_3d` returns `has_3d=False` | ETKDG fails (large macrocycle, strained ring) | Molecule skipped; stereo-perception falls back to 2-D — may under-count diastereotopic methylenes |
| CHD centre still "unspecified" | Called `AssignStereochemistry` instead of `AssignStereochemistryFrom3D` | Use `_assign_stereo()` from `spin_equivalence.py` |
| Molecule incorrectly accepted with N-1 groups | OH/NH not stripped | Ensure `strip_exchangeable_protons` runs first |
| Very slow screening | 3-D embedding is O(n·mol_size) | Use `--workers` in a future parallel variant; current single-process rate ≈ 5–10 mol/s |

---

## File map

```
generate/
├── config.py           ← N_SPIN_GROUPS and all thresholds (edit here only)
├── spin_equivalence.py ← core algorithm (embed, strip, test, count)
├── chembl_filter.py    ← fast heuristic pre-filter (streams ChEMBL)
├── screen.py           ← applies spin_equivalence to filter output
├── viewer.py           ← Tkinter gallery + deuterium triage GUI
├── cli.py              ← spinhance-gen CLI (filter / screen / view)
├── __init__.py         ← public API re-exports
├── CLAUDE.md           ← this file
├── README.md           ← human-facing docs
└── tests/
    ├── __init__.py
    └── test_spin_equivalence.py   ← pytest suite (no external deps)
```

---

## Data files (gitignored)

| File | Size | Description |
|------|------|-------------|
| `chembl/chembl_37_chemreps.txt` | 738 MB | ChEMBL v37 tab-separated chemreps |
| `data/8spin.csv` | varies | Final 8-group dataset (output of `run`) |

---

## Running the tests

```bash
conda activate spinhance
pytest generate/tests/ -v
```

All tests are pure (RDKit + stdlib only).  Expect ~30 s for the full suite
due to 3-D embedding.
