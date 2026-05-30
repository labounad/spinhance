# mol_to_matrix

Turn a molecule into a ¹H spin-system matrix: a symmetric block with chemical
shifts (ppm) on the diagonal, inter-group scalar couplings (Hz) off-diagonal,
and a degeneracy vector (protons per group).

## Pipeline

```
SMILES → 3D embed → 1H shifts (NMRShiftDB) → couplings (heuristics) → group → matrix
```

```python
from mol_to_matrix.pipeline import smiles_to_spin_system

system = smiles_to_spin_system("CCO")     # ethanol
system.matrix       # [[1.2, 6.8], [6.8, 3.7]]  (diag = shift, off-diag = J)
system.degeneracy   # [3, 2]   (CH3, CH2)
system.pack(8)      # (8, 9) array: 8x8 matrix + degeneracy column
```

Batch a SMILES CSV to `.npy` (packed 8×9) + `.json` per molecule:

```bash
python -m mol_to_matrix.pipeline input.csv out_dir --smiles-col smiles --id-col id
```

## Modules

| File | Role |
|------|------|
| `shifts.py` | ¹H/¹³C chemical shifts via the NMRShiftDB HOSE-code predictor; 3D embedding helper |
| `geminal.py` | ²J (same carbon), additive Pretsch model |
| `vicinal.py` | ³J (H–C–C–H): Karplus on ring dihedrals, empirical value for rotatable bonds |
| `olefinic.py` | ³J across C=C (cis/trans from geometry) |
| `aromatic.py` | ortho/meta/para ring couplings |
| `long_range.py` | ⁴J allylic |
| `coupling.py` | merges all coupling estimators into one J dict |
| `groups.py` | proton equivalence → spin groups + degeneracy |
| `matrix.py` | assembles the SpinSystem matrix; save/pack |
| `pipeline.py` | SMILES → matrix; batch CSV runner (CLI) |

## External requirement: NMRShiftDB predictor

Chemical-shift prediction shells out to the standalone **NMRShiftDB2 predictor
JARs** (Java) plus their CDK dependencies — these live in a separate SVN
checkout, **outside this repo** (`../nmrshiftdb2` by default, or set
`NMRSHIFTDB_HOME`). See **[SETUP.md](SETUP.md)** for one-time setup steps.

Coupling estimation and grouping need only RDKit, so those modules (and their
tests) run without Java; the shift/matrix steps require the predictor.

## Heuristic sources

Coupling values are transcribed from:
- Pretsch, Bühlmann, Badertscher, *Tables of Spectral Data for Structure
  Determination of Organic Compounds* (Springer, 2009) — §5.1.2 (geminal,
  vicinal/Karplus), §5.2 (olefinic).
- Hans Reich, *5-HMR* proton NMR notes (organicchemistrydata.org).

Reference PDFs are kept under `ref/` for development and are not committed.
