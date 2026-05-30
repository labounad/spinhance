# SpinHance

**Goal:** Automatically extract ¹H chemical shifts and coupling constants from low-field ¹H NMR spectra of small molecules using deep learning — enabling reconstruction of the spectrum at any field strength.

---

## Problem Statement

At low field (e.g. 90 MHz), ¹H NMR spectra are often non-first-order: peaks overlap and standard peak-picking fails. We train a neural network to invert the spectrum back to the underlying spin-system parameters (shifts δ in ppm, coupling constants *J* in Hz, and spin-group degeneracies), represented as a symmetric 8×8 matrix + 8×1 degeneracy vector. Given those parameters, any field strength can be simulated exactly.

---

## Repository Structure

```
spinhance/
├── generate/          # Task 1 — molecule generation & SMILES filtering
├── mol_to_matrix/     # Task 2 — 3D embedding + heuristic J/shift matrix
├── simulation/        # Task 3 — MNova spin simulation pipeline
├── ml_model/          # Task 4 — deep learning model (spectrum → matrix)
├── data/
│   ├── raw/           # SMILES lists, raw SDF files (gitignored if large)
│   └── processed/     # shift+J matrices, simulated spectra
├── environment.yml    # Shared micromamba environment (Python 3.14)
└── README.md
```

---

## Spin-System Representation

Each molecule is encoded as an **8×9 block**:

| | Group 1 | Group 2 | … | Group 8 | Degeneracy |
|---|---|---|---|---|---|
| **Group 1** | δ₁ (ppm) | J₁₂ (Hz) | … | J₁₈ (Hz) | n₁ |
| **Group 2** | J₂₁ | δ₂ | … | J₂₈ (Hz) | n₂ |
| … | | | | | |
| **Group 8** | J₈₁ | … | δ₈ | | n₈ |

- **Diagonal:** chemical shifts in ppm (field-independent)
- **Off-diagonal:** scalar coupling constants in Hz (field-independent)
- **Degeneracy vector:** number of protons per spin group (e.g. 3 for CH₃, 9 for *t*Bu)
- The matrix is symmetric; labels are arbitrary (invariant under S₈ permutation)

This is equivalently an **undirected labeled graph**: nodes carry (δ, n), edges carry *J*.

---

## Tasks

### Task 1 — GENERATE (`generate/`) 
**Screen and filter molecules to exactly 8 magnetically distinct spin groups.**

#### Subtasks
- [ ] Download/sample SMILES from USPTO or another large public database
- [ ] Parse SMILES with RDKit; assign CIP stereochemistry
- [ ] Identify chemically **and** magnetically equivalent proton groups (homotopic/enantiotopic analysis)
- [ ] Filter to molecules with **exactly 8** hard-equivalent spin groups
- [ ] Output: `data/raw/smiles_8group.csv` (SMILES, InChIKey, n_groups, group_sizes)

#### Key decisions
- Magnetic equivalence requires symmetry analysis beyond simple graph isomorphism — use RDKit `GetSymmSSSR` + point-group or topological equivalence
- Consider excluding paramagnetic, very flexible, or >50-heavy-atom molecules

---

### Task 2 — MOL → MATRIX (`mol_to_matrix/`)
**Generate physically plausible shift+J matrices from 3D-embedded molecules.**

#### Subtasks
- [ ] Embed molecules to 3D with ETKDG (RDKit) + MMFF94 minimization
- [ ] Assign ¹H chemical shifts via heuristics (ring-current corrections, α/β substituent tables, or a pre-trained predictor like NMRShiftDB lookup)
- [ ] Compute dihedral angles between each proton pair for Karplus-based ³*J* estimation
- [ ] Apply literature tables for geminal (²*J*), vinyl, aryl, benzylic, allylic, long-range (⁴*J*) couplings
- [ ] Zero out negligible couplings (|*J*| < 0.3 Hz threshold)
- [ ] Assemble symmetric 8×8 *J*-matrix + δ-diagonal + degeneracy vector
- [ ] Output: `data/processed/matrices/` — one `.npy` or `.json` per molecule

#### Key references
- Karplus (1959) — vicinal *³J* vs dihedral
- Altona & Hasnoot — modified Karplus for HCCH
- Standard geminal/aromatic coupling tables (see `mol_to_matrix/references/`)

---

### Task 3 — SIMULATION (`simulation/`)
**Simulate accurate ¹H NMR spectra at 90 MHz and 600 MHz using MNova.**

**Status: working.** See `simulation/README.md` for the architecture diagram,
usage, and one-time MNova setup.

#### Subtasks
- [x] Convert shift+J matrix → MNova spin-system XML (`simulation/xml_io.py`)
- [x] MNova JS batch script to load XMLs, run QM spin simulation, export spectra (`simulation/mnova_scripts/spinhanceBatch.qs`)
- [x] Simulate at **90 MHz** (low-field, non-first-order) and **600 MHz** (high-field, reference)
- [x] Post-process: normalize integral to 1, save `.npy` (`simulation/pipeline.py`)
- [x] Output: `data/processed/spectra/<field>MHz/` — 2¹⁴-point intensity arrays (0–12 ppm)

#### Notes
- MNova spin simulator handles strongly-coupled spin systems exactly
- All team members need an active MNova license
- Entry point: `python -m simulation.cli run` (see `simulation/README.md`)

---

### Task 4 — ML MODEL (`ml_model/`)
**Train a neural network: normalized spectrum → shift+J+degeneracy matrix.**

#### Subtasks
- [ ] Define input: 16384-point spectrum vector (90 MHz simulations)
- [ ] Define output: 8×9 matrix (symmetric 8×8 + degeneracy column); handle S₈ permutation invariance in loss
- [ ] Implement permutation-invariant loss (e.g. Hungarian matching on rows)
- [ ] Architecture search: 1D-CNN encoder → transformer/MLP decoder; or direct regression
- [ ] Train/val/test split (e.g. 80/10/10 by molecule)
- [ ] Evaluation metrics: MAE on δ (ppm), MAE on *J* (Hz), fraction of *J* within 1 Hz
- [ ] Stretch goal: condition on field strength to generalize beyond 90 MHz

#### Key challenges
- Permutation invariance of spin-group labels
- Dynamic range: δ spans ~10 ppm, *J* spans 0–20 Hz
- Strongly-coupled systems have non-additive line positions

---

## Environment Setup

```bash
# Install micromamba if needed
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

micromamba env create -f environment.yml
micromamba activate spinhance
```

See `environment.yml` for the full dependency list.

---

## Division of Labor

| Task | Module | Assignee |
|------|--------|----------|
| 1 — Generate | `generate/` | TBD |
| 2 — Mol → Matrix | `mol_to_matrix/` | TBD |
| 3 — Simulation | `simulation/` | TBD |
| 4 — ML Model | `ml_model/` | TBD |

Update this table once responsibilities are assigned.

---

## Data Flow

```
USPTO / pubchem SMILES
        │
        ▼
  [generate/]  ──→  smiles_8group.csv
        │
        ▼
  [mol_to_matrix/]  ──→  matrices/*.npy
        │
        ▼
  [simulation/]  ──→  spectra/{90MHz,600MHz}/*.npy
        │
        ▼
  [ml_model/]  ──→  trained model  ──→  predicted matrix from spectrum
```

---

## Reference XML

`predicted_mnova_1h (10).xml` in the repo root shows the MNova spin-system XML format used as input to Task 3.
