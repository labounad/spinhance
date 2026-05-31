# generate/ — Task 1: Molecule Screening

Screens the ChEMBL compound database (~2.3 M molecules) to identify small
organic molecules with **exactly 8 magnetically distinct ¹H spin groups** —
the input representation required by the downstream NMR simulation (Task 3)
and machine-learning (Task 4) modules.

---

## Quick start

```bash
conda activate spinhance

# 1. Screen ChEMBL → chembl_8spin.csv + chembl_8spin.xyz.gz in one pass
#    (takes several hours on the full dataset)
python generate/cli.py run

# 2. Browse results in the interactive viewer
python generate/cli.py view
```

`run` produces **both** the CSV and the 3-D annotated XYZ file in a single
pass.  It embeds and classifies each molecule once — the old separate `xyz`
step re-embedded every kept molecule a second time.  Pass `--no-xyz` to write
only the CSV.

The standalone `xyz` command is still available to (re)generate `chembl_8spin.xyz.gz`
from an existing CSV:

```bash
python generate/cli.py xyz          # chembl_8spin.csv → chembl_8spin.xyz.gz
```

All subcommands accept `--help` for full option listings.

---

## Compound sources

`run --source` selects the input format; the screening itself is identical
across sources.

| `--source` | File | Notes |
|---|---|---|
| `chembl` *(default)* | `chembl_XX_chemreps.txt` | TSV with header; InChIKey read from the file |
| `pubchem` | `CID-SMILES.gz` | `<CID>\t<SMILES>`, ~120 M rows; `source_id` = `CID<n>` |
| `zinc` | `*.smi` tranches | `<SMILES> <ZINC_id>` |
| `smiles` | any `.smi`/`.txt` | first token SMILES, optional id |

`.gz` inputs are read transparently.  For sources without an InChIKey column
(PubChem, ZINC) it is computed only for the molecules that pass — never for the
whole database.  A heavy-atom cap (`--max-heavy-atoms`, default 50) skips
peptides/polymers/macrocycles before the slow 3-D embedding.

```bash
# Screen all of PubChem locally (slow — see HPC sharding below for scale)
python generate/cli.py run --source pubchem --chembl CID-SMILES.gz \
  --output pubchem_8spin.csv --xyz-output pubchem_8spin.xyz.gz
```

PubChem's `CID-SMILES.gz` (~1.5 GB) is at
`https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz`.

### Screening at scale (HPC Slurm array)

For a whole-database screen, split the input across array tasks
(`index % num_shards == shard_index`); each task writes its own
`part_<id>.csv` / `part_<id>.xyz.gz`, then `merge` combines them.

```bash
# Submit the array (num_shards = array size, set via --array in the script)
sbatch --export=ALL,INPUT=$HOME/pubchem/CID-SMILES.gz,SRC=pubchem \
    generate/slurm/screen_array.slurm

# After it finishes, merge the shards into one dataset
python generate/cli.py merge data/screen/shards \
    --output data/screen/pubchem_8spin.csv \
    --xyz-output data/screen/pubchem_8spin.xyz.gz
```

The screen is embed-bound (the 3-D step dwarfs parsing), so 16 cores/task keeps
each shard near 100 % CPU utilisation.

---

## Pipeline

```
ChEMBL chemreps.txt  (738 MB)
         │
         ▼  heuristic pre-filter  (pipeline.py)
         │  fast atom-count check: n_CH_carbons ≤ 8  AND  n_CH_protons ≥ 8
         │  eliminates ~95 % of ChEMBL in O(1) per molecule
         │
         ▼  exact 3-D deuterium test  (spin_equivalence.py)
         │  ETKDG v3 + MMFF94 embedding → H→D substitution per proton
         │  AssignStereochemistryFrom3D → canonical isomeric SMILES
         │  HARD/SOFT/NONE tier assignment + magnetic equivalence check
         │  (single embed per molecule; reused for both outputs below)
         │
         ├─▶  chembl_8spin.csv     (~60 000 molecules)  ──▶  viewer.py  triage GUI
         └─▶  chembl_8spin.xyz.gz  3-D XYZ with spin-group annotations (xyz_writer.build_xyz_block)
```

The deuterium test is the expensive step (ETKDG embedding ≫ everything else).
`run` calls it **once** per molecule via `classify_spin_groups` and renders
both the CSV row and the XYZ block from that single classification, so adding
the XYZ output costs essentially nothing beyond the InChI string.

---

## Spin-group classification

Every C-H proton is assigned to exactly one of three tiers:

| Tier | Meaning | Example | Count rule |
|------|---------|---------|------------|
| **HARD** | Homotopic AND magnetically equivalent | CH₃, t-Bu, 1,3,5-symmetric aromatic pair | 1 spin group for N protons |
| **SOFT** | Chemically equivalent but magnetically inequivalent | Enantiotopic CH₂, AA′BB′ aromatic | 1 spin group per proton; shared averaged shift |
| **NONE** | Chemically distinct singleton | Isolated aromatic CH, diastereotopic CH₂ | 1 spin group |

Total spin groups = `n_HARD_groups + n_non_HARD_protons`.

### Why HARD ≠ "same D-substitution SMILES"

Two protons are HARD iff they are (1) homotopic — identical D-substitution
canonical SMILES — AND (2) magnetically equivalent — every external H is at
the same shortest-bond-path distance from each member.  This correctly
identifies methyl groups AND symmetric aromatic pairs (e.g. H4/H6 in a
1,3,5-trisubstituted benzene) while rejecting toluene's two ortho-H, which
are homotopic but form an AA′ system (different coupling patterns to meta-H).

### Why 3-D embedding is required

In 2-D SMILES, substituting either H of a diastereotopic CH₂ with D creates
a new CHD stereocentre whose configuration cannot be inferred from the
molecular graph — both substitutions produce the same canonical SMILES,
falsely merging two inequivalent protons.  Embedding the molecule with
ETKDG v3 + MMFF94 and calling `AssignStereochemistryFrom3D` (not
`AssignStereochemistry`) reads the 3-D atom positions to assign the CHD
centre correctly.

### Why exchangeable protons must stay in the molecule

Stripping N-H/O-H/S-H *before* the D-substitution test removes structural
context that can break apparent ring symmetry.  Classic failure: removing the
indole N-H makes the two ring-junction carbons appear equivalent, propagating
false symmetry to the aromatic CH protons.  The fix: keep exchangeable protons
in the molecule during the test; exclude them from the candidate list via
`candidate_atoms`.

---

## Output files

### `chembl_8spin.csv`

Produced by `python generate/cli.py run`.

| Column | Type | Description |
|--------|------|-------------|
| `chembl_id` | str | ChEMBL compound identifier |
| `smiles` | str | Canonical SMILES |
| `inchikey` | str | Standard InChIKey |
| `n_groups` | int | Number of spin groups (always 8) |
| `group_sizes` | str | Semicolon-separated proton counts per group, descending |

### `chembl_8spin.xyz.gz`

Produced by `python generate/cli.py xyz`.  Gzip-compressed multi-XYZ file
(~35–40 MB for ~60 000 molecules).  See [`XYZ_FORMAT.md`](XYZ_FORMAT.md) for
the full format specification including the spin-group annotation columns.

Each H atom gets two annotation columns:

```
H   x.xxxxxx   y.yyyyyy   z.zzzzzz   B   S2
│                                    │   ││
│                                    │   │└── class number (shared shift)
│                                    │   └─── tier: H=HARD  S=SOFT  N=NONE
│                                    └─────── spin-group letter (A–Z, AA…)
└──────────────────────────────────────────── element + 3-D coords (Å)
```

---

## Interactive viewer

```bash
python generate/cli.py view [--file PATH] [--n N] [--seed SEED]
```

Single-window app: 4×4 gallery on the left, detail panel on the right.

- **Gallery**: click any thumbnail to load that molecule in the detail panel.
  A blue ring marks the currently displayed molecule.
- **Detail panel**: 2-D structure rendered at 2× resolution (LANCZOS downscale
  for ChemDraw-quality crispness) with spin-group letter annotations on every
  H-bearing carbon.  HARD groups are steel blue; each distinct SOFT equivalence
  class gets its own colour; NONE groups are grey.
- **Spin-group table**: one row per spin group showing label, tier (H/S/−), and
  proton count.  SOFT protons sharing the same averaged shift share a background
  colour.
- **Bidirectional hover**: mouse over a table row → that group's atoms highlight
  on the structure.  Mouse over the structure → the nearest atom's row
  highlights in the table.
- **SMILES entry**: type or paste any SMILES in the top bar and click **View**
  to inspect an arbitrary molecule without needing it in the CSV.

---

## CLI reference

```
spinhance-gen run   [--chembl PATH] [--output PATH] [--n-groups N]
                    [--workers N]   [--chunk-size N]

    Stream ChEMBL, apply heuristic pre-filter then 3-D deuterium test.
    Workers run the expensive embedding+test step in parallel.
    Default output: generate/data/chembl_8spin.csv

spinhance-gen view  [--file PATH] [--n N] [--seed SEED]

    Launch the interactive gallery viewer.
    Default file: generate/data/chembl_8spin.csv

spinhance-gen xyz   [--input PATH] [--output PATH] [--workers N]

    Convert chembl_8spin.csv to a gzip multi-XYZ file with spin annotations.
    Default output: generate/data/chembl_8spin.xyz.gz
```

---

## Module structure

```
generate/
├── config.py            N_SPIN_GROUPS = 8 and all thresholds — edit here only
├── spin_equivalence.py  Core algorithm: embed, classify, HARD/SOFT/NONE tiers
├── pipeline.py          Multiprocess ChEMBL streaming pipeline
├── xyz_writer.py        Gzip multi-XYZ output with spin-group annotations
├── viewer.py            Single-window Tkinter triage GUI
├── cli.py               CLI entry point (run / view / xyz)
├── __init__.py          Public API re-exports
├── XYZ_FORMAT.md        Full XYZ format specification for downstream modules
├── CLAUDE.md            AI-facing context (algorithm details, failure modes)
├── README.md            This file
└── tests/
    ├── __init__.py
    └── test_spin_equivalence.py   21 pytest tests (RDKit + stdlib only)
```

---

## Running the tests

```bash
conda activate spinhance
pytest generate/tests/ -v
```

All 21 tests are pure Python + RDKit (no PyTorch, no MNova, no network).
Expect ~30 s due to 3-D conformer generation.

---

## Changing the target spin-group count

Everything derives from a single constant:

```python
# generate/config.py
N_SPIN_GROUPS = 8
```

Change it once to retarget the entire pipeline — pre-filter thresholds,
screening filter, viewer default file, and XYZ annotation labels all update
automatically.
