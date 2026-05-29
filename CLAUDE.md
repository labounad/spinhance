# SpinHance — Project Context for Claude

## What This Project Is

SpinHance is a hackathon project to automatically extract ¹H chemical shifts and scalar coupling constants from **low-field (100 MHz) ¹H NMR spectra** of small molecules using deep learning. The key insight: at low field, spin systems are strongly coupled (non-first-order), making simple peak-picking fail. We train a neural network to invert the spectrum back to the underlying spin-system parameters.

**Team of 3. Lucas (labounader@scripps.edu) owns Task 3 (simulation) and is the user in this project.**

---

## Repository

`/Users/labounader/Documents/Claude/Projects/spinhance/`

Git repo, `main` branch, initialized with the following structure:

```
spinhance/
├── generate/            # Task 1 — molecule screening (teammate)
├── mol_to_matrix/       # Task 2 — 3D embedding + J/shift heuristics (teammate)
├── simulation/          # Task 3 — MNova spin simulation pipeline (LUCAS)
│   ├── xml_utils.py         # Python: build/patch mnova-spinsim XML files
│   ├── spinhanceBatch.qs    # MNova JavaScript batch script (see caveats below)
│   ├── batch_simulate.py    # MNova Python batch script (abandoned — see below)
│   ├── run_batch.py         # Python orchestrator for the full pipeline
│   ├── test_xml_utils.py    # pytest suite for xml_utils.py
│   └── discover_mnova_api*.py  # API discovery scripts (scratch, not production)
├── ml_model/            # Task 4 — deep learning model (teammate)
├── data/
│   ├── raw/             # SMILES lists (gitignored if large)
│   └── processed/       # matrices, XMLs, spectra
├── environment.yml      # micromamba env, Python 3.14, conda-forge only
├── setup_env.sh         # creates/updates the micromamba environment
├── predicted_mnova_1h (10).xml  # example mnova-spinsim XML (reference format)
└── README.md            # full project documentation
```

---

## Spin-System Representation

Each molecule → **8×9 block**:
- **Diagonal (8×8):** chemical shifts δ in ppm (field-independent)
- **Off-diagonal (8×8):** scalar couplings *J* in Hz (field-independent, symmetric)
- **Extra column (8×1):** degeneracy (number of protons per spin group; e.g. 3 for CH₃)

This is an undirected labeled graph: nodes carry (δ, n), edges carry *J*. Labels are arbitrary (invariant under S₈ permutation). We restrict to molecules with **exactly 8 magnetically distinct spin groups**.

The XML format (`predicted_mnova_1h (10).xml`) encodes this as `<mnova-spinsim>` with `<group>` elements containing `<shift>`, `<jCoupling name="X">`, and a `number=` attribute for degeneracy.

---

## The Four Tasks

### Task 1 — GENERATE (`generate/`)
Screen SMILES from USPTO/PubChem, filter to exactly 8 hard-equivalent (chemically + magnetically equivalent) spin groups using RDKit. Output: `data/raw/smiles_8group.csv`.

### Task 2 — MOL → MATRIX (`mol_to_matrix/`)
3D embed with ETKDG + MMFF94 (RDKit), assign shifts via heuristic tables, compute *J* couplings via Karplus equations and geminal/aryl/vinyl/benzylic tables. Assemble the 8×8 J-matrix + degeneracy vector. Output: `data/processed/matrices/*.npy`.

### Task 3 — SIMULATION (`simulation/`) ← LUCAS'S TASK
Take the shift+J matrix, convert to MNova XML format, run MNova's quantum spin simulator at **100 MHz** (low-field, non-first-order) and **600 MHz** (high-field, reference). Output: 2¹⁴-point normalized intensity arrays as `.npy` files. See detailed status below.

### Task 4 — ML MODEL (`ml_model/`)
Train a neural network: input = 16384-point normalized spectrum (100 MHz), output = 8×9 shift+J+degeneracy matrix. Key challenge: permutation invariance of spin-group labels (Hungarian matching loss).

---

## Task 3 — Simulation: Detailed Status

### Environment
- **MNova version:** MestReNova v16.0.0-39276 (macOS, x86_64)
- **Python env:** micromamba `spinhance`, Python 3.14, conda-forge

### What Works

**`xml_utils.py` — fully working.** Builds and patches mnova-spinsim XML files:
```python
from simulation.xml_utils import matrix_to_xml, save_xml, patch_frequency, generate_field_pair

# Build XML from matrix
tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=100.0)
save_xml(tree, "output.xml")

# Patch an existing XML to a different field
patched = patch_frequency("existing.xml", 100.0)
save_xml(patched, "patched_100MHz.xml")

# Generate both field variants at once
lo, hi = generate_field_pair("source.xml", "output_dir/", stem="mol_001")
```

**`run_batch.py` — pipeline orchestrator, working except for MNova invocation.** Handles XML patching, config file writing, post-processing (txt → normalized npy).

**MNova Script Editor — confirmed working.** Opening `spinhanceBatch.qs` via Tools → Scripts → Edit Scripts and pressing the green play button successfully runs the batch function. The JS API calls are confirmed correct.

### What Does NOT Work — MNova CLI Invocation

We spent extensive time trying to invoke MNova from the command line. Here is what was learned:

| Attempt | Result |
|---|---|
| `--no-gui` flag | MNova 16 uses `--nogui` (one word) |
| `--py script.py` | `MnovaCore` only — bare Qt bindings, no NMR API |
| `--sf "functionName()"` from user scripts dir | "Not Found" — `--sf` doesn't find user scripts |
| `--sf "functionName()"` from app bundle scripts dir | "Not Found" — still not found |
| `--sf "/path/to/file.qs"` | Segfault — file paths not accepted |
| Script with top-level `spinhanceBatch()` call in user scripts dir | MNova crashes on startup |

**Root cause:** In MNova 16, `--sf` only calls functions from scripts that have been pre-registered via the GUI (File → Preferences → Scripts → Directories). Running `--sf` before doing that GUI step consistently fails.

**DO NOT put a script with top-level auto-executing code in `~/Library/Application Support/Mestrelab Research S.L./MestReNova/scripts/`** — it will crash MNova on startup. To recover: `rm ~/Library/Application\ Support/Mestrelab\ Research\ S.L./MestReNova/scripts/spinhanceBatch.qs`.

### Correct MNova JavaScript API (confirmed from Script Editor)
```javascript
// Open a spin-system XML (runs simulation synchronously)
serialization.open(xmlPath);

// Get the simulated spectrum
var rawSpec = Application.nmr.activeSpectrum();
var spec = new NMRSpectrum(rawSpec);
var nPoints = spec.count();          // number of spectral points
var intensity = spec.real(i);        // real intensity at point i

// List XML files in a directory
var files = dir.entryList("*.xml", Dir.Files);  // NOT setFilter/setNameFilters

// Close document without saving
mainWindow.activeDocument.close(false);

// Write to file
var f = new File(outPath);
f.open(File.WriteOnly | File.Text);
var stream = new TextStream(f);
stream.writeln(value);
f.close();

// Quit MNova
Application.mainWindow.close();
```

**WRONG API (do not use):**
- `Application.open()` — does not exist
- `nmr.activeSpectrum()` alone — use `Application.nmr.activeSpectrum()`
- `dir.setFilter()` / `dir.setNameFilters()` — not methods on Dir
- `Application.waitAllThreadsFinished()` — does not exist

### Next Steps for Task 3

**Option A (recommended if continuing with MNova):**
1. Open MNova GUI
2. File → Preferences → Scripts → Directories → add `~/Library/Application Support/Mestrelab Research S.L./MestReNova/scripts/`
3. Close MNova
4. Copy `simulation/spinhanceBatch.qs` to that directory (without auto-executing top-level call)
5. Test: `MestReNova --sf "spinhanceBatch()"`

**Option B (recommended for hackathon reliability — pure Python):**
Implement the spin simulation directly in Python using numpy/scipy. For N≤8 spin-½ systems, the Hilbert space is at most 2⁸ = 256 dimensional. Build the full spin Hamiltonian H = Σ δᵢ Izᵢ + Σ Jᵢⱼ (IᵢxIⱼx + IᵢyIⱼy + IᵢzIⱼz), diagonalize, compute transition frequencies and intensities, apply Lorentzian broadening, sum to spectrum. This bypasses MNova entirely and is fully scriptable.

---

## Environment Setup

```bash
# Install micromamba if not present (macOS)
brew install micromamba

# Create environment
cd /Users/labounader/Documents/Claude/Projects/spinhance
bash setup_env.sh
micromamba activate spinhance

# Run tests
python -m pytest simulation/test_xml_utils.py -v
```

**`environment.yml` key packages:** Python 3.14, rdkit, numpy, scipy, pandas, pytorch, scikit-learn, matplotlib, lxml, jupyterlab, pytest. Conda-forge channel only (no `defaults` — Anaconda SSL issues on this machine).

---

## Key Files to Read First

1. `README.md` — full project overview, data flow diagram, all subtasks
2. `predicted_mnova_1h (10).xml` — the mnova-spinsim XML format (reference)
3. `simulation/xml_utils.py` — fully working XML builder/patcher
4. `simulation/spinhanceBatch.qs` — the confirmed-correct MNova JS batch script
5. `simulation/run_batch.py` — the Python orchestrator (needs MNova invocation fixed)

---

## Notes for Collaborators

- All 4 tasks are designed to be independent modules with clean interfaces (CSV → npy → npy → model)
- The shift+J matrix is the shared data contract between Tasks 2 and 3
- Task 3 needs Task 2's output (XML files or matrices) to proceed at scale; for testing, use `predicted_mnova_1h (10).xml` as a stand-in
- Degeneracy is stored in the XML `number=` attribute and in the 9th column of the matrix
- Field strength is baked into the XML `<frequency>` tag; `xml_utils.patch_frequency()` generates multi-field variants
- Target spectral grid: 2¹⁴ = 16384 points, 0–12 ppm, normalized so ∫ intensity dppm = 1
