# SpinHance — Project Context for Claude

## What This Project Is

SpinHance is a hackathon project to automatically extract ¹H chemical shifts and scalar coupling constants from **low-field (90 MHz) ¹H NMR spectra** of small molecules using deep learning. The key insight: at low field, spin systems are strongly coupled (non-first-order), making simple peak-picking fail. We train a neural network to invert the spectrum back to the underlying spin-system parameters.

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
│   ├── README.md            # human-facing docs + architecture diagram
│   ├── CLAUDE.md            # this file (AI-facing contract)
│   ├── __init__.py          # public API re-exports
│   ├── xml_io.py            # build/patch mnova-spinsim XML (pure)
│   ├── mnova_runner.py      # MestReNova CLI invocation
│   ├── pipeline.py          # patch → simulate → convert orchestration
│   ├── plotting.py          # QC plot of 90 vs 600 MHz spectra
│   ├── cli.py               # `python -m simulation.cli run|plot`
│   ├── mnova_scripts/
│   │   └── spinhanceBatch.qs   # MNova JS batch script (register this folder)
│   └── tests/
│       └── test_xml_io.py      # pytest suite (no MNova required)
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
Take the shift+J matrix, convert to MNova XML format, run MNova's quantum spin simulator at **90 MHz** (low-field, non-first-order) and **600 MHz** (high-field, reference). Output: 2¹⁴-point normalized intensity arrays as `.npy` files. See detailed status below.

### Task 4 — ML MODEL (`ml_model/`)
Train a neural network: input = 16384-point normalized spectrum (90 MHz), output = 8×9 shift+J+degeneracy matrix. Key challenge: permutation invariance of spin-group labels (Hungarian matching loss).

---

## Task 3 — Simulation: Status (WORKING)

The full pipeline runs end to end: matrix/XML → MNova simulation at 90 & 600 MHz
→ normalised `.npy`. Confirmed 2026-05-29 on the reference molecule (16384-point
spectra, integral = 1).

### Environment
- **MNova version:** MestReNova v16.0.0-39276 (macOS, x86_64)
- **Python env:** micromamba `spinhance`, Python 3.14, conda-forge

### Public API (import contract)
```python
from simulation import (
    matrix_to_xml, save_xml, patch_frequency, generate_field_pair,  # xml_io
    prepare_xmls, txt_to_npy, run_pipeline,                          # pipeline
    run_mnova_batch, MNOVA_DEFAULT,                                  # mnova_runner
    LOW_FIELD_MHZ, HIGH_FIELD_MHZ, DEFAULT_FIELDS_MHZ, N_POINTS,
)

tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=90.0)
save_xml(tree, "output.xml")
lo, hi = generate_field_pair("source.xml", "out_dir/", stem="mol_001")
run_pipeline(Path("xmls_source"), Path("data/processed"))  # patch→sim→npy
```

### Module responsibilities
- `xml_io.py` — pure matrix ⇄ XML; no MNova or numpy dependency.
- `mnova_runner.py` — the ONLY module that invokes MestReNova.
- `pipeline.py` — orchestration + numpy post-processing (`txt_to_npy`).
- `plotting.py` — QC overlays (`plot_field_comparison`).
- `cli.py` — `python -m simulation.cli run|plot`.

### MNova CLI invocation — the rules (solved; do not regress)
The CLI works **without** `-nogui`:
```bash
MestReNova -sf spinhanceBatch,<xmlDir>,<outDir>   # exit 0, writes <stem>.txt per XML
```
- **Single dash, function NAME, no parens, comma-separated args.** `--sf "fn()"` fails "Not Found".
- The script's folder (`simulation/mnova_scripts`) must be registered in
  **Edit → Preferences → Scripting → Directories**, then MNova restarted. `-sf`
  only resolves names from registered dirs. File name must equal function name.
- **Never pass `-nogui`:** under it `Application.quit()` cannot terminate (no GUI
  event loop) and MNova hangs forever despite writing output. With the window
  visible it runs the whole batch in one launch and exits cleanly.
- The `.qs` must have **no top-level auto-executing call** (crashes MNova at
  startup). `spinhanceBatch(xmlDir, outDir, argNoQuit)` — pass a truthy 3rd arg
  to skip the quit for interactive Script-Editor testing.
- The Script Editor caches the loaded file; after editing on disk, reload it.

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

// Quit MNova (only in CLI runs; guarded in the .qs)
Application.quit();                  // global mainWindow.close() is the fallback
```

**WRONG API (do not use):**
- `Application.open()` — does not exist
- `Application.mainWindow.close()` — `Application.mainWindow` is undefined; use `Application.quit()` or the global `mainWindow`
- `nmr.activeSpectrum()` alone — use `Application.nmr.activeSpectrum()`
- `dir.setFilter()` / `dir.setNameFilters()` — not methods on Dir
- `Application.waitAllThreadsFinished()` — does not exist

### Parallelism (single-instance gotcha)
MNova's batch loop is single-threaded (one core). `pipeline.run_pipeline(...,
workers=N)` and `mnova_runner.run_mnova_parallel(...)` round-robin shard the XMLs
across N concurrent instances. MestReNova is **single-instance**: a plain second
launch hands off to the running process, so parallel workers launch with
`open -na MestReNova --args -sf …` (`launcher="open"`, default) to force separate
processes; completion is detected by polling output `.txt` counts. Fallback
`launcher="direct"` runs the binary directly (only if concurrent direct launches
don't hand off). Measured single-process cost: ~0.68 s/sim (8-spin), startup
~4.5 s cold. Benchmark with `--workers` to find the core sweet spot.

### pyspin — pure-Python engine (`engine="python"`)
Validated alternative to MNova (r=0.9993 vs MNova on R-5-MCH). License-free,
parallel across cores, HPC-capable.
- `pyspin/simulator.py` — spin-½ reference sim (Iz-block) + shared
  `lorentzian_broaden` (FFT stick-convolution; O(points·log points)).
- `pyspin/composite.py` — composite-particle reduction: each equivalent group
  treated by its total spin (`spin_reps(d)` → (S, multiplicity)); spectrum is
  the multiplicity-weighted sum over per-group total-spin combinations. Vectorised
  Hamiltonian (einsum diagonal + searchsorted flip-flop), scipy BLAS eigh when
  present. ~0.01 s for 10-proton, ~0.8 s for a dense 16-proton (was 23 s naive).
- `pyspin/batch.py` — `multiprocessing` (spawn) across molecules, BLAS pinned to
  1 thread/worker (else oversubscription). `run_pipeline(engine="python", workers=N)`.
Perf wins were: FFT broadening (the real bottleneck, not eigh) and composite
reduction (makes high-degeneracy groups tractable).

### Engine scaling & the `auto` router
Benchmarked (fully-coupled chain, worst case): pyspin is exact so cost ~ C(N,N/2)³
— N=13 ~1.6s, N=14 ~12s, N=15 ~85s, N≥16 >2min. MNova is FLAT ~2.4-2.9s up to
N=20 (2^20): it uses overlapping local spin-cluster decomposition with first-order
gluing (near-exact on sparse graphs, linear scaling). pyspin vs MNova still agree
at N=14 (r=0.9947, 40/40 peaks), so MNova's approximation is excellent.
Implication: pyspin wins for sparse/decomposable molecules (the norm); MNova wins
for one large coupled fragment. `engine="auto"` (`_run_auto` in pipeline.py) routes
per molecule by `largest_component_spins` (≤ pyspin_max_spins=13 → pyspin, else
MNova) and prints the routing distribution. Metric is conservative (total
component spins ≥ post-reduction cost). Future: implement local-cluster
approximation in pyspin for full HPC scaling without MNova.

### Possible future work
- **Scale:** MNova engine opens every XML in one session; for pyspin, scale by
  adding workers / HPC nodes (embarrassingly parallel across molecules).
- **Pure-Python fallback (if MNova ever blocks):** for N≤8 spin-½ systems the
  Hilbert space is ≤ 2⁸ = 256-d. Build H = Σ δᵢ Izᵢ + Σ Jᵢⱼ (IᵢxIⱼx + IᵢyIⱼy +
  IᵢzIⱼz), diagonalise, compute transitions + intensities, Lorentzian-broaden.
  Fully scriptable, bypasses MNova.

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
python -m pytest simulation/tests -v
```

**`environment.yml` key packages:** Python 3.14, rdkit, numpy, scipy, pandas, pytorch, scikit-learn, matplotlib, lxml, jupyterlab, pytest. Conda-forge channel only (no `defaults` — Anaconda SSL issues on this machine).

---

## Key Files to Read First

1. `simulation/README.md` — architecture diagram, usage, MNova setup
2. `predicted_mnova_1h (10).xml` — the mnova-spinsim XML format (reference)
3. `simulation/xml_io.py` — XML builder/patcher (pure)
4. `simulation/pipeline.py` — patch → simulate → convert orchestration
5. `simulation/mnova_scripts/spinhanceBatch.qs` — the MNova JS batch script

---

## Notes for Collaborators

- All 4 tasks are designed to be independent modules with clean interfaces (CSV → npy → npy → model)
- The shift+J matrix is the shared data contract between Tasks 2 and 3
- Task 3 needs Task 2's output (XML files or matrices) to proceed at scale; for testing, use `predicted_mnova_1h (10).xml` as a stand-in
- Degeneracy is stored in the XML `number=` attribute and in the 9th column of the matrix
- Field strength is baked into the XML `<frequency>` tag; `xml_io.patch_frequency()` generates multi-field variants
- Target spectral grid: 2¹⁴ = 16384 points, 0–12 ppm, normalized so ∫ intensity dppm = 1
