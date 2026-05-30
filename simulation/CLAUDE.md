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
├── simulation/          # Task 3 — spin simulation (LUCAS); two engines
│   ├── README.md            # human-facing docs + architecture diagram
│   ├── CLAUDE.md            # this file (AI-facing contract)
│   ├── __init__.py          # public API re-exports
│   ├── graph_io.py          # Task 2 contract: spin-graph ⇄ arrays/XML + JSONL I/O
│   ├── xml_io.py            # matrix ⇄ mnova-spinsim XML (matrix_to_xml/xml_to_matrix)
│   ├── mnova_runner.py      # MestReNova CLI (run_mnova_batch/run_mnova_parallel)
│   ├── pipeline.py          # orchestration: run_pipeline (engine mnova/python/auto)
│   ├── plotting.py          # QC plot of 90 vs 600 MHz spectra
│   ├── cli.py               # `python -m simulation.cli run|plot`
│   ├── mnova_scripts/spinhanceBatch.qs   # MNova JS batch script (register folder)
│   ├── pyspin/              # pure-Python engine (engine="python")
│   │   ├── simulator.py        # spin-½ reference + shared FFT broadening
│   │   ├── composite.py        # composite reduction + component split (exact)
│   │   ├── cluster.py          # local-cluster approx + wall-free dispatcher
│   │   ├── batch.py            # multiprocessing XML→npy driver
│   │   └── validate_vs_mnova.py
│   ├── benchmarks/          # benchmark_fields / _pyspin / _scaling
│   ├── examples/            # sample spin systems (incl. reference_15group.xml = format ref)
│   └── tests/               # 44 tests (xml_io, graph_io, mnova_runner, composite, cluster, fields)
├── model/            # Task 4 — deep learning model (teammate)
├── data/
│   ├── raw/             # SMILES lists (gitignored if large)
│   └── processed/       # XMLs, spectra
├── environment.yml      # micromamba env, Python 3.14, conda-forge only
├── setup_env.sh         # creates/updates the micromamba environment
└── README.md            # full project documentation
```

---

## Spin-System Representation

Each molecule → **8×9 block**:
- **Diagonal (8×8):** chemical shifts δ in ppm (field-independent)
- **Off-diagonal (8×8):** scalar couplings *J* in Hz (field-independent, symmetric)
- **Extra column (8×1):** degeneracy (number of protons per spin group; e.g. 3 for CH₃)

This is an undirected labeled graph: nodes carry (δ, n), edges carry *J*. Labels are arbitrary (invariant under S₈ permutation). We restrict to molecules with **exactly 8 magnetically distinct spin groups**.

The XML format (`simulation/examples/reference_15group.xml`) encodes this as `<mnova-spinsim>` with `<group>` elements containing `<shift>`, `<jCoupling name="X">`, and a `number=` attribute for degeneracy.

---

## The Four Tasks

### Task 1 — GENERATE (`generate/`)
Screen SMILES from USPTO/PubChem, filter to exactly 8 hard-equivalent (chemically + magnetically equivalent) spin groups using RDKit. Output: `data/raw/smiles_8group.csv`.

### Task 2 — MOL → MATRIX (`mol_to_matrix/`)
3D embed with ETKDG + MMFF94 (RDKit), assign shifts via heuristic tables, compute *J* couplings via Karplus equations and geminal/aryl/vinyl/benzylic tables. Assemble the 8×8 J-matrix + degeneracy vector. Output: `data/processed/matrices/*.npy`.

### Task 3 — SIMULATION (`simulation/`) ← LUCAS'S TASK
Take the shift+J matrix, convert to MNova XML format, run MNova's quantum spin simulator at **90 MHz** (low-field, non-first-order) and **600 MHz** (high-field, reference). Output: 2¹⁴-point normalized intensity arrays as `.npy` files. See detailed status below.

### Task 4 — ML MODEL (`model/`)
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
from simulation.xml_io import xml_to_matrix          # XML → {shifts,couplings,degeneracy,meta}
from simulation.pyspin.composite import simulate_spectrum_composite, largest_component_spins

# build / parse XML
tree = matrix_to_xml(shifts, couplings, degeneracy, frequency_mhz=90.0); save_xml(tree, "m.xml")
m = xml_to_matrix("m.xml")

# pure-Python sim (exact, composite reduction)
ppm, spec = simulate_spectrum_composite(shifts, couplings, degeneracy, 90.0)

# full pipeline; engine = "mnova" | "python" | "auto"
run_pipeline(Path("xmls_source"), Path("out"), engine="python", workers=8)
```

### Task 2 → Task 3 contract (`graph_io.py`) — FINALIZED
Task 2 emits a single JSON ARRAY (`mol_to_matrix/data/spin_systems.json`); each
element: `{"chembl_id","smiles","inchikey", "labels":["A",...], "spin_groups":[[shift_ppm,n],...] (aligned to labels), "couplings":[["A","B",J_Hz],...]}`.
Absent couplings ⇒ J=0; sign retained (geminal negative). Keys are constants in
`graph_io.py` (KEY_LABELS/KEY_GROUPS/KEY_COUPLINGS/ID_KEYS). `record_to_arrays`
→ (labels, shifts, couplings, degeneracy) using the record's OWN label order;
`record_to_xml` for MNova; `read_spin_systems` (JSON array, tolerates JSONL);
`spin_systems_to_xml_dir` materialises XMLs. CLI: `run --graphs spin_systems.json`
(engine=python consumes records directly via `run_pyspin_batch_graphs`;
mnova/auto materialise XMLs first). Outputs `spectra/<field>MHz/mol_<idx>.npy` +
`spectra/index.csv` (idx→chembl_id). Verified end-to-end on the 5-molecule
sample (10/10 sims).

### Module responsibilities
- `graph_io.py` — Task 2 spin-graph contract: graph ⇄ arrays/XML, JSONL I/O, validation.
- `xml_io.py` — pure matrix ⇄ XML (build, parse, patch-frequency, field-pair); no MNova/numpy.
- `mnova_runner.py` — the ONLY module that invokes MestReNova (batch + parallel + retry).
- `pipeline.py` — orchestration + routing; `run_pipeline(engine=…)`, `_run_auto`, `txt_to_npy`.
- `pyspin/` — pure-Python engine: `composite.py` (production), `simulator.py` (reference + broadening), `batch.py` (multiprocessing).
- `plotting.py` — QC overlays (`plot_field_comparison`).
- `spectrum_io.py` — three representations + unified `load_spectrum(path)→dense`: **dense** (`mol_<i>.npy`), **sparse** (`.npz` idx/val, drop ≤cutoff·max, renorm ∫=1), **peaks** (`.npz` centers/amps + linewidth/field; lineshape convolved on load). Broadening split in `simulator.py` into `build_stick`+`lorentzian_convolve`+`peaks_to_spectrum`; engines expose `composite_transitions`/`clustered_transitions`/`transitions_pyspin` (line lists). `run --format peaks` (graphs+python) stores peaks; `export --sparsify` does dense→sparse.
- `export.py` — pack a `spectra/` dir into one `.tar.gz` (two tqdm bars: compress, zip); passes `.npz` through, sparsifies dense `.npy` if asked.
- `cli.py` — `python -m simulation.cli run|plot|export` (`--engine`, `--workers`, `--launcher`, `--pyspin-max-spins`; export: `--spectra_dir --out --no-sparsify --cutoff --no-renormalize`).

### CLI flags (run)
`--xml_dir --out_dir --mnova --fields 90 600 --workers N --launcher {open,direct} --engine {mnova,python,auto} --pyspin-max-spins 13`

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
Implication: pyspin's EXACT engine wins for sparse/decomposable molecules; one
large coupled fragment used to need MNova. `engine="auto"` (`_run_auto`) still
routes per molecule by `largest_component_spins` (≤ pyspin_max_spins=13 → pyspin,
else MNova) and prints the distribution.

### pyspin.cluster — local-cluster approximation (wall removed)
`pyspin/cluster.py` implements MNova's trick: `partition_clusters` cuts the
weakest bonds (Kruskal-style, size-capped union-find) so each cluster ≤
max_cluster spins; each cluster is simulated EXACTLY (composite) while cut bonds
are treated first-order — the far-side spins act as a classical Iz bath that
shifts the cluster's resonances by J·m per bath total-Iz value m (binomial
multiplicity), reproducing ordinary multiplet splitting. Reduces to exact when no
bond is cut. Validated vs exact: corr 1.0 (no cut), 0.992 (N=14 chain), and runs
a 100-spin chain in ~0.2 s (linear). `simulate_spectrum_pyspin(... exact_max_spins
=12, max_cluster=9)` dispatches exact-vs-clustered by largest component; `batch.py`
uses it, so the `python` engine is now WALL-FREE (no molecule can stall a batch).
It IS an approximation (first-order between clusters) — excellent on sparse
graphs, like MNova.

### Possible future work
- **Cluster-approximation accuracy tuning:** current cut is first-order (Iz·Iz
  only). Could keep next-nearest strong bonds, use overlapping clusters, or raise
  max_cluster for borderline cases. Validate against MNova on real large-fragment
  molecules and pick max_cluster from the accuracy/speed trade-off.
- **Task 2 adapter:** if Task 2 emits matrices as `.npy` rather than XML, add a
  small `matrix.npy → matrix_to_xml → save` shim ahead of `run_pipeline`.

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
2. `simulation/examples/reference_15group.xml` — the mnova-spinsim XML format (reference)
3. `simulation/xml_io.py` — XML builder/patcher (pure)
4. `simulation/pipeline.py` — patch → simulate → convert orchestration
5. `simulation/mnova_scripts/spinhanceBatch.qs` — the MNova JS batch script

---

## Notes for Collaborators

- All 4 tasks are designed to be independent modules with clean interfaces (CSV → npy → npy → model)
- The shift+J matrix is the shared data contract between Tasks 2 and 3
- Task 3 needs Task 2's output (`spin_systems.json`) to proceed at scale; for testing, use the molecules in `simulation/examples/` as stand-ins
- Degeneracy is stored in the XML `number=` attribute and in the 9th column of the matrix
- Field strength is baked into the XML `<frequency>` tag; `xml_io.patch_frequency()` generates multi-field variants
- Target spectral grid: 2¹⁴ = 16384 points, 0–12 ppm, normalized so ∫ intensity dppm = 1
