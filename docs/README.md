# Spinhance — project site (`docs/`)

A static, single-page site for the Spinhance hackathon project, served via **GitHub Pages from `/docs`**.

The hero is a **scroll-driven "field sweep"**: a real ¹H spin system is simulated across a
geometric sweep of spectrometer fields (90 → 600 MHz) and the bold trace morphs from
overlapping, second-order multiplets to crisp, first-order peaks as you scroll. A different
molecule is chosen on every reload. Dark / light theme toggle in the nav (persisted).

## Files

| path | what |
|---|---|
| `index.html` | the landing page + hero "field sweep" (inline CSS) |
| `models.html` | **Explore the Models** — the recipe explainer (025–030) + the held-out results cards and the interactive test-molecule explorer |
| `dataset.html`, `citations.html`, `slides.html` | dataset viewer, references, presentation |
| `assets/sweep.js` | hero canvas animation + scroll/theme logic + data decode |
| `assets/viewers.js` | the results viewers — learning curves, the held-out comparison/grid, and the **test-molecule explorer** (`RECIPE_DESC` recipe labels live here) |
| `data/field_sweep.json` | precomputed **stick** spectra per molecule/field (base64 float32 centers + uint16 amps); broadened into smooth Lorentzians client-side |
| `data/test_eval.json` | per-model held-out **test + val** metrics (the results-card source of truth) |
| `data/learning_curves.json` | per-model training curves (by epoch) |
| `data/test_explorer.json` | the explorer payload: per-model predictions + rendered spectra + the refinement overlay, on the shared held-out molecules (**adaptive RDP mesh, ~4 MB**, down from ~18 MB) |
| `data/spin_systems_pubchem.json` | the hero pool — a random 1000-molecule subset of the PubChem set that `build_field_sweep.py` scores and samples from |
| `sample_pubchem_subset.py` | reservoir-samples the 1000-molecule pool from `mol_to_spin_system/data/spin_systems_pubchem.json.tar.gz` (~2.3M molecules) |
| `build_field_sweep.py` | regenerates `data/field_sweep.json` from the pyspin simulator |
| `export_test_explorer.py` | regenerates `data/test_explorer.json` — runs every finished model on the shared held-out molecules and (on the HPC, where the checkpoints + parts live) computes the refinement overlay |
| `.nojekyll` | tells Pages to serve files as-is (no Jekyll) |

## Explore the Models (`models.html`)

A grounded walkthrough of the shared `spingraph_decoder` backbone and the **025–030 recipe
ladder** (peak channel, soft-equivalence, focal loss, cumulative-integral; see `RECIPE_DESC` in
`assets/viewers.js`), with the held-out results cards (from `data/test_eval.json`) and an
interactive **test-molecule explorer** over the shared leakage-controlled held-out split.

Recent explorer features (`assets/viewers.js` + `export_test_explorer.py`):
- **Adaptive peak mesh** — each spectrum ships its own Ramer–Douglas–Peucker-decimated x-mesh, so
  `test_explorer.json` is ~4 MB (was ~18 MB) with no visible fidelity loss.
- **All six finished 500k models** (recipes 025–030) plus the 64k fleet and the CNN baseline.
- **Per-trace toggles** (target / prediction / refined) + **line transparency** so overlapping
  traces stay readable, on a taller plot.
- **Refinement overlay** — a violet "refined" trace (test-time analysis-by-synthesis,
  `model/inference/refine.py`) with the per-molecule **raw→refined shift-MAE**; molecules skipped
  by the refiner's cost guard (dense systems) are not overlaid.

## Rebuild the spectra dataset

Run from the repo root. To redraw the 1000-molecule pool from the full PubChem set:

```bash
python docs/sample_pubchem_subset.py     # -> docs/data/spin_systems_pubchem.json
```

Then regenerate the hero spectra (uses `simulation/pyspin`, the pool above, and the
3D structures in `generate/data/pubchem_8spin.xyz.gz`):

```bash
python docs/build_field_sweep.py          # -> docs/data/field_sweep.json
```

Tunables live at the top of the script: number of molecules, number of geometric field
frames, display resolution, linewidth, and the molecule-selection score (which favours
distinct-but-close coupled shifts — the most visually dramatic low→high sweeps).

## Enable GitHub Pages

Repo **Settings → Pages → Build and deployment → Source: Deploy from a branch**, then choose
branch `main` and folder `/docs`. The site publishes at:

```
https://labounad.github.io/spinhance/
```
