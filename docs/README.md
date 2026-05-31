# Spinhance — project site (`docs/`)

A static, single-page site for the Spinhance hackathon project, served via **GitHub Pages from `/docs`**.

The hero is a **scroll-driven "field sweep"**: a real ¹H spin system is simulated across a
geometric sweep of spectrometer fields (90 → 600 MHz) and the bold trace morphs from
overlapping, second-order multiplets to crisp, first-order peaks as you scroll. A different
molecule is chosen on every reload. Dark / light theme toggle in the nav (persisted).

## Files

| path | what |
|---|---|
| `index.html` | the whole site (inline CSS) |
| `assets/sweep.js` | canvas animation + scroll/theme logic + data decode |
| `data/field_sweep.json` | precomputed **stick** spectra per molecule/field (base64 float32 centers + uint16 amps); broadened into smooth Lorentzians client-side |
| `data/spin_systems_pubchem.json` | the hero pool — a random 1000-molecule subset of the PubChem set that `build_field_sweep.py` scores and samples from |
| `sample_pubchem_subset.py` | reservoir-samples the 1000-molecule pool from `mol_to_spin_system/data/spin_systems_pubchem.json.tar.gz` (~2.3M molecules) |
| `build_field_sweep.py` | regenerates `data/field_sweep.json` from the pyspin simulator |
| `.nojekyll` | tells Pages to serve files as-is (no Jekyll) |

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
