# SpinHance — project site (`docs/`)

A static, single-page site for the SpinHance hackathon project, served via **GitHub Pages from `/docs`**.

The hero is a **scroll-driven "field sweep"**: a real ¹H spin system is simulated across a
geometric sweep of spectrometer fields (90 → 600 MHz) and the bold trace morphs from
overlapping, second-order multiplets to crisp, first-order peaks as you scroll. A different
molecule is chosen on every reload. Dark / light theme toggle in the nav (persisted).

## Files

| path | what |
|---|---|
| `index.html` | the whole site (inline CSS) |
| `assets/sweep.js` | canvas animation + scroll/theme logic + data decode |
| `data/field_sweep.json` | precomputed spectra (base64 uint16, per-molecule normalized) |
| `build_field_sweep.py` | regenerates `data/field_sweep.json` from the pyspin simulator |
| `.nojekyll` | tells Pages to serve files as-is (no Jekyll) |

## Rebuild the spectra dataset

Run from the repo root (uses `simulation/pyspin` + `mol_to_matrix/data/spin_systems.json`):

```bash
python docs/build_field_sweep.py
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
