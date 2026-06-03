# Pipeline Debug & Re-train Roadmap

_Started 2026-06-02. Goal: fully debug generation → simulation → data → training,
regenerate clean datasets, and re-run every inverse model **beyond the CNN
baseline** on clean data + clean augmentation._

## Bugs

### Fixed (merged)
| # | Bug | Where | PR |
|---|-----|-------|----|
| 1 | Symmetric/enantiotopic **rotor over-count** — remote symmetric *and* enantiotopic methyl rotors (incl. isopropyl / gem-dimethyl) split into per-proton groups instead of one HARD methyl each | `generate/spin_equivalence._classify_equivalence_classes` | #71 |
| 2 | Spurious **methyl geminal ²J** (−14.3 Hz between a methyl's protons) | `mol_to_spin_system.geminal.geminal_couplings` | #71 |
| 3 | **Deuterium counted as ¹H** — ²H/³H counted as groups; D→D substitution is a no-op so all D also collapsed into one bogus equivalence class (duplicated shifts) | `generate/spin_equivalence`, `mol_to_spin_system.groups.proton_groups` | #72 |
| 4 | **Baseline-drift augmentation** — non-physical for processed (baseline-corrected) data | `model/data/transforms.augment_spectrum` | #73 |
| 5 | **Global referencing-shift augmentation** — slid the spectrum ±0.01 ppm while leaving labels fixed → ~0.01 ppm pure label noise on already-referenced spectra | `model/data/transforms.augment_spectrum` | #73 |

### Identified, not yet fixed
| # | Bug | Where | Plan |
|---|-----|-------|------|
| 6 | **Noise scaled to `spec.max()`** instead of a 1H-singlet reference. Punishes large-singlet molecules (a 9H tBu makes its minor peaks see ~9× too much noise) — likely cause of poor tBu performance | `model/data/transforms.augment_spectrum` | Approach A: `noise_sigma = frac × H_unit / N`, `N = Σ degeneracy`. Training-time only, no re-sim |
| 7 | **Per-group shift augmentation never wired in** — `randomized_shifts` + stored `shift_range` are unused; spectra & labels both use raw predictor means (two inequivalent tBu → identical 1.38) | `mol_to_spin_system.augment` (unused), `simulation.graph_io.record_to_arrays` | Decide: bake sampled shifts in at generation (needs re-sim) or leave |

### To audit (not yet investigated)
- **Coupling modules** beyond geminal: vicinal/Karplus, aromatic ortho/meta/para, olefinic, benzylic/long-range — verify each is applied to the correct proton pairs, group-mapped correctly, and that intra-group couplings are dropped (geminal had a bug; the others are unaudited).
- **Grouping edge cases:** vinyl pairs, AA′BB′ aromatics, diastereotopic CH₂ under 3-D embedding failure (`has_3d=False` fallback), strained/large rings.
- **Shift predictor:** fallback/default behavior, solvent, resolution limits (e.g. two inequivalent tert-butyls predicted identical = predictor resolution, not a code bug — but a modeling limitation worth noting).
- **Degeneracy normalization** (known d=2 information-limited problem) and its interaction with the ∫=1 normalization.
- **Standardization / splits / `encode_target`** sanity.

## Roadmap
1. **Finish the audit** (couplings, grouping edge cases, predictor) — find remaining bugs *before* committing compute.
2. **Implement fixes:** noise scaling (Approach A), shift-augmentation decision, any newly-found bugs. Each with regression tests.
3. **Full dataset rebuild** (not targeted removal — we're retraining everything anyway): re-screen ChEMBL + PubChem with fixed grouping (drops false positives **and** recovers wrongly-excluded molecules), rebuild spin systems (predictor + fixed couplings), optionally bake in shift augmentation, re-simulate spectra, rebundle 500k / 1M / 64k. Requires the nmrshiftdb predictor on HPC (`java` present; locate/confirm `predictorh.jar`).
4. **Re-train all inverse models beyond the CNN baseline** (025 / 026 / 027 / 028 / 029 / 030 + xl + 1M variants) on clean data + clean augmentation. Cancel the currently-running contaminated jobs first.
5. **Validate** vs held-out test molecules; refresh the website.

## Open decisions
- **Shift augmentation (#7):** bake sampled per-group shifts in at generation, or leave labels = predictor means?
- **Predictor availability:** confirm `predictorh.jar` / `NMRSHIFTDB_HOME` on HPC for the full rebuild.
- **Running jobs:** cancel now (we will re-run them) vs let finish. Earlier choice "let finish" predates the decision to re-run everything.
- The earlier validated **targeted-removal** regen (6,580/500k) is now superseded by the full-rebuild plan unless you prefer the cheaper path.
