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
| 6 | **Noise scaled to `spec.max()`** instead of a 1H-singlet reference — punished large-singlet molecules (a 9H tBu made its minor peaks see ~9× too much noise). Now `noise = frac × (1H height = peak/sum × Σspec / N)`, `frac` log-uniform 0.3–1.5% per spectrum | `model/data/transforms.augment_spectrum` (+ `dataset` threads `n_protons`) | #75 |

### Identified, not yet fixed
| # | Bug | Where | Plan |
|---|-----|-------|------|
| 7 | **Variability never applied — severe value reuse (CONFIRMED, quantified).** `augment.sample_record`/`bake_file` jitters shifts (`N(mean, σ_from_range)`, floor 0.05 ppm) AND couplings (per-type σ), class-aware — but was never run on the dataset. 500k bundle: shifts snap to ~1000 values (top-20 = 21%, e.g. 7.57 ppm ×21,550); couplings = **~10 constants** (7.5 Hz = 19%). The matrices are not a faithful sample of chemical space. | `mol_to_spin_system.augment` (built, unused); generation never bakes it | **Phase 3 rebuild: sample once per molecule (class-aware), then simulate the spectrum from the SAMPLED values** so spectrum+label stay consistent. Consider K realizations for more coverage. Tune σ floor/cap/k + per-type J σ. |

| 8 | **Heteroaromatic couplings use benzene values** — `aromatic_couplings` applies ortho/meta/para = 7.5/1.5/0.7 to *every* aromatic ring. Real values differ a lot (furan ³J₂₃ ≈ 1.8 vs 7.5; pyridine/thiophene ³J₂₃ ≈ 4.9). Systematic error on the many heteroaromatic drug molecules. | `mol_to_spin_system.aromatic` | Tier-1 Pretsch extraction: ring-type-aware J lookup from the heteroaromatic coupling tables |

### Decisions locked
- **#7 variability = ONE realization per molecule, baked as ground truth.** It is *not* instrument/physical variability — purely to sample chemical space — so it's drawn once and never resampled per epoch (detector-noise augmentation stays separate). Sample shifts+couplings (class-aware) → simulate the spectrum from the sampled values → that matrix IS the molecule's spin system.
- **Deuterium molecules eliminated entirely** — the rebuild screen rejects any molecule containing explicit ²H/³H (stronger than fix #3, which only excluded D from counting).
- **Pretsch enrichment:** surgical, prioritized. Tier 1 (heteroaromatic couplings, #8) folded into THIS rebuild. Tier 2 (substituent-dependent olefinic/vicinal) optional same-pass. Tier 3 (shift additivity Zα/Zβ) deferred — HOSE predictor already empirical. Extract specific table pages via image rendering + validated transcription (text extraction mangles the tables); not a full-book campaign.

### Phase 1 audit — findings (complete)
- **Grouping: sound.** styrene (vinyl NONE + ortho/meta SOFT pairs + para NONE = 8), p-/o-xylene (HARD methyls + AA′BB′), diastereotopic CH₂ (split, 3-D resolved), cyclopropane/acetylacetone all classify correctly after fixes #1–#3.
- **Couplings: physically correct mechanisms, correctly targeted** (geminal CH₂-only after #2; vicinal Karplus-on-ring vs rotatable-empirical; aromatic ortho/meta/para; olefinic cis/trans; allylic/benzylic 4J). All **deterministic discrete values** → feed the reuse problem (#7).
- **Minor smells (low priority):**
  - Coupling modules iterate over all H incl. deuterium (harmless — D-couplings dropped downstream — but inconsistent with the D fix; should skip isotope≠protium for cleanliness).
  - `aromatic_couplings` uses shortest-path separation → fused-ring peri/cross-ring pairs get no coupling (naphthalene peri ⁴J missed).
  - `sample_record` class-aware grouping keys on `(mean, range)` as an equivalence proxy, not the true tier class (works in practice; fragile to accidental ties).
  - 3-D embed failure (`has_3d=False`) → 2-D fallback can under-count diastereotopic CH₂ (documented limitation, rare).
- **Still worth a look:** degeneracy normalization vs ∫=1 (known d=2 issue); standardization/splits/`encode_target` sanity.

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
