# Pipeline Audit 2 — Spin-System Generation Correctness

**Status:** LANDED (opened 2026-06-03; core complete same day). Single source of
truth for the second audit + generation-code fix/refactor + full dataset regeneration
+ retrain of all PubChem models + website propagation. Audit + fix + regeneration +
leakage-split are **done**; the 64k·026 model finished and its held-out eval is live on
the site; the rest of the fleet (500k/3M + 64k ablations) is training. Update as it lands.

---

## 0. Why this audit (the miss)

The first audit (PIPELINE_DEBUG_ROADMAP.md) missed a **class of correctness bug**:
the per-molecule variability bake decides *which groups must share a value* using a
**`(base_shift, range)` proxy** instead of the **true HARD/SOFT equivalence tier**
from the spin-equivalence classifier.

**Consequence:** two chemically *distinct* proton groups that merely share a
coincidentally-equal Pretsch base shift get **locked to one identical jittered
value** instead of each receiving independent Gaussian noise.

**Confirmed example** — `CC1=C(C=CC(=C1)F)OCC(=O)C2=CC=C(S2)Cl` (mol_3078877),
true shifts in `records_3M_test`:
`[7.159, 7.159, 6.814, 6.568, 6.568, 5.162, 5.162, 2.362]` — **three exact-duplicate
pairs**. Only the OCH₂ (5.162) is a legitimate diastereotopic/soft-equiv duplicate;
the **aromatic pairs (7.159, 6.568) are artifacts** — distinct ring positions Pretsch
can't resolve, then frozen exactly equal by the share-by-`(m,range)` logic.

**Root cause:** `mol_to_spin_system/augment.py::sample_record` (~line 148):
```python
classes[(m, tuple(r))].append(i)        # WRONG: equality of (shift, range) is not equivalence
```
Equivalence (who must share a drawn value = **SOFT-equivalent siblings only**, e.g.
genuine AA′BB′) must come from `generate/spin_equivalence.py`. Everyone else —
including coincidentally-equal base shifts — must be sampled **independently**.

The `mol_to_spin_system` package was written under hackathon time pressure. This
audit is a comprehensive correctness review and, where warranted, a refactor.

---

## 1. Audit scope (3 delegated workstreams)

- **A — Equivalence & augmentation (core).** `generate/spin_equivalence.py`
  (HARD/SOFT/NONE classification: deuterium-substitution test, rotor handling,
  symmetry) + `mol_to_spin_system/augment.py` (the `(m,range)` keying bug). Map how
  SOFT-sibling info flows (or fails to) into the record + bake; define the correct
  share key and the record-schema change to carry it.
- **B — Shifts & couplings.** `shifts_pretsch.py` (additivity correctness, base-shift
  collisions, the `shift_range` metadata) + all coupling modules (geminal, vicinal,
  aromatic, olefinic, long_range, heteroaromatic, coupling.py). 3D-conformer
  dependence, diastereotopic handling, sign/constant correctness vs Pretsch.
- **C — Grouping, matrix, driver & consistency.** `groups.py` (proton_groups),
  `matrix.py` (build_spin_system), `xyz.py` (entry_to_spin_system), `generate/`
  driver + 8-group filter. Investigate the **deg-2-vs-split inconsistency** (a fresh
  `build_spin_system` merged the OCH₂ to deg-2 while `records_3M` split it into two
  1H groups — conformer-dependent grouping). Assess whether a refactor is warranted.

Each workstream → a severity-ranked findings list (file:line, repro, recommended fix).

---

## 2. Touchpoints (what a regeneration cascades to)

1. **Generation code** — `mol_to_spin_system/*`, `generate/*` (fixes + refactor).
2. **Regression tests** — `model/tests/`, `generate` tests, new equivalence/bake tests.
3. **Dataset** — full 3M PubChem rebuild: `records_3M{,_train,_test}.json.gz`, stacked
   `parts/`, the leakage-controlled split (re-cluster after values change).
4. **Models** — retrain ALL PubChem tiers: 64k, 500k (57M), 3M (137M) × configs.
5. **Website (docs/)** — every data file derived from the dataset/models must be
   regenerated: `dataset_explorer.json` (histograms), `test_explorer.json`,
   `learning_curves.json`, `test_eval.json`, comparison table; copy that describes
   the data (dataset.html heuristics, stages 5/6).
6. **Held-out eval** — `eval_heldout`, the global split, `heldout_eval.json` per run.
7. **Refinement** — `model.inference.refine` uses the exact renderer; verify it stays
   consistent with the regenerated data's lineshape/values.

---

## 3. Phased plan (gates)

1. **Audit** (3 agents) → populate the Findings ledger (§5). GATE: findings reviewed.
2. **Code fixes / refactor** — equivalence-keyed sharing (independent Gaussian for
   non-equivalent groups) + all CONFIRMED findings. Schema change to carry per-group
   equivalence-class id end-to-end. GATE: regression tests green.
3. **Regression tests** — equivalence-keying test (the mol_3078877 case must produce
   distinct aromatic shifts), diastereotopic determinism, coupling/shift spot-checks.
4. **Dataset regeneration** — full 3M rebuild on HPC; re-run leakage split.
5. **Retrain** — 64k → 500k → 3M PubChem models on the fixed data (I/O-fix launch).
6. **Website propagation** — regenerate all docs/data + refresh copy.

---

## 4. Decisions / conventions (LOCKED)
- **Two distinct concepts — do not conflate:**
  - **Symmetry equivalence (orbit):** structural; homotopic/enantiotopic protons
    (`CanonicalRankAtoms` orbit). Drives **shift sharing in generation** — same orbit
    ⇒ exact same drawn shift; different orbit ⇒ **independent Gaussian** (even if the
    Pretsch base shifts coincide). This is the `augment` fix.
  - **Soft-equivalence FLAG (the graph edge label):** = chemically equivalent (same
    orbit) AND not hard-equivalent (not rotor-merged). PURELY symmetry-determined.
    `soft_equiv_target[i,j] = (equiv_orbit[i]==equiv_orbit[j])` for distinct groups,
    built in the dataset and consumed by `soft_equiv_loss` (the old `|Δδ|≤tol`
    derivation is REMOVED — it conflated symmetry with coincidence).
  - **Accidental equivalence:** a *phenomenon*, not a label — distinct protons whose
    shifts merely overlap. The model must know nothing about it as a concept. Such
    cases SHOULD exist in the data (close shifts, different orbits, flag=0) so the model
    learns "close ≠ soft-equivalent"; they were only ever *over-represented* by the old
    forced sharing, and independent Gaussians restore their natural (rare) rate.
- **Convention:** degeneracy (one HARD node) = a freely-rotating rotor's magnetically-
  equivalent protons (CH₃→3, tBu→9; may span carbons). Everything else (CH₂, aromatic,
  cross-atom symmetry) = separate nodes; co-orbit → shared shift, else independent.
  No grouping change needed (matches `classify_spin_groups`); fix is sharing-only.
- **Conformers:** fixed seed + pinned RDKit (no consensus, no dropping); skip embed-fail.

---

## 5. Findings ledger  *(populated by the audit agents)*

### A — Equivalence & augmentation  *(complete)*
Data flow (verified): `classify_spin_groups` computes true SOFT/HARD classes →
survives into `entry_to_spin_system`'s `class_of` dict and the XYZ tier column →
**dropped at `xyz.py::to_dict` (line 42)** → record has no class id → `sample_record`
re-guesses from `(shift,range)`.

- **A1 — CRITICAL.** `augment.py:148` keys shift-sharing on `(shift, range)`; `range`
  is always degenerate (`min==max==mean`, `xyz.py:144`) so it collapses to shift alone
  → distinct positions with colliding Pretsch shifts get locked to one draw.
  **REFINED FIX (verified on COC1=CC(C)=CC(OC)=C1 & COC1=CC=C(OC)C=C1):** the correct
  "must-share-a-shift" key is the **canonical symmetry orbit** of the group's parent
  atom (`Chem.CanonicalRankAtoms(mol, breakTies=False)`), NOT the SOFT tier and NOT
  `(shift,range)`. The orbit is BROADER than the classifier's group/class: e.g. two
  symmetry-equivalent OMe are *separate HARD groups* but the same orbit and MUST share a
  shift — a SOFT-tier key would wrongly let them drift apart. Rule: **sample one shift
  per symmetry orbit, broadcast to its groups; independent Gaussian across orbits**
  (even when Pretsch base shifts coincide). Thread a per-group `equiv_orbit` id (from
  canonical rank, computed where the mol is in hand — `entry_to_spin_system`) into the
  record and key `sample_record` on it.
- **A2 — HIGH.** Per-group equivalence-class id never persisted: `xyz.py::to_dict`
  (42–61) emits labels/spin_groups/couplings/shift_range/coupling_types but not class
  membership, though `entry_to_spin_system` holds `class_of` (line 149). Root enabler of A1.
- **A3 — MEDIUM.** 1,3,5-symmetric aromatics (e.g. mesitylene) classified SOFT not HARD,
  contradicting the docstring contract (`spin_equivalence.py:420–469`); affects group
  COUNT not sharing. Consider a canonical-rank/automorphism-orbit test for magnetic equiv.
- **A4 — MEDIUM (owner decision).** `merge_enantiotopic` hard-wired ON
  (`spin_equivalence.py:319,524`) but `generate/CLAUDE.md` says don't merge. Count is
  unaffected (still separate labels); it forces enantiotopic partners to share a shift
  (defensible — isochronous in achiral env). Reconcile doc vs intent explicitly.
- **A5 — LOW.** Couplings sampled independently per group-pair — CORRECT for AA′BB′
  (no sharing bug). `sign(0)*…` zeros a sampled J if base J==0 (edge case).
- **A6 — LOW.** `np.mean([],axis=0)` → silent NaN shift if a class has no predicted
  atoms (`xyz.py:157–160`); guard empty classes.

### B — Shifts & couplings  *(complete)*
Collision quantification (400 ChEMBL mols, 2620 groups): **24.4% of groups share an
exact δ with a distinct group; 30.6% of aromatic groups collide.** Drivers: (a) the
substituent-blind fused/hetero base-shift paths (B1 — a BUG), (b) genuine additivity
coarseness (inherent). Monosubstituted-benzene additivity verified excellent (±0.05).

- **B1 — CRITICAL.** Substituted heteroaromatics emit *unsubstituted-parent* shifts
  with a confident flag: the "substituted?" test (`shifts_pretsch.py:853–855`) inspects
  only the proton's own carbon, not the whole ring. Furfural ≡ 2-methylfuran shifts
  (off ~0.8 ppm). **Fix:** flag on any ring carbon bearing a heavy substituent; ideally
  per-ring Z-increment tables. Major collision driver between genuinely-distinct positions.
- **B2 — HIGH.** Pyrazine base shift (8.63) is dead code — missing `_NAME` key
  `(6,((1,7),(4,7)))` in `heteroaromatic.py:91–103` → pyrazine falls to 7.30 (off ~1.3 ppm).
- **B3 — HIGH.** Ring vicinal ³J is a single-conformer lottery (`vicinal.py:124–126`):
  cyclohexane J-set flips chair↔twist by embed seed. Reproducible (fixed seed) but not
  correct. **Fix:** Karplus averaged over an ETKDG ensemble, or ring-type defaults.
- **B4 — HIGH (overlaps C).** Diastereotopic CH₂ collapses to one group
  (`groups.py:23` `CanonicalRankAtoms(breakTies=False)`) → geminal ²J dropped (matrix.py
  skips gi==gj) + distinct shifts averaged. The mol_3078877 OCH₂ inconsistency. **Fix:**
  detect diastereotopicity and split deterministically.
- **B5 — MEDIUM.** `vicinal.py:121` applies ethane base (7.3 Hz) to sp²–sp² single bonds
  (butadiene C2–C3 → 7.3, true ~10.4). Branch on hybridization.
- **B6 — MEDIUM.** Geminal additive model undershoots multi-EN-substituted carbons
  (`geminal.py:10`; CH₂Cl₂ −9.2 vs Pretsch −7.5). Nonlinear/per-pair correction.
- **B7 — MEDIUM.** Fused/peri ⁴J missing (`aromatic.py:47` whole-mol shortest path routes
  through fusion bond; naphthalene peri H1–H8 absent).
- **B8 — MEDIUM.** Placeholder rows in `AROM_INCR`: `CH=CH2`≡`CH2Cl` (line 79), `C#N`≡`NCS`
  (line 134) — not transcribed Pretsch values.
- **B9 — LOW.** `shift_range` degenerate (confirms A); cosmetic until augment adds σ.
- **B10 — LOW.** Olefinic geminal =CH₂ base +2.0 vs lit +2.5 (`geminal.py:9`).
- **B11 — LOW.** Per-bond olefinic J uses a single dominant substituent for the whole C=C.

### C — Grouping, matrix, driver & consistency  *(complete)*
**Architecture finding:** TWO independent grouping algorithms. (1) **Dataset path**
`generate/pipeline.py → classify_spin_groups → xyz_writer → entry_to_spin_system`
(3D deuterium-substitution HARD/SOFT/NONE; splits enantiotopic + diastereotopic CH₂ —
CORRECT). (2) **Utility path** `mol_to_spin_system/pipeline.py → build_spin_system →
groups.py:proton_groups` (CanonicalRankAtoms, does NOT split). The earlier "OCH₂
deg-2 vs split" was just me calling path (2); the dataset (path 1) split it correctly.

- **C1 — HIGH.** `groups.py:proton_groups` splits neither enantiotopic nor
  diastereotopic CH₂; its docstring is false. Not used for the dataset (only a utility +
  tests). **Fix:** make `build_spin_system` delegate to `classify_spin_groups` (single
  source of truth) or delete path (2); at minimum fix the docstring.
- **C2 — HIGH.** Production classification is conformer-dependent: scanning 200 mols at
  4–6 ETKDG seeds, **2% flip tier, 0.5% flip group COUNT** (e.g. CHEMBL8185: 8 groups on
  3 seeds, 7 on 3). The fixed seed pins one sample of a multimodal answer → unstable
  across RDKit/platform → train/serve skew + unstable regeneration. **Fix:** classify
  over multiple conformers (consensus tier) and/or drop count-unstable mols; pin RDKit.
- **C3 — MEDIUM.** 2D embed-failure fallback (`use_3d=False`) mis-groups diastereotopic
  CH₂ as an (impossible) 2-H HARD rotor → wrong count. **Fix:** skip embed-failed mols
  (or guard `_is_magnetically_equivalent` behind the rotor-shape check).
- **C4 — MEDIUM.** InChI is ALWAYS empty: `xyz_writer.py:186` references `mol` not
  `mol_h` → NameError swallowed by bare except. **Fix:** `MolToInchi(mol_h)` + tighten except.
- **C5 — LOW.** `generate/pipeline.py:353–362` writes in completion order (nondeterministic
  row order); `dedup.py` "first occurrence" → surviving representative varies run-to-run.
  **Fix:** sort final dataset by a stable key (InChIKey) for byte-reproducibility.
- **C6 — LOW.** Same doc/intent mismatch as A4 (merge_enantiotopic) — code correct, doc stale.

**C verdict:** the dataset path is deterministic under a fixed seed with the correct
grouping rule; before regeneration MUST fix C4, harden C2/C3 (consensus or drop +
pin RDKit), and C5 (ordering/dedup). Unify the two grouping paths (C1) to prevent future divergence.

---

## 7. Synthesis & recommended rework (post-audit)

The generation code has **one critical correctness bug (A1/A2)**, a **high-impact shift
bug (B1)**, **determinism/stability gaps (C2–C5)**, **two divergent grouping paths
(C1)**, and a cluster of **constant/rule errors (B2,B5,B6,B8,B10)**. A targeted rework is
warranted (not a full rewrite — the dataset path's core is sound).

**Rework workplan (Phase 2 — code), with regression tests for each:**
1. **Equivalence-keyed sharing (A1/A2):** persist per-group `equiv_class` id in
   `entry_to_spin_system`/`to_dict`; key `sample_record` on it; independent Gaussian for
   all non-siblings. Regression: mol_3078877 → distinct aromatic shifts.
2. **Heteroaromatic substituents (B1) + pyrazine (B2):** whole-ring substituent flag;
   add missing `_NAME` entry; (stretch) per-ring Z-increments.
3. **Unify grouping (C1):** single source of truth via `classify_spin_groups`.
4. **Determinism/stability (C2/C3/C4/C5):** multi-conformer consensus + drop
   count-unstable, skip embed-failed, fix InChI, stable sort/dedup, pin RDKit.
5. **Coupling/constant fixes (B5,B6,B8,B10):** hybridization gate, multi-EN geminal,
   transcribe AROM_INCR rows, =CH₂ base.
6. **Defer/low:** B7 peri-⁴J, B11 per-bond olefinic, A6 NaN guard.

## 8. Open decisions (owner) — gate the regeneration
- **D1 (conformer stability, C2/C3):** consensus over K conformers + DROP molecules whose
  group count is unstable (recommended) vs keep single-seed. Affects dataset composition.
- **D2 (symmetric aromatics, A3):** make 1,3,5-symmetric ring H truly HARD (automorphism
  test) — chemically correct but changes group counts / dataset membership.
- **D3 (rework scope):** the staged rework above (recommended) vs minimal (A1/A2 + B1/B2
  only) for a faster regeneration.

---

## 6. Status log
- 2026-06-03 (eve): **AUDIT-2 LANDED — regenerated, split, retraining, website live.**
  - **Regeneration done**: 1605/1605 shards, **0 failures**; spot-check (23,402 records / 12
    shards) → 0 missing `equiv_orbit`, 0 co-orbit pairs failing to share a shift, cross-orbit
    coincidental overlap **0.031%** (natural rate, no longer forced).
  - **Merged + leakage-split**: `records_3M.json.gz` = **3,126,829** molecules (spectra rows
    ALIGNED); cluster split → **train 2,814,147 / test 312,682** (10.00%, leakage-controlled).
  - **Fleet retraining** on corrected data — the data×capacity sweep at the **026 recipe**:
    64k=`light` **10.05M**, 500k=`med` **56.6M**, 3M=`xl` **137.4M** (`train_3M_026`
    created). Plus the **64k ablation sweep** 025 (matrix-only) / 027 (focal) / 028 (cum-integral)
    / 029 (026+focal). **64k·026 finished** (early-stop ep98): held-out **test 0.0452 ppm /
    1.38 Hz / F1 0.909 / deg 0.950 ≥ val** → no overfitting.
  - **Website propagated** (PRs #121/#122/#123/#124, merged): dataset-explorer stats regenerated
    from the corrected train split; 3.16M→3.13M; dashboard filter→`rebuild`; model-viewer
    generators retargeted to the rebuild fleet (turnkey `docs/refresh_website_data.sh`); the
    **held-out test-set evaluation + molecule explorer are live** for the finished 64k·026.
  - Remaining: 500k/3M finish → backfill their viewer columns; 027/028/029 backfill as they land.

- 2026-06-03: **Validation PASSED at scale** (3000-mol sample, fixed pipeline):
  co-orbit groups always share (0 violations); **0 FORCED cross-orbit duplicates**
  (seed-persistence test over 5 seeds) — the old forced-duplication (~24% of groups)
  is eliminated; accidental overlaps survive at their natural rate (unflagged);
  soft-equiv = orbit. Pre-flight: `sim_to_part.py` preserves `equiv_orbit` into
  `recs_K.json` → training. **Full 3M regeneration LAUNCHED** (job 42344849, array
  0-1604%100) on the fixed code; old buggy parts/recs renamed to
  `rebuild3M/{parts,recs}_audit1_buggy`. Next: merge → re-split → retrain → website.

- 2026-06-03: audit opened; root-cause equivalence-keying bug confirmed (mol_3078877);
  3 audit agents dispatched (A/B/C).
- 2026-06-03: Workstream A COMPLETE (findings A1–A6 logged). Root cause precisely
  localized: true class id dropped at `xyz.py::to_dict`; fix = persist per-group
  `equiv_class` + key `sample_record` on it. B (shifts/couplings) + C (grouping/matrix/
  driver determinism) running. Open owner decisions: A3 (symmetric-aromatic HARD/SOFT),
  A4 (enantiotopic-merge doc vs code).
- 2026-06-03: (parallel track) test-time spectral refinement validated on 64k held-out
  — shift MAE 0.073→0.049 ppm (−34%), cos 0.38→0.95, 28/30 improved. Re-validate after
  regeneration (current "true" shifts still carry the artifact).
- 2026-06-03: **Phase-2 code rework COMPLETE** (all findings fixed; 196 tests pass, 1
  skipped). A1/A2: `xyz.py` computes/persists per-group `equiv_orbit` (canonical symmetry
  orbit); `augment.sample_record` keys shift-sharing on it (independent Gaussian off-orbit).
  A6 NaN guard. B1/B2/B8 (shifts), B5/B6/B10 (couplings), C3/C4/C5 (determinism) landed via
  agents with regression tests. `soft_equiv_loss` deliberately UNCHANGED. Verified
  end-to-end on mol_3078877 (cross-orbit Pretsch collision now independent; OCH₂ shares),
  3,5-(MeO)₂-toluene and 1,4-(MeO)₂-benzene (orbit sharing exactly as intended). Next:
  regenerate the 3M dataset on the fixed pipeline → retrain → propagate to the website.
