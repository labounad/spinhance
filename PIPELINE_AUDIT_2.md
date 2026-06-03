# Pipeline Audit 2 — Spin-System Generation Correctness

**Status:** ACTIVE (opened 2026-06-03). Single source of truth for the second
audit + generation-code fix/refactor + full dataset regeneration + retrain of all
PubChem models + website propagation. Update the ledger and status as work lands.

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

## 4. Decisions / conventions
- "Must share a jittered shift" ⇔ **SOFT-equivalent siblings** (genuine AA′BB′), as
  determined by `spin_equivalence`. HARD-equivalent protons are already one group.
  Everything else (incl. coincidentally-equal Pretsch shifts) → **independent draw**.
- Independent draw = base shift + Gaussian(σ from range, floor as today) per group.

---

## 5. Findings ledger  *(populated by the audit agents)*

### A — Equivalence & augmentation  *(complete)*
Data flow (verified): `classify_spin_groups` computes true SOFT/HARD classes →
survives into `entry_to_spin_system`'s `class_of` dict and the XYZ tier column →
**dropped at `xyz.py::to_dict` (line 42)** → record has no class id → `sample_record`
re-guesses from `(shift,range)`.

- **A1 — CRITICAL.** `augment.py:148` keys shift-sharing on `(shift, range)`; `range`
  is always degenerate (`min==max==mean`, `xyz.py:144`) so it collapses to shift alone
  → distinct positions with colliding Pretsch shifts get locked to one draw. Repro
  (mol_3078877, seed 0): classifier says B,C,D,G,H = NONE (distinct), E,F = SOFT (OCH₂);
  yet C==D and G==H get the same draw (artifacts). **Fix:** thread a per-group
  `equiv_class` id into the record (data already in `class_of`), key `sample_record`'s
  `classes` dict on it, independent-draw fallback. Only genuine SOFT/HARD siblings share.
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

### B — Shifts & couplings
- _pending_

### C — Grouping, matrix, driver & consistency
- _pending_

---

## 6. Status log
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
