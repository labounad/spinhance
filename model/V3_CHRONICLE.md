# v3 Development Chronicle

Running log of the gauge-equivariant output-representation effort. Design + rationale:
`model/PERMUTATION_INVARIANT_DESIGN.md`. Append newest entries at the bottom of §Runs.
Every run records: run-id, config hash, what changed, status, and held-out result vs the
matched v2 baseline. Keep entries terse and factual so this doubles as a debug ledger.

## Baselines (v2, already trained — the numbers to beat)
| run | tier | held-out shift-MAE (ppm) | J-MAE (Hz) | presence-F1 | deg-bal-acc |
|---|---|---|---|---|---|
| rebuild_64k_025_v2 | light/10M | 0.0460 | 1.214 | 0.904 | 0.948 |

(n_test=20000, the standard global held-out set; run `20260609_001552_rebuild_64k_025_v2_774240`.)
> v3 MUST be eval'd on the SAME 20k held-out set via `eval_heldout` for apples-to-apples
> (its internal 6400-mol test split is a different, smaller set — do not compare to that).

## Runs
### v3_pia_64k — Rung 1, PIA (Sinkhorn-align), τ=0.05
- **Started:** 2026-06-09.
- **Config:** `model/configs/train_64k_v3_pia.yaml` (= 025 recipe, `matrix` →
  `sinkhorn_align`). Data: consolidated_v2 PubChem, MAXN=64000, 90 MHz. Same light tier,
  same WSD schedule, seed 0 as 025.
- **Hypothesis under test:** H1/H2/H3 (design §3) — permutation-invariant loss removes the
  near-degenerate ranking penalty without harming resolvable couplings.
- **Watch:** `assign_offdiag` (should be ~0 on resolvable, >0 on near-degenerate);
  shift/jmag/presence/deg sub-losses vs 025; held-out near-degenerate-stratified J-MAE.
- **run-id / hash:** `20260609_185637_v3_pia_64k_9bdcbc` (sbatch 42403947, a100/nodec0823).
- **Status:** RUNNING (launched 2026-06-09 18:56). 10.05M params (= 025), split
  44800/12800/6400 leak-free, ~0.075 s/step → ~100 ep in well under an hour. Plumbing
  validated through ep2: loss 0.75→0.62, no NaN, grad_norm ~3-4, all `sinkhorn_align/*`
  metrics logged. Parallel to v2 (separate a100 node; v2 3M runs untouched on c0819/c0821).
- **WATCH:** `assign_offdiag` = 0.75 at ep2 — expected high while predicted shifts are
  still untrained/unsorted (P≈uniform). It MUST fall toward ~0 for resolvable molecules as
  shifts sharpen (P→identity), leaving off-diagonal mass only on genuine near-degeneracies.
  If it stays high, τ=0.05 is too soft (loss isn't anchoring the ranking) → lower τ and
  relaunch. Check at ep~20/50.
- **Result: FAILED (cold-start).** Killed at ep17. `assign_offdiag` stuck ~0.73 from ep4
  (never anchored); val collapsed/worsened (ep17 shift **1.19 ppm**, J **7.16 Hz**, F1
  **0.38** vs baseline 0.046/1.214/0.904). **Diagnosis — chicken-and-egg:** from a cold
  init predicted shifts are ≈constant, so the assignment cost `(psh−tsh)²` is near-uniform
  across pred nodes → P stays loose → the averaged shift loss `Pᵀ·shifts` gives no pressure
  to make shifts *distinct & ordered* → the model never escapes the mushy regime (training
  loss falls by collapsing shift variance, val craters). **This is the warm-start lesson
  (design §2) confirmed for the STRUCTURAL loss, not just the spectral one:** soft
  assignment only has signal once the basin (sharp shifts) exists. τ wasn't the (sole)
  problem — cold-start was.

### v3_pia_64k_warmup — Rung 1b, matrix→sinkhorn HANDOFF
- **Config:** `train_64k_v3_pia_warmup.yaml`. Composite: `matrix` (weight 1.0, decays
  40→55 to 0) then `sinkhorn_align` (ramps 40→55 to 1.0, τ=0.05). So 0–40 pure matrix
  establishes sharp/ordered shifts; 55–100 pure sinkhorn relaxes near-degenerate ranking on
  a good basin. Since sinkhorn ≈ matrix for resolvable shifts (τ→0 equivalence), the
  crossfade is near-continuous in loss — only near-degenerates change.
- **Watch:** `assign_offdiag` should be ~0 at handoff (sharp shifts) and stay low except on
  near-degenerate molecules; val J-MAE should hold ≈ matrix through the handoff then ideally
  improve on near-degenerate-stratified J-MAE.
- **run-id / hash:** `20260609_191043_v3_pia_64k_warmup_b4cb2c` (sbatch 42403950, a100).
- **Result: warmup GREAT, τ=0.05 sinkhorn too soft.** Warmup phase trained like 025
  (ep38: shift **0.068**, J **1.398**, deg-bal 0.949 — best_ep=38). But at the handoff
  (ep42) `assign_offdiag`=**0.527** *with already-sharp shifts*, and shift_mae regressed
  0.068→0.124 as sinkhorn ramped in. **Diagnosis (measured):** `shift_std`=2.356 ppm, so
  τ=0.05 ⇒ soft/hard boundary √0.05·2.356 ≈ **0.5 ppm** — it treats groups half a ppm apart
  as swappable, so even sharp shifts give a loose P and de-sharpen. τ must be calibrated to
  the standardized shift scale, not picked blind. Killed at ep44.

### v3_pia_64k_warmup (τ=0.001) — Rung 1b recalibrated
- **Change:** sinkhorn τ 0.05→**0.001** (boundary √0.001·2.356 ≈ **0.075 ppm** ≈ 90 MHz
  resolution: ≈matrix-hard for resolvable shifts, soft only for genuine near-degeneracies);
  n_iters 50→80. Same warmup→handoff schedule.
- **run-id / hash:** `20260609_193228_v3_pia_64k_warmup_9cffca` (sbatch 42404015, a100).
- **Result: KEY NEGATIVE — pure-sinkhorn phase DEGRADES shifts even at calibrated τ.** Ran
  all 100 ep. **best_epoch=38** (end of matrix warmup: shift 0.068 / J 1.398). After the
  handoff, as matrix decays to 0 and sinkhorn takes over, val regresses monotonically — by
  ep99 shift **0.28→0.34 ppm**, J **2.76→3.04 Hz**, F1 0.84→0.78; ep99 `failure_analysis`
  dominant = **large_shift_error (52%)** + a grad spike (norm 53 vs ema 5).
- **Interpretation:** with matrix fully removed, sinkhorn-alone de-sharpens shifts — the
  averaged `al_sh = Pᵀ·psh` plus the loss of a hard per-slot shift anchor lets shifts drift.
  So **permutation-invariance must COMPLEMENT the matrix loss (residual anchor), not REPLACE
  it.** The full matrix→sinkhorn decay-to-zero handoff is the wrong design.
- **→ Rung 1c (next):** keep `matrix` at a residual weight (e.g. 0.3, no decay) for the whole
  run and ADD `sinkhorn_align` on top — matrix anchors shifts/structure, sinkhorn relaxes
  only the near-degenerate coupling ranking. Launch residual variants once the running
  τ-sweep + handoff-sweep confirm the across-the-board pure-sinkhorn degradation.

## Parallel sweep (launched 2026-06-09 19:35) — two 1-D axes through the τ=0.001/h40 anchor
rtxa6000 confirmed compatible with the spinhance torch build (the `gpu`/gtx1080 partition is
NOT) → ample parallel GPU capacity. All warmup→handoff, 64k/light, vs 025 baseline
(J 1.214 / shift 0.046 / F1 0.904 / deg 0.948). Eval each best.pt+last.pt on the same 20k
held-out, stratified by near-degeneracy, when done (~22 min/run).

| run | τ | handoff@ | sbatch | node | status |
|---|---|---|---|---|---|
| v3_pia_64k_warmup (anchor) | 0.001 | 40 | 42404015 | a100 | RUNNING |
| v3_pia_t0005_h40 | 0.0005 | 40 | 42404017 | rtxa6000 | RUNNING |
| v3_pia_t002_h40  | 0.002  | 40 | 42404018 | rtxa6000 | RUNNING |
| v3_pia_t004_h40  | 0.004  | 40 | 42404019 | rtxa6000 | queued |
| v3_pia_t001_h30  | 0.001  | 30 | 42404020 | rtxa6000 | queued |
| v3_pia_t001_h20  | 0.001  | 20 | 42404021 | rtxa6000 | queued |
| v3_pia_t001_h10  | 0.001  | 10 | 42404022 | rtxa6000 | queued |

τ axis maps the soft/hard boundary (√τ·2.356 ppm): 0.0005→0.053, 0.001→0.075, 0.002→0.105,
0.004→0.149 ppm. Handoff axis maps how little matrix warmup the soft assignment can bootstrap
from (h10 is near-cold — expect it to approach the cold-start failure; brackets the minimum).

### Sweep RESULTS (2026-06-09 ~20:45)
**τ-sweep — the degradation is τ-INDEPENDENT.** All four τ (incl. the 0.001 anchor) peak at
the matrix-warmup (best_ep ~35–38, shift ~0.064–0.070 / J ~1.39–1.44) then the pure-sinkhorn
phase degrades to shift ~0.22 / J ~2.6–2.9:
| τ | best (shift/J) | final (shift/J) |
|---|---|---|
| 0.0005 | 0.067 / 1.429 | 0.219 / 2.631 |
| 0.001 (anchor) | 0.068 / 1.398 | 0.34 / 3.04 |
| 0.002 | 0.064 / 1.393 | 0.228 / 2.713 |
| 0.004 | 0.070 / 1.436 | 0.217 / 2.895 |
→ **Not a τ problem. Pure sinkhorn (matrix removed) de-sharpens shifts at every τ.** Confirms
the residual-anchor diagnosis decisively.
**Handoff-sweep (running): earlier handoff = worse best** (h10 best 0.140 ≫ h40 best 0.067) —
less matrix warmup → worse, as predicted (approaches cold-start).

### Rung 1c — matrix RESIDUAL anchor + sinkhorn on top (launched ~20:45)
Configs `train_64k_v3_pia_resid{02,03,05}.yaml`: `matrix` stays at weight {0.2,0.3,0.5} (NO
decay) for the whole run; `sinkhorn_align` (τ=0.001) ramps in at ep40. Hypothesis: matrix
anchors shifts/structure so they don't de-sharpen, while sinkhorn relaxes only the
near-degenerate coupling ranking → val should HOLD ≈ matrix (no shift regression) and ideally
improve near-degenerate-stratified J. Runs `v3_pia_resid02/03/05` on rtxa6000. **WATCH:** does
post-ep40 val stay flat (vs the τ-sweep's regression)? If yes, Rung 1c is the viable design.

## Smoke tests (correctness gates before any fleet run)
Run: `PYTHONPATH=. python3 /tmp/smoke_sinkhorn.py` (2026-06-09, torch 2.10 local). All PASS.
- [x] **τ→0 continuity (H2):** distinct shifts, τ=0.01 → matrix total = sinkhorn total =
  **1.99155, |diff|=0.000000**; `assign_offdiag`=0.0. Exact reduction to element-wise loss.
- [x] **forward/backward:** finite loss + finite grads. PASS.
- [x] **near-degenerate softening:** equal-shift pair → `assign_offdiag`=0.128 (>0);
  loss **exactly invariant** to relabeling those two target nodes (|diff|=0.000000).
- [ ] **2-epoch micro-run** on 64k via the real trainer (config loads, term registered,
  metrics logged incl. `assign_offdiag`) — pending launch.

## Decisions / lessons (append as they happen)
- 2026-06-09: chose 025 (matrix-only) as the A/B base over 026 to isolate the loss change
  (no soft_equiv/peak-channel confounds). 026-based variant deferred.

### Rung 1c RESULT + Rung 1d (launched ~21:13)
Rung 1c (matrix residual {0.2,0.3,0.5} + sinkhorn weight 1.0) STILL degrades post-handoff
(resid02/03 at ep48-49: shift 0.067→~0.21, J back up) — because sinkhorn(1.0) OVERPOWERS the
matrix residual. The anchor must be the DOMINANT term, not a minority residual.
**Rung 1d:** matrix weight 1.0 (dominant, no decay) + sinkhorn_align AUXILIARY at low weight
{0.2, 0.5}, ramp@40, τ=0.001. Runs `v3_pia_aux02/aux05`. WATCH: val should now HOLD ≈ matrix
(matrix dominates) while sinkhorn nudges near-degenerate couplings. If aux holds AND improves
near-degenerate-stratified J vs pure-matrix 025, that's the win condition.

### Rung 1d/1e RESULTS + Rung 1f (the isolation fix)
- **Rung 1d (matrix 1.0 + sinkhorn aux):** FIRST ADDITIVE SIGNAL. aux02 finished J 1.42→**1.26**
  (improved!) but shift 0.069→0.130 (regressed); aux05 worse shift, same J. Lower sinkhorn
  weight better. So permutation-invariance HELPS J but the term also costs shift.
- **Rung 1e (coupling-only, shift weight 0):** STILL regressed shift (jonly05 0.069→0.114).
  Diagnosis: the soft assignment P is a function of psh, so the coupling-alignment gradient
  flows into shifts THROUGH P even with no shift loss term.
- **Rung 1f (coupling-only + `detach_assign_shifts=true`):** detach psh in the Sinkhorn cost
  → P uses current shifts to assign but couplings can't push shifts (verified: shift grad
  norm = 0). Shifts now PURELY matrix-supervised. Runs `v3_pia_jdetach05/10`. **WIN CONDITION:
  shift holds ≈ matrix warmup (~0.069) AND J improves (<1.40).** If met, this is the validated
  minimal form of the permutation-invariant term → then held-out eval (near-deg-stratified).

### Rung 1f RESULT — WIN (preliminary, ep47, confirming at completion)
detach_assign_shifts WORKS. Past the handoff, BOTH variants hold shift AND improve J:
  jdetach05: warmup 0.074/1.42 -> ep47 **0.063/1.34**
  jdetach10: warmup 0.070/1.40 -> ep47 **0.059/1.36**
Shift held (slightly better) + J improved — the permutation-invariant term is purely ADDITIVE
on couplings, zero shift cost. This is the validated minimal form:
  **matrix (full) + sinkhorn_align coupling-only (weights shift/deg/presence=0) with
  detach_assign_shifts=true, ramped in after a matrix warmup.**
TODO at completion: held-out eval (20k, near-degeneracy-stratified) on jdetach05/10 best+last
vs 025 baseline (held-out J 1.214 / shift 0.046) to quantify the gain and confirm it
concentrates on near-degenerate molecules (the hypothesis).

### Rung 1f FINAL (ep97) — degradation fully solved
After LR decay both jdetach variants land at/above the matrix baseline on aggregate val:
  jdetach05: best ep90 shift **0.049** / J **1.22** ; jdetach10: best ep76 0.048 / 1.27.
(025 held-out: 0.046 / 1.214.) So the detached coupling-only sinkhorn is non-destructive AND
the model matches pure-matrix on aggregate. Aggregate held-out evals launched (ev_jdetach05/10
on 20k). NEXT: near-degeneracy-STRATIFIED held-out J-MAE (jdetach vs 025) — the hypothesis is
the gain concentrates on near-degenerate molecules (~6%), invisible in the aggregate. Need a
custom stratified eval (split held-out by min pairwise |Δδ|); build next tick.
