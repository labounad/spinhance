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
- **run-id / hash:** _TBD on launch._
- **Status:** _launching (kill+relaunch on the same a100)._
- **Watch:** post-warmup `assign_offdiag` should now be ~0 (sharp P for resolvable), shift
  should NOT regress at the handoff; sinkhorn phase tests the near-degenerate hypothesis.

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
