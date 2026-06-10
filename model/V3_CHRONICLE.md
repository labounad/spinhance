# v3 Development Chronicle

Running log of the gauge-equivariant output-representation effort. Design + rationale:
`model/PERMUTATION_INVARIANT_DESIGN.md`. Append newest entries at the bottom of §Runs.
Every run records: run-id, config hash, what changed, status, and held-out result vs the
matched v2 baseline. Keep entries terse and factual so this doubles as a debug ledger.

## Baselines (v2, already trained — the numbers to beat)
| run | tier | held-out shift-MAE (ppm) | J-MAE (Hz) | presence-F1 | deg-bal-acc |
|---|---|---|---|---|---|
| rebuild_64k_025_v2 | light/10M | _TBD (read from heldout_eval.json)_ | _TBD_ | _TBD_ | _TBD_ |

> Action item: pull `rebuild_64k_025_v2/heldout_eval.json` to fill this row before judging v3.

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
- **run-id / hash:** _TBD on launch_
- **Status:** _IMPLEMENTED — smoke-test + launch pending._
- **Result:** _TBD._

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
