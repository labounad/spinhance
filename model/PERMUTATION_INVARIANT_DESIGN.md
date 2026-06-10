# v3: Gauge-Equivariant Output Representation — Design Document

**Status:** ACTIVE (started 2026-06-09). Preliminary upgrade candidates explored **in
parallel** with the production "v2" fleet. v2 runs are untouched.
**Owner:** Lucas (Task-3 / model). **Companion log:** `model/V3_CHRONICLE.md`.

---

## 1. Problem statement

The model predicts a **spin-system graph**: nodes = groups (chemical shift δ, degeneracy
n), edges = J-couplings. The physical object is this graph **modulo the node-permutation
group Sₙ** — the node labels are a *gauge* with no observable consequence. The only
observable, the 90 MHz spectrum, is a permutation-invariant function of the system.

Today we train a regressor against a **canonical section** of that quotient — "sort nodes
by shift, descending" — with an **element-wise** loss (`model/losses/matrix_loss.py`:
smooth-L1 on shifts and masked couplings, BCE on presence, CE on degeneracy). That works
where the section is smooth and single-valued. It is **neither** at degeneracies:

* **Exactly-equal shifts** (true chemical/magnetic equivalence): the sort order is an
  *arbitrary tie-break*. The section is **multivalued**; the supervision is noise.
* **Near-equal shifts** (accidental near-degeneracy): the section is **nearly
  discontinuous** — an infinitesimal change in two close shifts flips their sort order and
  **transposes whole rows/columns of the coupling matrix**. The loss surface inherits a
  branch cut at every shift coincidence.

So the element-wise loss demands the model **perfectly rank every group**, including ones
the 90 MHz spectrum cannot resolve. The harshness is worst exactly where the observable
is least informative — backwards. This is the defect Lucas identified: *we are not
representing the output properly; the loss lives on the wrong space.*

Eval already half-acknowledges this: `model/evaluation/symmetry.py::align_pred_couplings`
aligns predictions over **exactly-equal-shift orbits** before scoring J-MAE. PR #178
propagated that to the explorer display. But it is a *post-hoc patch on the section*: it
removes the multivaluedness at exact ties and does nothing for near-ties — by design.

## 2. Prior art in this repo (what failed, and the lesson)

* **Element-wise canonical** (`matrix`, production) — the status quo above.
* **Hungarian** (`model/losses/hungarian_loss.py`) — hard node assignment by shift, then
  compare couplings under it. *Failed:* a single wrong hard match transposes a whole
  row/col, scrambling the relational coupling structure; node-assignment ≠ graph-matching.
* **Pure spectral / analysis-by-synthesis from scratch**
  (`model/configs/train_64k_surrogate_spectral*.yaml`) — train against the simulated
  spectrum (inherently permutation-invariant). *Failed spectacularly:* (a) ill-posed —
  the loss is flat along the fiber of spectrum-equivalent systems; (b) ~1 Hz linewidths
  make the loss ≈ flat unless peaks already overlap, so it cannot bootstrap from a cold
  init; (c) eigh gradients blow up at level crossings (= degeneracies); (d) no signal for
  discrete structure (group count, integer degeneracy, coupling presence).

**The lesson (load-bearing):** `model/inference/refine.py::refine_system` does
analysis-by-synthesis *at test time* and works (+43%/+77%) — **because it is warm-started
from a good point estimate**. The same machinery failed at training time **because it was
cold-started**. Spectral consistency has signal *only inside the basin*; something else
must establish the basin. The previous mistake was making the spectrum do **two jobs at
once** — supply invariance *and* supply supervision. It is good at the first and
catastrophic at the second. The v3 thesis is to **decouple** them.

## 3. Hypotheses

* **H1 (diagnosis).** The near-degenerate penalty is caused by the canonical-section
  discontinuity, not by genuine model error. A permutation-invariant loss that is *only as
  flexible as the shifts are close* will remove it **without** improving (or harming)
  well-separated couplings.
* **H2 (no over-credit).** Such a loss does not inflate apparent quality on resolvable
  structure — at τ→0 with distinct shifts it reduces **exactly** to the element-wise
  matrix loss.
* **H3 (headroom).** Removing the artificial discontinuity yields a measurable
  improvement on held-out J-MAE / degeneracy-balanced accuracy at fixed capacity, most of
  it concentrated on near-degenerate molecules.

## 4. Approach — implementation ladder

Built as a ladder so each rung **de-risks the next** and is independently shippable.

### Rung 1 — PIA: permutation-invariant alignment loss  *(this rung; implemented)*
Replace the element-wise `matrix` term with a **shift-gated soft-assignment** matrix loss
(`model/losses/sinkhorn_align_loss.py`, `SinkhornAlignLoss`). No architecture change, no
spectral term. Directly tests H1/H2/H3. Detailed in §5.

### Rung 2 — warm-started spectral consistency
Add a differentiable forward-model term (`simulate_spectrum_composite`) **gated to activate
only once shift error is small** (curriculum: λ ramps from 0; or hard-gate on a shift-MAE
threshold) so it never drives from a cold init. This moves the proven test-time refinement
*into* training, as a refinement of a structurally-good prediction. Risk: eigh gradients
near crossings — to be probed on a handful of systems before fleet use.

### Rung 3 — generative posterior
Reframe as amortized posterior inference: a **permutation-equivariant** conditional
generative model (graph diffusion / flow) over spin systems, spectrum as **conditioning**
(never the loss target). Trained with a denoising objective on the data manifold (good
gradients), so it sidesteps the rugged spectral landscape entirely. The posterior spread
*is* the spectrum's information content; near-degenerate ambiguity lives in the
distribution, not in a forced point estimate. Most ambitious; gated on Rung 1/2 evidence.

## 5. Rung 1 design — `SinkhornAlignLoss`

### 5.1 Construction (per batch element; standardized space, matching `matrix`)
Inputs: predicted `psh (G)`, `pJ (G,G)`, `pPres (G,G)` logits, `pDeg (G,C)`; target
`tsh (G)`, `tJ (G,G)`, `tmask (G,G)`, `tdeg (G)`. G = 8 (`model/schemas/constants.py`).

1. **Cost** `C[i,j] = (psh[i] − tsh[j])²` — pred node *i* vs target node *j* by shift
   proximity (v3.0: shift only; see §8).
2. **Soft assignment** `P = Sinkhorn(−C/τ)` in log-space (`sinkhorn_log`): doubly-
   stochastic, `P[i,j]` = mass assigning pred *i* → target *j*. τ = entropic temperature.
3. **Align prediction into the target frame:**
   * shifts: `al_sh[j]   = Σᵢ P[i,j]·psh[i]`            (`einsum bij,bi->bj`)
   * couplings: `al_J = Pᵀ J P`                          (`einsum bij,bik,bkl->bjl`)
   * presence: `al_Pres = Pᵀ Pres P` (logits)
   * degeneracy: `al_Deg[j] = Σᵢ P[i,j]·pDeg[i]`         (`einsum bij,bic->bjc`)
   The **bilinear `Pᵀ J P`** is the crux: couplings move *with* their nodes, so edge
   structure is preserved (the thing Hungarian broke).
4. **Standard terms on the aligned prediction** vs the fixed target: smooth-L1 shift,
   masked smooth-L1 jmag (upper-tri), BCE presence, CE degeneracy. Same weights as `matrix`.

### 5.2 Why it is correct and bounded
* **Continuity / H2:** distinct shifts + small τ ⇒ `C` strongly favors the diagonal ⇒
  `P → I` ⇒ `al_* = pred` ⇒ **exactly the element-wise matrix loss.** (Smoke-tested.)
* **No over-credit:** `P` is gated by *shift proximity*, so it can only relabel nodes the
  observable already cannot distinguish; the bilinear form **rearranges** coupling values,
  never changes them. A genuinely wrong coupling between resolvable shifts cannot be
  relabeled into a match. (Contrast full-S8, which over-credits — see
  `project_jmae_alignment` / the 30k study.)
* **Differentiable & stable:** Sinkhorn is a smooth fixed point of log-space
  normalizations; gradient flows through `P`. No discrete assignment, no eigh.

### 5.3 Live diagnostics (logged every step via `LossOutput.metrics`)
* `assign_offdiag` = mean(1 − diag(P)): 0 ⇒ identity (resolvable); >0 ⇒ relabeling
  (near-degenerate). **This is the direct empirical test of H1** — off-diagonal mass
  should concentrate on near-degenerate molecules and be ~0 elsewhere.
* `assign_entropy` = mean row entropy of P (assignment softness).
* `shift`, `jmag`, `presence`, `deg` sub-losses (parity with `matrix`).

### 5.4 Hyperparameters
* `tau` (default 0.05, standardized-shift² units): smaller → harder/closer to matrix loss.
  The knee — large enough to soften genuine near-degeneracies, small enough to stay ≈
  identity for resolvable shifts — is the main thing to sweep.
* `n_iters` (default 50): Sinkhorn iterations; 50 is ample for G=8.

## 6. Experiment plan

| run | config | base | change | baseline to beat |
|---|---|---|---|---|
| `v3_pia_64k` | `train_64k_v3_pia.yaml` | 025 | matrix → sinkhorn_align (τ=0.05) | `rebuild_64k_025_v2` |

**Tier:** start at 64k/light (~10M) — fast A/B, the 025 baseline is already trained.
**Data:** consolidated_v2 PubChem (identical to the v2 64k fleet; via `train_v2.slurm`).
**Metrics (held-out, `eval_heldout.py`):** shift-MAE (ppm), J-MAE (Hz), presence-F1,
deg-balanced-acc — plus a **near-degenerate-stratified J-MAE** (split molecules by min
pairwise |Δδ|) to localize where any gain comes from.

**Success criteria:**
* **H2 sanity:** on well-separated-shift molecules, v3 J-MAE ≈ 025 (no regression).
* **H1/H3:** on near-degenerate molecules, v3 J-MAE < 025 by a meaningful margin; overall
  J-MAE ≤ 025 with deg-balanced-acc not worse.
* **Diagnostic:** `assign_offdiag` tracks the near-degenerate fraction (confirms the
  mechanism, not a coincidental gain).
**Decision:** if H1–H3 hold → sweep τ, then promote to 500k and proceed to Rung 2.
If v3 ≈ 025 everywhere → H1 is wrong (the penalty is real model error, not the section);
stop and reconsider. If v3 worse on resolvable → τ too high / bilinear leakage; lower τ.

## 7. Run & namespace conventions (parallel to v2 — non-negotiable)
* Run names are **`v3_*`** (never `*_v2`). Launch via the existing `train_v2.slurm` with
  `CONFIG=<v3 yaml>`, `RUN_NAME=v3_pia_64k`, `MAXN=64000`. Outputs land in the same
  `$SP/runs/` dir under a distinct, hash-suffixed run-id, so no v2 artifact is touched.
* Fleet monitor / GPU usage: v3 is one extra `gpu:1` job; it must not preempt v2.
* Every run is appended to `model/V3_CHRONICLE.md` with its run-id, config hash, and result.

## 8. Known limitations / planned refinements
* **v3.1 — degeneracy-aware cost.** Two near-equal-shift nodes of *different* multiplicity
  ARE spectrum-distinguishable; v3.0's shift-only cost would soft-swap them. Add a
  degeneracy-mismatch penalty to `C` (forbid those swaps). Deferred to keep v3.0 a clean
  test of the shift-ranking hypothesis.
* **Presence-logit averaging.** `Pᵀ Pres P` averages *logits*; exact only at P=I. Harmless
  near convergence; revisit if presence-F1 regresses.
* **Padding nodes (G fixed at 8).** Inactive groups are included uniformly (as `matrix`
  does); their couplings are masked out and equal padding shifts only soft-swap among
  themselves (harmless). Watch `assign_offdiag` for pathological padding behavior.

## 9. Reproduction
```bash
# on Garibaldi (parallel to v2; does not touch any v2 run)
cd /gpfs/home/labounader/code/spinhance
sbatch --job-name=v3_pia_64k \
  --export=ALL,CONFIG=model/configs/train_64k_v3_pia.yaml,RUN_NAME=v3_pia_64k,MAXN=64000,PRELOAD=true \
  /gpfs/group/shenvi/Users/labounader/spinhance/train_v2.slurm
# held-out eval after best.pt lands:
PYTHONPATH=. python -m model.experiments.eval_heldout --run_dir $SP/runs/<run-id>
```
