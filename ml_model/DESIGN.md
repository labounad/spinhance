# Task 4 — Spectrum → Matrix Model: Design Decisions

Living record of the architecture/training decisions for Task 4 (predict the
field-independent shift+J+degeneracy matrix from a low-field ¹H spectrum), with
the reasoning behind each, so we can revisit them later. Update this as choices
change.

## Problem recap

- **Input:** one normalized ¹H spectrum, `2^14 = 16384` intensity points over
  0–12 ppm, ∫ = 1, simulated at the **low field of 90 MHz** (strongly coupled,
  non-first-order). Low-field-only input is the real use case.
- **Output:** an 8×9 representation of a labeled graph —
  - 8×8 symmetric matrix: chemical shifts (ppm) on the diagonal, scalar
    couplings *J* (Hz) off-diagonal;
  - 8×1 vector: per-group proton degeneracy (e.g. 3 = CH₃, 9 = t-Bu).
- The target lives in this space **modulo S₈** (the 8 group labels are
  arbitrary), which is the central modeling difficulty.

## Decisions (with rationale)

### 1. Encoder — **ResNet-1D**
1-D conv residual stack + global pooling → embedding. Fast, stable, well-matched
to peaky local 1-D structure (multiplets are local patterns), few knobs.
Alternatives (patch-transformer for long-range coupling structure; CNN-stem →
transformer hybrid) are deferred to v2 if long-range structure is being missed.
The encoder sits behind a clean interface, so swapping it later is cheap.

### 2. Output heads — four typed heads (not one flat 72-vector)
- 8 shifts → regression (ppm)
- 28 couplings (upper triangle) → regression (Hz)
- 28 coupling-presence → binary logits ("is there a coupling at all"); most
  off-diagonals are 0, so separating *whether* from *how big* keeps the
  magnitude head from fighting thousands of structural zeros.
- 8 degeneracies → classification over a small vocab ({1,2,3,4,6,9,…}); discrete,
  so classification beats regression.

### 3. Target ordering (the S₈ problem) — **canonical sort** (baseline)
Pre-sort the 8 groups by shift descending (tie-break: degeneracy, then |J|
row-sum) so the matrix is well-defined and trainable with plain per-element
losses. Weakness: near-equal shifts make the sort order unstable → mild label
noise. Upgrade path (unchanged head structure): set prediction with Hungarian
matching on node features, applied to the coupling block before loss.
**Note:** a spectral loss (Decision 6) is automatically permutation-invariant,
so the ordering only matters for the matrix-loss phase.

### 4. Matrix loss — **standardize + manual weights**
Per-element loss under the canonical ordering:
- shifts → Huber/smooth-L1 (robust to outlier shifts)
- coupling magnitude → Huber, **masked by ground-truth presence** (only penalize
  where a real coupling exists)
- coupling presence → binary cross-entropy; at inference zero any coupling with
  presence prob < 0.5
- degeneracy → cross-entropy over the vocab

Balance the four terms by z-scoring shifts & couplings to unit variance
(train-set stats) then a few hand-tuned weights vs the classification terms.
Transparent and debuggable per-term (vs learned-uncertainty weighting, deferred).

### 5. Differentiable renderer — **torch port of pyspin composite + regularized eigh** ✅ de-risked
For a loss computed *on spectra*, the matrix→spectrum simulator must be in the
autograd graph (MNova can't be — external + slow). pyspin is fast (composite
reduction + connected-component split + Mz-block trick → cost bounded by the
largest coupled fragment, not total protons), but NumPy/SciPy, so not
differentiable as written. The port is narrow because the expensive part is
parameter-independent bookkeeping (block/index maps, composite reps, F⁺ maps);
only H assembly + eigendecomposition + overlaps depend on the parameters.

**Spike result (see `eigh-grad-spike` memory; `test_diff_renderer.py`):**
- Well-separated eigenvalues (~5 Hz gap): naive eigh backward is exact (matches
  finite differences to ~1e-8). No mitigation needed.
- Near-degenerate, sub-Hz gaps (realistic crowded low-field multiplets, ~0.37 Hz):
  naive backward still matched FD to ~1e-8. **Not** a problem.
- Exact degeneracy from expanding equivalent spins (CH₃→3 spin-½, gap ~1e-13):
  the `1/(λᵢ−λⱼ)` term blows up; torch hits 1/0 → NaN. **The real risk.**
- **Mitigation:** Lorentzian-regularized VJP `F_ij = ΔE/(ΔE²+ε²)`. ε from 1e-3
  to a few Hz recovers FD-accurate gradients on degenerate systems and barely
  touches well-separated ones; tie ε to the linewidth (~1 Hz). Implemented as
  `RegularizedEigh` in `diff_renderer_torch.py`.

Architectural note: the dangerous exact degeneracies come specifically from the
*explicit* equivalent-spin expansion. The composite engine represents each group
as total-spin manifolds and never builds those permutation-degenerate states, so
porting composite (which we want for speed anyway) structurally avoids most of
them; the regularized backward is cheap insurance for residual accidental ones.

Files: `diff_renderer_ref.py` (NumPy oracle, verified: forward corr 0.9999 vs
pyspin, gradient ~1e-6 vs FD), `diff_renderer_torch.py` (production twin +
`autograd.gradcheck` self-test), `test_diff_renderer.py` (verification).

### 6. Loss strategy — **staged: matrix anchor → spectral consistency**
- **Stage 1:** matrix loss only (Decision 4) for a stable, identifiable baseline.
- **Stage 2:** add a spectral-consistency loss — render the predicted matrix and
  compare to the real spectrum. Use **Wasserstein** (1-D earth-mover; cheap via
  CDF difference and degrades gracefully when peaks are slightly misaligned,
  unlike intensity-MSE) plus a lineshape term.
  - Render at **90 MHz** (self-consistency: the predicted matrix must reproduce
    the spectrum we actually fed in — the more important, better-posed term) and
    optionally **600 MHz** (the project's reproducibility goal).
  - The exact differentiable renderer (Decision 5) is valid at *both* fields, so
    we are not limited to a first-order approximation.
- Why staged and not spectral-only: at 90 MHz many matrices give nearly the same
  spectrum (the problem is genuinely ill-posed), so a spectral-only signal can
  drift to a spectrum-equivalent but wrong matrix. The matrix loss anchors it.
- Compute spectral Wasserstein as an **eval metric from day one** regardless of
  whether it's in the training loss yet.

### 7. Training scheme

**Optimizer/schedule (baseline):** AdamW, lr ≈ 3e-4, short linear warmup (2–5% of
steps) then cosine decay, weight decay ~1e-2, grad clip 1.0, bf16 mixed
precision. Batch 256–1024 (input is only 16k floats; encoder sets the ceiling).
Early-stop + checkpoint on validation loss; log each loss component separately.

**Data flow:** precompute the 90 MHz *input* spectra once with pyspin (µs–ms
each; ~6–13 GB at fp16 for 100–200k). Augment on-the-fly each epoch (additive
noise, small global referencing shift, baseline drift, linewidth/phase jitter).
The Stage-2 spectral-loss spectrum is rendered **in-graph from the predicted
matrix every step**, not precomputed.

**Stage 1 → Stage 2 handoff — `curriculum blend`:** pretrain on the matrix loss,
then ramp the spectral-loss weight in from 0 over a few epochs while keeping a
decayed matrix anchor. Avoids cold-start instability (full-strength spectral loss
on an untrained model → erratic gradients) and identifiability drift (spectral
signal alone wandering to a spectrum-equivalent wrong matrix).

**In-graph render cost — `bucket + stochastic subset`:** each molecule's Hilbert
space differs (different degeneracy pattern / coupled-fragment size), so samples
can't be naively stacked into one batched diagonalization. (a) Bucket samples by
degeneracy/structure so each micro-batch shares operator matrices (reuse the
`struct` object `simulate()` accepts) → each bucket is a uniform batch; AND
(b) render only a random ~10–25% of each batch per step. Subsampling leaves the
expected gradient unbiased (just noisier — fine for SGD) while capping per-step
cost.

### 8. Data split — **scaffold split + matrix dedup**, stratified, by molecule
70/20/10 train/val/test. Non-negotiable: assign folds **at the molecule level**
so all of a molecule's derived spectra (90 + 600 MHz + augmentations) stay in the
same fold — otherwise the model sees the answer (90 MHz in train, 600 MHz in
test) and the score is inflated.

Grouping into folds:
- **Bemis–Murcko scaffold split (RDKit):** whole scaffolds are assigned to a
  single fold, so test scaffolds are structurally unseen → honest generalization
  estimate (vs random split, which mostly measures interpolation).
- **Near-duplicate matrix dedup:** the model only ever sees the spectrum/matrix,
  so the leakage that matters most is two different molecules with nearly
  identical 8×9 systems straddling folds. After canonical ordering, detect
  near-identical matrices (shift/J within tolerance, same degeneracy vector) and
  force them into the same fold (or drop dups).
- **Stratify** fold assignment so degeneracy patterns and coupling regimes
  (e.g. strongly- vs weakly-coupled) are balanced across train/val/test — avoids
  all t-Bu-heavy or strongly-coupled systems landing in one split and skewing
  metrics.

**Later / harder upgrade (deferred):** embed the matrices → dimensionality
reduction → cluster, then hold out **entire clusters** as val/test. Forces the
model to generalize beyond primitive/local patterns to genuinely novel regions
of spin-system space. More involved to build and tune; introduce once the
baseline works.

## Open items / v2 upgrades
- Swap explicit expansion → composite reduction in `diff_renderer_torch.py` for
  scale.
- Set-prediction + Hungarian loss if canonical-ordering label noise plateaus.
- Transformer / hybrid encoder if long-range coupling structure is missed.
- Learned-uncertainty loss weighting if manual weights prove fiddly.
- Cluster-holdout split (embed → reduce → cluster → hold out whole clusters) as
  a harsher generalization test (Decision 8).
