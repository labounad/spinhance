# SpinHance AutoAI Ideas

This file guides the autonomous ML loop in `autoai/`. It should be treated as a ranked menu of model/loss/representation experiments. The goal is not to implement every idea in order blindly, but to explore increasingly structured architectures under a consistent evaluation protocol.

---

## Project Context

SpinHance learns an inverse map:

```text id="o12f81"
90 MHz 1H NMR spectrum → field-independent spin-system parameters
```

Input:

```text id="evndcz"
spectrum: 16384-point normalized 1D array over 0–12 ppm
```

Output:

```text id="k7uap3"
8 spin groups
8 chemical shifts
28 pairwise scalar couplings
8 proton degeneracies
```

Equivalent matrix representation:

```text id="wzimrn"
8×8 symmetric shift/J matrix
+
8-element degeneracy vector
=
8×9 target
```

Central difficulty:

```text id="p2j0kk"
The eight spin-group labels are arbitrary.
The same spin system has 8! equivalent matrix orderings.
```

Therefore, every serious loss/evaluation method should either:

1. use a canonical ordering only as a baseline convenience, or
2. use permutation-invariant matching, ideally Hungarian matching over predicted and true spin groups.

---

## Current Baseline Context

A simple benchmark model is being developed separately as Architecture Family A.

Baseline idea:

```text id="mzlk4x"
full dense spectrum
→ 1D CNN / ResNet encoder
→ global embedding
→ typed output heads
```

Typed heads:

```text id="ht55gq"
shifts: regression
J magnitudes: regression
J presence: binary classification
degeneracies: classification
```

This baseline is valuable and should remain the reference point for all more ambitious experiments.

Do not spend too many AutoAI cycles reimplementing the same baseline unless needed to establish a reproducible metrics floor.

---

## AutoAI Operating Principles

Each experiment should produce at least:

```text id="i9k0fw"
metrics.json
training log
checkpoint, if training succeeded
short summary of what was tried and what should be tried next
```

Metrics should include as many of the following as possible:

```text id="yq0ozx"
validation total loss
shift MAE in ppm
J MAE in Hz
coupling presence precision / recall / F1
degeneracy accuracy
Hungarian-matched shift MAE
Hungarian-matched J MAE
spectral reconstruction distance, if available
training time
parameter count
```

Prefer experiments that are directly comparable to prior runs.

Do not trust visual or prose-only improvements. Prefer verified metrics written to artifact files.

---

## Evaluation Priorities

The model should be evaluated in three layers.

### Layer 1: Matrix-level accuracy

Use direct supervised target comparison.

For canonical-output models, compare after canonical sorting.

For set/graph-output models, compare after Hungarian matching.

Important metrics:

```text id="w3cfst"
shift MAE
J MAE on true nonzero couplings
J presence F1
degeneracy accuracy
```

### Layer 2: Spectrum-level accuracy

Render the predicted spin system back to a spectrum and compare to the target spectrum.

Useful metrics:

```text id="zb9yh3"
Wasserstein-1 spectral distance
smoothed MSE
cosine similarity
peak-position error
```

### Layer 3: Chemical interpretability

When possible, log qualitative artifacts:

```text id="akkapy"
predicted vs true shift/J matrix
predicted vs true spectrum overlay
detected support-region plot
attention over spectral regions
failure-case examples
```

---

## Architecture Family A — Dense CNN / ResNet Baseline

### Approach

Use the full dense spectrum as a 1-channel 1D signal:

```text id="o06ked"
(B, 16384)
→ Conv1D / ResNet1D
→ pooled embedding
→ typed heads
```

Use typed heads rather than one flat vector:

```text id="qq2xhv"
shift_head
j_magnitude_head
j_presence_head
degeneracy_head
```

### Motivation

This is the simplest, most stable architecture and gives a necessary performance floor. It directly uses existing spectrum arrays without preprocessing or tokenization.

### Loss

Use canonical-order matrix loss initially:

```text id="wwolvr"
Huber shift loss
+ masked Huber J loss
+ BCE coupling-presence loss
+ degeneracy cross-entropy
```

### Success Criteria

This model is successful if it trains stably, produces nontrivial shift/J/degeneracy accuracy, and creates a reliable baseline for more structured architectures.

### Priority

Very high, but only until a reliable baseline is established.

---

## Architecture Family B — Dense CNN with Learned Attention Pooling

### Approach

Replace global average pooling with learned attention pooling over spectral positions.

```text id="ife4k7"
spectrum
→ CNN feature map over ppm positions
→ attention weights over positions
→ weighted pooled embedding
→ typed heads
```

Optional variant:

```text id="eqvou6"
multi-head attention pooling
```

where several pooling heads can attend to different spectral regions.

### Motivation

The input spectrum is sparse in the sense that most bins contain little useful signal. Global pooling may dilute the informative regions. Attention pooling lets the model focus on peaks/multiplets while still seeing the whole spectrum.

### Loss

Same as Architecture Family A.

### Success Criteria

Improve shift MAE, J F1, or degeneracy accuracy over the dense ResNet baseline without major instability.

Also save attention plots for several examples.

### Priority

High. Low implementation risk and likely useful.

---

## Architecture Family C — CNN Stem + Transformer Encoder over Spectral Tokens

### Approach

Preserve the full spectrum, but reduce it to a sequence of learned spectral tokens.

```text id="ff869r"
spectrum
→ CNN stem
→ downsampled feature sequence
→ positional encoding in ppm
→ transformer encoder
→ typed heads or query decoder
```

Example token length targets:

```text id="cf5c04"
L = 256
L = 512
L = 1024
```

### Motivation

Convolutions capture local multiplet shape, but scalar coupling can create nonlocal relationships between spectral regions. A transformer over spectral-position tokens can model long-range interactions.

### Loss

Start with canonical matrix loss. Later add Hungarian loss.

### Success Criteria

Beat the dense CNN baseline, especially on coupling presence and J magnitude.

### Priority

High. This is a natural out-of-the-box deep-learning upgrade.

---

## Architecture Family D — Support-Region Tokenization

### Approach

Preprocess each spectrum into contiguous support regions where the signal is meaningfully above baseline/noise.

A support region is an interval such as:

```text id="sks3pu"
aryl region
vinyl region
aliphatic multiplet region
isolated methyl doublet
overlapped multiplet cluster
```

Each support region becomes a token.

Token features should include:

```text id="gtfeti"
local normalized spectral window
start ppm
end ppm
center ppm
width ppm
raw integral
relative integral
max intensity
number of local maxima
local moments / skewness
optional region type embedding
```

Model:

```text id="o3ma7n"
spectrum
→ support-region extraction
→ local window encoder
→ region tokens
→ transformer
→ spin-system prediction
```

### Motivation

A human spectroscopist reads spectra as spectral objects, not as 16,384 unrelated bins. This representation may make the inverse problem better posed by explicitly separating molecule-specific regions of support.

### Important Constraints

Do not normalize away integration. Local shape can be normalized, but raw area and relative area must be preserved as metadata.

Do not assume one region equals one spin group. Multiple spin groups can overlap into one support region, and one spin group may contribute to complex features.

Always retain some global spectrum context or global metadata so that absence of signal in a region remains available to the model.

### Loss

Start with canonical matrix loss.

Then add Hungarian-matched graph loss if the output is set-like.

### Success Criteria

Improve degeneracy accuracy and shift MAE over dense CNN baseline.

Save debug artifacts showing detected support regions and integrals.

### Priority

Very high. This is one of the most chemically motivated directions.

---

## Architecture Family E — Global Context + Support-Region Tokens

### Approach

Use two branches:

```text id="hp9xck"
Branch A:
    full spectrum
    → low-resolution CNN
    → global context tokens

Branch B:
    spectrum
    → support-region extraction
    → local region tokens

Fusion:
    global context tokens + region tokens
    → transformer
    → spin-system predictor
```

### Motivation

Support-region tokenization is powerful but can miss weak shoulders, broad peaks, or information from empty regions. A global context branch prevents hard thresholding from becoming a bottleneck.

### Loss

Canonical or Hungarian graph loss.

Optional spectral reconstruction loss.

### Success Criteria

Beat pure support-region tokens or pure dense CNN on validation metrics, especially for overlapped spectra.

### Priority

Very high after Family D is working.

---

## Architecture Family F — Learned Region Proposal / Soft Spectral Tokens

### Approach

Instead of hard thresholding, learn K soft spectral windows.

Possible variants:

```text id="u96xvf"
top-K activation tokens
learned Gaussian windows
differentiable attention masks
multi-scale proposal windows
```

Model:

```text id="ba2t3a"
spectrum
→ CNN feature map
→ proposal head predicts K soft regions
→ region embeddings
→ transformer
→ spin graph
```

### Motivation

Hard support detection may fail for weak, overlapping, or strongly coupled features. Learned proposals can discover useful spectral regions without fixed threshold rules.

### Loss

Use matrix/graph loss.

Add diversity regularization to avoid all proposed windows collapsing onto the same intense region.

### Success Criteria

Improve over hard support-region tokenization, especially on crowded or low-SNR spectra.

### Priority

Medium-high. Higher implementation risk, high upside.

---

## Architecture Family G — Spin-Group Query Decoder

### Approach

Use a DETR-like decoder with eight learned spin-group queries.

```text id="wc6fy6"
spectral tokens
→ transformer encoder
→ 8 learned spin-group queries
→ cross-attention
→ 8 spin-group embeddings
```

Node heads:

```text id="sgo8y4"
chemical shift
degeneracy
```

Edge head:

```text id="svrcvr"
coupling presence
coupling magnitude
```

Pairwise edge decoder should be symmetric by construction:

```text id="r1iub7"
edge_ij = MLP([h_i + h_j, |h_i - h_j|, |delta_i - delta_j|])
```

### Motivation

The desired output is an unordered spin graph. Learned queries are a natural way to infer latent spin-group objects from spectral evidence tokens.

### Loss

Use Hungarian matching over spin groups.

Matching cost should include:

```text id="d9h531"
shift error
degeneracy mismatch
weak coupling-row profile cost
```

After matching, compute:

```text id="i0cxv0"
shift Huber
degeneracy cross-entropy
coupling presence BCE
masked J Huber
```

### Success Criteria

Reduce sensitivity to canonical-order label noise and improve validation metrics, especially when shifts are close or nearly tied.

### Priority

High. This is probably the cleanest long-term model head.

---

## Architecture Family H — Integration-Aware Models

### Approach

Use local integration as explicit input metadata and/or auxiliary supervision.

For support-region tokens, include:

```text id="m8bte1"
raw integral
relative integral
integral rank
estimated proton count
integration uncertainty
```

Potential auxiliary objective:

```text id="oau53u"
predicted degeneracy distribution should be consistent with attended region integrals
```

### Motivation

Integration is one of the strongest NMR cues for proton degeneracy. It should help distinguish CH, CH2, CH3, t-Bu-like groups, etc., especially when regions are isolated.

### Caution

Integration is unreliable under overlap, baseline errors, and normalization artifacts. Use it as soft evidence, not a hard rule.

### Loss

Add a weak integration-consistency loss only after a base model trains stably.

### Success Criteria

Improve degeneracy accuracy without harming shift/J metrics.

### Priority

Very high. Low cost and chemically meaningful.

---

## Architecture Family I — Soft Peak-Shape / Multiplicity Features

### Approach

For each support region, compute soft template-match features.

Candidate pattern scores:

```text id="xq5wpe"
singlet score
doublet score
triplet score
quartet score
doublet-of-doublets score
AB pattern score
AA'XX' pattern score
complex/overlap score
```

Concatenate these scores to region-token metadata.

### Motivation

Peak shape contains information about coupling structure. A local doublet-like pattern suggests one dominant coupling; a triplet-like pattern suggests two similar couplings; complex second-order patterns suggest strong coupling or overlapping spin systems.

### Caution

At 90 MHz, many spectra are non-first-order. Hard symbolic multiplicity assignments can be wrong. Therefore, these should be soft features, not deterministic rules.

### Loss

Same as the support-region transformer or query decoder.

### Success Criteria

Improve J presence and J magnitude metrics, especially for simpler first-order-like regions, without degrading strongly coupled cases.

### Priority

Medium. Useful but potentially brittle.

---

## Architecture Family J — Multi-Task Auxiliary Prediction

### Approach

Train the encoder to predict additional chemically relevant quantities alongside the main spin-system target.

Auxiliary tasks:

```text id="a9yxiz"
number of support regions
total proton count
degeneracy histogram
coupling density
strong-coupling flag
aromatic/vinyl/aliphatic region occupancy
spectrum complexity score
```

### Motivation

Auxiliary tasks can force the encoder to learn chemically meaningful intermediate structure and may improve data efficiency.

### Loss

Main loss plus small auxiliary weights.

### Success Criteria

Improve main validation metrics or training stability.

### Priority

Medium. Useful after baseline is stable.

---

## Architecture Family K — Permutation-Invariant Set / Graph Loss for Existing Heads

### Approach

Keep the current output shape but change the loss.

Current-style outputs:

```text id="bafawj"
8 shifts
28 J values
28 J presence logits
8 degeneracy predictions
```

Instead of comparing in canonical order, compute the best assignment between predicted and true spin groups with Hungarian matching.

Then permute the target or prediction before computing losses.

### Motivation

This is a minimal way to address the central S8 symmetry problem without rewriting the whole architecture.

### Loss

Hungarian-matched version of:

```text id="c4s7t7"
shift Huber
J Huber
J presence BCE
degeneracy CE
```

### Success Criteria

Improve validation loss and reduce failures for near-equal shifts.

### Priority

Very high once the canonical baseline plateaus.

---

## Architecture Family L — Spectral Consistency Loss / Differentiable Renderer

### Approach

After a model has learned a reasonable matrix prediction, add a spectral reconstruction loss.

```text id="r8ph6x"
predicted spin system
→ differentiable simulator
→ predicted 90 MHz spectrum
→ compare with input spectrum
```

Loss options:

```text id="wg1rax"
Wasserstein-1
smoothed MSE
cosine distance
multi-resolution spectral loss
```

### Motivation

The final prediction should reproduce the observed spectrum. Spectral loss is automatically permutation-invariant and directly tied to the physical goal.

### Caution

Do not train spectral-only from scratch. The inverse problem is not fully identifiable from low-field spectra alone. Keep a supervised matrix/graph anchor.

### Training Schedule

Use staged training:

```text id="jngwa3"
Stage 1:
    supervised matrix/graph loss only

Stage 2:
    ramp in spectral loss gradually
    keep a decayed supervised anchor
```

### Success Criteria

Improve spectral reconstruction metrics without degrading shift/J/degeneracy accuracy.

### Priority

High after a stable supervised model exists.

---

## Architecture Family M — Neural Prediction + Local Physics Refinement

### Approach

Use the neural model to initialize the spin system, then run local differentiable optimization on shifts and J values to better match the spectrum.

```text id="u8ty0r"
spectrum
→ neural spin-system prediction
→ differentiable spectral fitting
→ refined spin system
```

### Motivation

Neural models may get close but have small shift/J errors. Physics refinement can polish the final prediction and improve spectrum overlays.

### Caution

Refinement can move toward spectrum-equivalent but chemically wrong solutions. Keep constraints and do not allow arbitrary large changes.

### Success Criteria

Improve spectral distance and modestly improve shift/J metrics on validation examples.

### Priority

Medium-high. Good for final polishing, not first-pass training.

---

## Architecture Family N — Modern Out-of-the-Box Sequence Models

### Approach

Try strong generic 1D sequence models without heavy chemical customization.

Candidates:

```text id="kbuh3l"
1D ConvNeXt
TCN / WaveNet-style dilated CNN
Perceiver IO
Transformer encoder with patch embeddings
Mamba / state-space sequence model
Hyena-style long-convolution model
```

### Motivation

The project should explore whether generic sequence architectures outperform custom chemical models on this dataset.

### Loss

Start with canonical matrix loss.

Then test Hungarian graph loss where practical.

### Success Criteria

Beat the ResNet baseline at similar or reasonable parameter count and training time.

### Priority

Medium. Good AutoAI exploration target.

---

## Architecture Family O — Multi-Resolution Spectrum Encoder

### Approach

Represent the spectrum at multiple resolutions.

```text id="koxuxo"
full resolution local windows
medium-resolution downsampled spectrum
coarse global spectrum
```

Fuse these streams through concatenation, attention, or a feature pyramid.

### Motivation

NMR spectra contain both sharp local features and broad global region information. Multi-resolution encoding can capture local multiplet structure while preserving coarse chemical-region occupancy.

### Loss

Any supervised matrix/graph loss.

### Success Criteria

Improve shift and J metrics relative to single-resolution CNN.

### Priority

High. Particularly compatible with support-region tokens.

---

## Architecture Family P — Mixture-of-Experts by Spectral Regime

### Approach

Train a router that classifies spectra into regimes, then dispatches to specialized experts.

Possible regimes:

```text id="ze1bkk"
mostly first-order
strongly coupled
crowded/overlapped
aromatic-rich
aliphatic-rich
high degeneracy / methyl-rich
```

### Motivation

Different spectral regimes may benefit from different inductive biases.

### Caution

This is more complex and should only be tried after simpler models establish failure clusters.

### Success Criteria

Improve aggregate metrics and especially improve known failure-mode subsets.

### Priority

Low-medium. Later-stage idea.

---

## Architecture Family Q — Graph-Structured Latent Model

### Approach

Directly parameterize the output as a graph.

```text id="tu2cn3"
spectrum encoder
→ latent node embeddings
→ node attributes
→ edge attributes
```

Use message passing among latent spin-group nodes before final prediction.

```text id="eoxx8u"
8 latent nodes
→ graph neural network layers
→ node/edge heads
```

### Motivation

The predicted quantities are inherently graph-structured: couplings are edges between spin groups.

### Loss

Hungarian graph loss.

### Success Criteria

Improve coupling prediction over independent edge heads.

### Priority

Medium-high, especially after query decoder works.

---

## Architecture Family R — Region-to-Spin Bipartite Attention Model

### Approach

Represent spectral support regions and latent spin groups as two sets of nodes.

```text id="teu1ml"
region tokens ↔ spin-group queries
```

Use bipartite cross-attention:

```text id="u3pu42"
regions provide evidence for spin groups
spin groups explain regions
```

The model can output attention maps showing which regions support which spin groups.

### Motivation

A spectral region is not necessarily a spin group. This model explicitly allows many-to-one and one-to-many relationships between spectral evidence and spin-system objects.

### Loss

Hungarian graph loss plus optional attention/integration consistency.

### Success Criteria

Improve performance on overlapped spectra and produce interpretable attention maps.

### Priority

High as an ambitious custom architecture.

---

## Architecture Family S — Data Augmentation and Domain Robustness

### Approach

Apply on-the-fly spectrum augmentations.

Candidate augmentations:

```text id="yje802"
additive noise
baseline drift
small ppm referencing shift
linewidth variation
phase-like asymmetry
intensity scaling
small local broadening changes
solvent/residual peak masks
random low-intensity artifacts
```

### Motivation

Synthetic spectra may be cleaner than real spectra. Robust augmentation should improve generalization and prevent the model from exploiting simulator artifacts.

### Caution

Do not over-augment before the baseline is understood. Track whether each augmentation improves or hurts validation metrics.

### Success Criteria

Improve validation/test robustness, especially under augmented validation conditions.

### Priority

High.

---

## Architecture Family T — Curriculum and Training Strategy Search

### Approach

Search training hyperparameters and curricula.

Variables:

```text id="pxq86j"
learning rate
weight decay
warmup fraction
batch size
dropout
EMA decay
loss weights
presence positive class weight
degeneracy class weighting
stage1/stage2 split
spectral loss ramp length
gradient clipping
```

### Motivation

The architecture may not be the bottleneck until the loss and schedule are tuned.

### Success Criteria

Improve the current best architecture without changing the model class.

### Priority

Very high once one architecture is stable.

---

## Recommended AutoAI Triage Order

The agent should generally prioritize:

```text id="l3a9n1"
1. Verify baseline metrics for Architecture Family A.
2. Add attention pooling to the dense CNN.
3. Add multi-resolution CNN features.
4. Add Hungarian-matched loss to the existing head.
5. Build support-region tokenization and a debug artifact.
6. Train support-region transformer.
7. Add integration metadata.
8. Add global context + region token fusion.
9. Add spin-group query decoder.
10. Add spectral consistency loss.
11. Add soft peak-shape features.
12. Explore OOTB architectures such as ConvNeXt1D, TCN, Perceiver, Mamba/SSM.
13. Try local physics refinement.
```

Do not jump to highly complex models before establishing reliable metrics for the simple baseline.

---

## Suggested Initial TaskSpecs

### TaskSpec 1 — Attention Pooling Baseline Upgrade

Objective:

```text id="euwpmt"
Implement a ResNet1D variant that replaces AdaptiveAvgPool1d with learned multi-head attention pooling over spectral positions.
```

Architecture:

```text id="iln3pj"
Conv/ResNet encoder preserving final sequence dimension
multi-head attention pooling with 4 or 8 learned pooling heads
concatenate or average pooled heads
existing typed heads for shifts, J magnitude, J presence, degeneracy
```

Loss:

```text id="jlspj2"
canonical matrix loss:
shift Huber
masked J Huber
J presence BCE
degeneracy CE
```

Output artifacts:

```text id="v9nfky"
autoai/runs/<run>/training.py
autoai/runs/<run>/metrics.json
autoai/runs/<run>/summary.md
autoai/runs/<run>/checkpoint.pt, if successful
```

Success criteria:

```text id="rltg8v"
Runs end-to-end and improves at least one major validation metric over the dense ResNet baseline without catastrophic degradation of the others.
```

---

### TaskSpec 2 — Hungarian Loss for Existing Model Head

Objective:

```text id="opk2ew"
Implement a permutation-invariant Hungarian-matched matrix loss for the existing typed output format.
```

Architecture:

```text id="peoqyq"
Use the current ResNet/typed-head model.
Change only the loss/evaluation matching.
```

Loss:

```text id="t7dzr5"
Compute optimal assignment between predicted and true spin groups using shift and degeneracy matching cost, optionally with weak coupling-row cost.
Apply matched shift, degeneracy, J magnitude, and J presence losses.
```

Success criteria:

```text id="upgvmy"
Improves validation stability and/or metrics relative to canonical loss, especially for examples with close chemical shifts.
```

---

### TaskSpec 3 — Support-Region Token Dataset and Debugger

Objective:

```text id="q1n8an"
Implement support-region extraction for 16384-point spectra and save a debug artifact visualizing detected regions and integrals.
```

Architecture:

```text id="c6hxc5"
No full model required in first pass.
Create preprocessing utility and dataset wrapper.
```

Required features:

```text id="mzby0z"
adaptive threshold
region merging
margin expansion
local window extraction
ppm bounds
center/width
raw integral
relative integral
max intensity
region mask for batching
```

Success criteria:

```text id="p55r6m"
Produces stable region tokens for a sample batch and writes plots/JSON summaries showing region boundaries and metadata.
```

---

### TaskSpec 4 — Support-Region Transformer

Objective:

```text id="s9z5q5"
Train a transformer model over support-region tokens to predict the spin-system target.
```

Architecture:

```text id="xcthun"
local CNN encodes each region window
metadata MLP encodes region metadata
sum/concatenate shape + metadata embeddings
transformer encoder over region tokens
pooled representation or spin-query decoder
typed output heads
```

Loss:

```text id="j6q1qm"
Start with canonical matrix loss.
If stable, add Hungarian matching.
```

Success criteria:

```text id="wqk6p9"
Improves shift MAE or degeneracy accuracy over dense baseline and logs support-region attention or token importance.
```

---

### TaskSpec 5 — Global Context + Region Tokens

Objective:

```text id="mfnjfl"
Combine dense global spectral context with support-region tokens.
```

Architecture:

```text id="usmx2p"
global CNN branch over full spectrum
region-token branch over detected support regions
fusion transformer
typed heads or spin-query decoder
```

Loss:

```text id="xnib83"
Hungarian graph loss preferred; canonical matrix loss acceptable for first implementation.
```

Success criteria:

```text id="kfz01b"
Outperforms both pure dense CNN and pure region-token model on validation metrics.
```

---

### TaskSpec 6 — Spin-Group Query Decoder

Objective:

```text id="h4rasz"
Implement an 8-query spin-group decoder that predicts unordered spin-group nodes and pairwise coupling edges.
```

Architecture:

```text id="iwq2rm"
spectral encoder produces tokens
8 learned spin-group queries cross-attend to spectral tokens
node heads predict shift and degeneracy
symmetric pairwise edge head predicts J presence and J magnitude
```

Loss:

```text id="mz2810"
Hungarian graph loss over spin-group nodes plus matched edge losses.
```

Success criteria:

```text id="ogf5gc"
Improves performance over canonical-output models and reduces label-ordering failure cases.
```

---

### TaskSpec 7 — Integration Metadata Ablation

Objective:

```text id="a1dypu"
Test whether local integration metadata improves degeneracy prediction.
```

Architecture:

```text id="jzoqkd"
Use the best current support-region-token model.
Run with and without integral/area metadata.
```

Loss:

```text id="o6dmo1"
Same as support-region model.
Optionally add weak degeneracy/integration consistency loss.
```

Success criteria:

```text id="etjnxr"
Improves degeneracy accuracy without degrading shift/J metrics.
```

---

### TaskSpec 8 — Soft Multiplicity Features

Objective:

```text id="ix300z"
Add soft local peak-shape descriptors to support-region tokens.
```

Approach:

```text id="fkw7pb"
For each region, compute template-fit or heuristic scores for singlet, doublet, triplet, quartet, dd, AB-like, complex/overlap.
Concatenate these scores to token metadata.
```

Loss:

```text id="t2xvww"
Same as support-region transformer.
```

Success criteria:

```text id="ryo7af"
Improves J presence F1 or J magnitude MAE.
```

---

### TaskSpec 9 — Spectral Consistency Fine-Tuning

Objective:

```text id="b6hwtd"
Fine-tune the best supervised model with a differentiable spectral reconstruction loss.
```

Architecture:

```text id="xrwdeu"
Use best checkpoint from previous experiments.
```

Loss:

```text id="epw35y"
supervised matrix/graph loss
+
ramped spectral Wasserstein loss
+
optional smoothed MSE lineshape term
```

Success criteria:

```text id="d71jjb"
Improves spectral reconstruction distance without worsening shift/J/degeneracy metrics.
```

---

### TaskSpec 10 — OOTB Sequence Model Sweep

Objective:

```text id="lsvplq"
Compare several strong generic 1D sequence architectures under the same loss/eval protocol.
```

Candidates:

```text id="x87ho4"
TCN / dilated CNN
ConvNeXt1D
patch transformer
Perceiver IO
state-space / Mamba-like model, if dependencies permit
```

Loss:

```text id="irnfrc"
canonical matrix loss first
Hungarian loss if easy to reuse
```

Success criteria:

```text id="y0195b"
Identify any OOTB architecture that beats the ResNet baseline at acceptable compute cost.
```

---

## Failure Analysis Instructions

After every run, classify failures if possible:

```text id="mkw4w5"
large shift error
wrong degeneracy
false positive couplings
false negative couplings
bad J magnitude
near-equal shift label swap
overlapped-region failure
strong-coupling failure
methyl/t-Bu degeneracy failure
aromatic-region failure
```

The next experiment should target the most common failure mode.

---

## Stopping Criteria

Stop or pause the loop if:

```text id="c1be9k"
metrics plateau across several architectures
all high-priority ideas have been tried
training runs are failing for infrastructure reasons
best model is good enough for hackathon demo
budget limit is reached
```

The final summary should report:

```text id="gk5nqv"
best architecture
best metrics
what failed
what should be tried next
recommended model for demo
recommended model for long-term development
```

---

## Current Best Hypothesis

The most promising long-term architecture is:

```text id="plh6nx"
global dense spectral context
+
support-region tokens with local shape and integration metadata
→ transformer encoder
→ 8 spin-group query decoder
→ symmetric node/edge spin graph heads
→ Hungarian graph loss
→ optional spectral reconstruction fine-tuning
```

The most practical near-term path is:

```text id="h4xpkd"
ResNet baseline
→ attention pooling
→ Hungarian loss
→ support-region tokens
→ integration metadata
→ query decoder
```
