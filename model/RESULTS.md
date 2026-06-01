# SpinHance Model Results

## Current status (2026-06-01)

Lineage of the `spingraph_decoder` production model:

| run | model | data | recipe | result | status |
|---|---|---|---|---|---|
| 022 | medium 10M | 64k | canonical matrix + surrogate-spectral | 0.064 / 0.91 / 0.916 / 0.928 | superseded |
| **025** | medium 10M | 64k | matrix, **shift wt 2×**, matrix-only, WSD LR | **0.037 / 0.59 / 0.94 / 0.945** | **production** |
| 026 | medium 10M | 64k | 025 + **peak channel** + **soft-equiv** | in progress (ep35: 0.066/1.00, soft-equiv flag 99% acc) | running |
| 027 | **xl 57M** | 500k PubChem | 025 recipe (scale-up baseline) | queued (capacity) | — |
| 028 | **xl 57M** | 500k PubChem | 026 recipe | queued (capacity) | — |

Metrics are `shift_mae_ppm / j_mae_hz / presence_f1 / deg_balanced_acc`.

**Two architecture ideas added in 026** (`a64f608`):
- **Peak-channel input** (`model.use_peak_channel`): a 2nd conv input channel — an in-model
  peak-emphasis map (local maxima > per-sample threshold, Gaussian-smoothed) derived from the
  spectrum — as a shift-localization prior. No data-pipeline change (no train/serve skew).
- **Soft-equivalence flag** (`PairwiseEdgeHead` 3rd logit + `SoftEquivLoss` + decode averaging):
  two groups with the same shift but different couplings (accidental degeneracy) are flagged
  per-edge (BCE) and their predicted shifts pulled together (consistency penalty) + hard-averaged
  at decode, so a degenerate pair renders as one peak, not a spurious split doublet. In 026 the
  flag head hits ~99% acc/recall on the ~8.5% of edges that are soft-equivalent.

**3M+ PubChem scaling** (`ff46b0a`): `SIZE_PRESETS['xl']` (~57M params at dim512/enc4/dec6);
`StackedSpectra` reads the stacked `part_NNNNN.npy` shards (1000 mols each, 90 MHz only) keyed by
record order to `spin_systems_pubchem.json.gz`; `data.parts` selects this path. **Lesson:** at
≥500k records, per-worker DataLoader copies of the record list (COW broken by refcounting) OOM a
16 GB box — use `num_workers≤2` (or a big-RAM node); the full 3.2M set needs ≥32 GB RAM regardless.

## ⭐⭐ PRODUCTION MODEL — `spingraph_decoder` (structured query decoder)

The IDEAS north-star architecture (Families C+G+L) **dramatically beats the dense-CNN
baseline** and is the new production model. session022 (`train_64k_spingraph_canonical.yaml`):

| metric | CNN baseline floor | **spingraph_decoder (022)** | improvement |
|--------|--------------------|-----------------------------|-------------|
| shift MAE (ppm) | 0.279 | **0.064** | **4.4×** |
| J MAE (Hz) | 1.80 | **0.91** | **2.0×** |
| presence F1 | 0.807 | **0.916** | +0.11 |
| deg acc | 0.987 | **0.996** | + |
| deg **balanced**-acc | 0.732 | **0.928** | rare-class problem solved |

(h_shift 0.062 ≈ shift 0.064 → not a permutation artifact. best ep71, early-stopped ep91.
checkpoint: `s3://spinhance-data/training/session022/runs/.../checkpoints/best.pt`.)

**Architecture:** ResNet1D conv stem → dim-projected global tokens + ppm positional
encoding → pre-LN Transformer encoder → 8 learned spin-group queries → Transformer
decoder → per-node heads (shift + degeneracy) + symmetric pairwise edge head
(`edge_ij = MLP([h_i+h_j, |h_i−h_j|])`). Returns the standard `ModelOutput`.

**What worked / what didn't (ablation findings):**
1. **The set-structured decoder is the big win** — 8 queries + symmetric edges suit the
   unordered spin-graph output far better than CNN+typed-heads. Even the canonical-only
   warmup phase hit shift 0.080 (session021).
2. **Hungarian set-matching loss HURT** (session021): for distinct-shift NMR data the
   canonical shift-sorted order is unambiguous and highly learnable; Hungarian's assignment
   freedom scrambled the relational coupling structure (J 1.35→2.1, never recovered → early
   stop). → use the **canonical `matrix` loss**, not Hungarian, for this architecture.
3. **The surrogate spectral loss (Branch 5/6) adds substantially on top** — once it ramped in
   (ep20), J MAE halved (1.35 → 0.91) and shift improved (0.080 → 0.064). The full stack
   composes: structured decoder + canonical anchor + surrogate spectral consistency.
4. **Support-region tokens (Family D/E/H) did NOT help** (session023): at matched epochs they
   were no better than global-only and ~2.5–2.8× slower (per-item region extraction
   bottlenecks the dataloader). The transformer's attention over the full spectrum already
   captures the region structure. → region tokens dropped; the `data.region_tokens` flag and
   `model/data/regions.py` remain (off by default) for future revisit.
5. **Integration-aware aux loss (Family H) unnecessary** — the architecture already solved the
   rare-class degeneracy imbalance (balanced-acc 0.73 → 0.93).

**Winning recipe:** `spingraph_decoder` (medium, ~10M params) + canonical `matrix` anchor
(weight 1.0) + early-ramp `surrogate_spectral` to 0.6 (ep20→30, hold) + WSD LR. Config
`train_64k_spingraph_canonical.yaml`.

---

# Reference Baseline (superseded — kept for the record)

Reference results for the **dense-CNN + attention-pool + typed-heads** architecture
(`resnet1d_attention_pool`, IDEAS Families A/B) with the Branch-5 differentiable
surrogate renderer and Branch-6 spectral-consistency loss (Family L). This is the
**floor every new architecture must beat**. All runs: 64k ChEMBL (`spin_systems_chembl.json`),
**90 MHz input only**, molecule+dedup split (`split: none`), seed 0, batch 256, bf16,
medium model (~4.96M params), on g6e.xlarge (L40S) unless noted.

Validation metrics (held-out), best epoch by `score = shift_mae_ppm + j_mae_hz/10`:

| Run | Loss / schedule | shift MAE (ppm) | J MAE (Hz) | presence F1 | deg acc | deg bal-acc | score |
|-----|-----------------|-----------------|------------|-------------|---------|-------------|-------|
| **session015** baseline | Stage-1 matrix only, 80 ep | 0.345 | 2.02 | 0.787 | 0.978 | ~0.53 | 0.547 |
| **session016** Run A | + spectral ramp 40→50 to 0.3, hold | 0.308 | 1.935 | 0.794 | 0.982 | — | 0.502 |
| **session017** Run B | + spectral trapezoid (0.3, decay 60→70) | 0.307 | 1.946 | 0.793 | 0.982 | — | 0.502 |
| **session018** Run C | Stage-2 spectral ONLY (no matrix anchor) | 1.89 | 11.9 | 0.443 | — | — | diverged |
| **session019** Run C2 | early-ramp: matrix 0–19, ramp→0.6 over 20–30, hold, 100 ep | 0.279 | 1.935 | 0.790 | 0.984 | 0.632 | 0.473 |
| **session020** Run C3 ⭐ | C2 + **WSD LR** (hold peak to ~ep63, floor 1.2e-4) | **0.279** | **1.80** | **0.807** | **0.987** | **0.732** | **0.459** |

## ⭐ Winning recipe (the floor to beat)

**session020** — `train_64k_surrogate_spectral_earlyramp_wsd.yaml`:
**shift 0.279 ppm · J 1.80 Hz · presence F1 0.807 · deg acc 0.987 (balanced 0.732)**.
- Loss: matrix anchor (weight 1.0, always on) + frozen surrogate spectral term
  (`surrogate_spectral`, W1 + 0.5·(1−cos)) early-ramped to 0.6 over epochs 20–30, held to 100.
- LR: WSD schedule — warmup 3%, hold peak 3e-4 through ~ep63, cosine-decay to a 1.2e-4
  floor (`lr_stable_frac 0.60`, `lr_min_factor 0.40`).

## Key findings

1. **Stage-1 + Stage-2 (spectral consistency) beats Stage-1 alone** — the frozen surrogate
   renderer used as a ramped spectral-consistency loss improves matrix accuracy
   (shift 0.345 → 0.279, ~19%). This is the validated payoff of Branches 5–6.
2. **Stage-2 alone diverges** (session018): with no matrix anchor the model finds
   spectrally-consistent but structurally-wrong matrices — the 90 MHz inverse problem is
   under-determined. The matrix anchor is essential; spectral loss is a *refinement*, not an
   *identification*, signal.
3. **A ≈ B (ramp-hold vs trapezoid) tie at weight 0.3**; the trapezoid showed the spectral
   gain *persists* after the term decays off (locked in by mid-training).
4. **Heavier + earlier spectral (0.6, ramp at ep20) + 100 epochs is better** (session019),
   mainly via shift MAE.
5. **LR schedule matters (session020 > 019):** keeping the LR high through the spectral-learning
   phase (WSD + raised floor) improved J MAE (1.94 → 1.80), F1 (0.790 → 0.807), and especially
   **rare-class degeneracy** (balanced-acc 0.632 → 0.732). Validated the "cosine decays too early"
   intuition.
6. **Loss split during the Stage-2 hold:** ~46% matrix / ~54% spectral (the 0.3/0.6 weight isn't
   "gentle" — the spectral raw magnitude is ~4× the matrix term). Within the spectral term the
   `cosine` component dominates W1 (cosine runs low ~0.2 on sparse 90 MHz spectra) — a candidate
   `cosine_weight` ablation for the future.

## Remaining bottlenecks (motivating the architecture rework)

Train-vs-val curves show **mild overfitting / information-limit, not capacity-starvation**
(train losses keep falling while val plateaus; bigger model not indicated). The ceilings:
- **S8 permutation symmetry** — 8! equivalent group orderings; near-equal-shift label swaps cap
  shift/F1. → motivates a set/graph output (query decoder + Hungarian).
- **Rare-class degeneracy** — balanced-acc ~0.73 vs raw 0.987; class imbalance + weak proton-count
  cues. → motivates integration-aware input metadata.
- **Low-field under-determination / peak overlap.** → motivates support-region tokenization.

Next: the IDEAS north-star structured spin-graph model (Families D+E+H+G+K+L) — see
`autoai/IDEAS.md` and the plan in `model-rebuild/spingraph-decoder`. Deferred 90-MHz-legal levers
(not yet tried): stronger realistic augmentation (linewidth-variability/noise — bridges sim→real),
focal loss / oversampling for degeneracy, `cosine_weight` sweep.

## Reproduce / artifacts

- Configs: `model/configs/train_64k.yaml` (baseline), `train_64k_surrogate_spectral.yaml` (A),
  `..._trapezoid.yaml` (B), `..._only.yaml` (C), `..._earlyramp.yaml` (C2), `..._earlyramp_wsd.yaml` (C3 ⭐).
- Surrogate renderer checkpoint (frozen Stage-2 teacher): `s3://spinhance-data/training/session012/.../checkpoints/best.pt`
  (test-set fidelity cos@90 0.986 / cos@600 0.990); local mirror `model_artifacts/surrogate/session012_best.pt`.
- Run artifacts (status/metrics/checkpoints): `s3://spinhance-data/training/session0{15..20}/runs/`.
- Launch: `model/scripts/launch_ec2.sh` (env `TRAIN_CONFIG`/`RUN_TAG`/`SESSION_OVERRIDE`/`SURROGATE_CKPT_S3`).
