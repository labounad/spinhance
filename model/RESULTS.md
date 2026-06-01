# SpinHance Model Results — Reference Baseline

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
