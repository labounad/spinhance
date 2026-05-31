"""Verify torch-free training infrastructure: schedules + decode/metrics.

Run: PYTHONPATH=<repo root> python3 model/test_train_infra.py
"""
import numpy as np
from model import schedules as SCH
from model import metrics as M
from model import targets as T

rng = np.random.default_rng(3)
G = 8
NP = G * (G - 1) // 2


def test_schedules():
    # curriculum: matrix-only before stage1, full ramp after
    assert SCH.curriculum_weights(0, 20, 10) == (1.0, 0.0)
    assert SCH.curriculum_weights(19, 20, 10) == (1.0, 0.0)
    wm, ws = SCH.curriculum_weights(25, 20, 10, spectral_max=1.0, matrix_anchor=0.3)
    assert abs(ws - 0.5) < 1e-9 and abs(wm - 0.65) < 1e-9, (wm, ws)  # halfway
    wm, ws = SCH.curriculum_weights(40, 20, 10, spectral_max=1.0, matrix_anchor=0.3)
    assert abs(ws - 1.0) < 1e-9 and abs(wm - 0.3) < 1e-9             # plateau
    # lr: warmup linear, then cosine to min
    assert abs(SCH.lr_factor(0, 100, 1000)) < 1e-9
    assert abs(SCH.lr_factor(50, 100, 1000) - 0.5) < 1e-9
    assert abs(SCH.lr_factor(100, 100, 1000) - 1.0) < 1e-9
    assert SCH.lr_factor(1000, 100, 1000, min_factor=0.05) <= 0.05 + 1e-6
    print("schedules: curriculum + lr  OK")


def _make_target_batch(B=16):
    vocab = T.DegeneracyVocab()
    recs = []
    for _ in range(200):
        s = rng.uniform(0.5, 9, G)
        c = np.zeros((G, G))
        for i in range(G):
            for j in range(i + 1, G):
                if rng.random() < 0.4:
                    c[i, j] = c[j, i] = float(rng.uniform(1, 12))
        d = rng.choice(T.DEFAULT_DEG_VOCAB, size=G).astype(int)
        recs.append(dict(shifts=s, couplings=c, degeneracy=d))
    std = T.Standardizer().fit(recs, vocab)
    batch = [std.transform(T.encode_target(r["shifts"], r["couplings"],
             r["degeneracy"], vocab)) for r in recs[:B]]
    target = {k: np.stack([b[k] for b in batch]) for k in
              ("shifts", "j_mag", "j_presence", "deg_class")}
    return target, std, vocab


def test_metrics_perfect_and_perturbed():
    target, std, vocab = _make_target_batch()
    B = target["shifts"].shape[0]
    C = len(vocab)

    # PERFECT prediction: copy standardized targets; logits one-hot for presence/deg
    pred = {
        "shifts": target["shifts"].copy(),
        "j_mag": target["j_mag"].copy(),
        "j_presence": np.where(target["j_presence"] > 0.5, 20.0, -20.0),
        "deg_logits": np.full((B, G, C), -20.0),
    }
    for b in range(B):
        for g in range(G):
            pred["deg_logits"][b, g, target["deg_class"][b, g]] = 20.0

    met = M.compute_metrics(pred, target, std, vocab)
    assert met["shift_mae_ppm"] < 1e-4, met
    assert met["j_mae_hz"] < 1e-4, met
    assert met["presence_acc"] == 1.0 and met["presence_f1"] > 0.999, met
    assert met["deg_acc"] == 1.0, met
    print("metrics (perfect pred):", {k: round(v, 4) for k, v in met.items()})

    # PERTURBED: errors should grow, accuracies should drop
    pred2 = dict(pred)
    pred2["shifts"] = target["shifts"] + 0.5 / std.shift_std       # +0.5 ppm
    met2 = M.compute_metrics(pred2, target, std, vocab)
    assert abs(met2["shift_mae_ppm"] - 0.5) < 1e-3, met2
    print("metrics (+0.5 ppm shift):", {k: round(v, 4) for k, v in met2.items()})

    # wasserstein eval: identical spectra -> 0; shifted -> > 0
    a = np.zeros((1, 256)); a[0, 100] = 1.0
    b = np.zeros((1, 256)); b[0, 110] = 1.0
    assert M.wasserstein1_np(a, a)[0] < 1e-9
    assert M.wasserstein1_np(a, b)[0] > 0
    print("wasserstein1_np: identical=0, shifted>0  OK")


def test_metrics_hungarian_perfect_permuted():
    """With a permuted-but-otherwise-perfect prediction, Hungarian MAE == 0
    while canonical MAE > 0 (assuming permutation is non-identity)."""
    target, std, vocab = _make_target_batch(B=8)
    B, G = target["shifts"].shape
    C = len(vocab)

    # Permute ground truth by swapping first two groups
    perm = list(range(G))
    perm[0], perm[1] = perm[1], perm[0]

    pred_shifts_std = target["shifts"][:, perm]
    # Re-build j_mag for the permuted group order — pairs ij become perm[i]perm[j]
    iu     = np.triu_indices(G, 1)
    tgt_full = np.zeros((B, G, G))
    tgt_full[:, iu[0], iu[1]] = target["j_mag"]
    tgt_full[:, iu[1], iu[0]] = target["j_mag"]
    perm_arr = np.array(perm)
    perm_full = tgt_full[:, perm_arr][:, :, perm_arr]
    pred_jmag_std_perm = np.zeros((B, len(iu[0])))
    for b in range(B):
        pred_jmag_std_perm[b] = perm_full[b, iu[0], iu[1]]
    pred_pres_perm = np.where(pred_jmag_std_perm != 0, 20.0, -20.0)

    pred_deg_logits = np.full((B, G, C), -20.0)
    for b in range(B):
        for g in range(G):
            pred_deg_logits[b, g, target["deg_class"][b, perm[g]]] = 20.0

    pred = {
        "shifts":     pred_shifts_std,
        "j_mag":      pred_jmag_std_perm,
        "j_presence": pred_pres_perm,
        "deg_logits": pred_deg_logits,
    }
    met = M.compute_metrics(pred, target, std, vocab)

    # Hungarian matching should recover the permutation → near-zero error
    if "h_shift_mae_ppm" in met:
        assert met["h_shift_mae_ppm"] < 1e-3, (
            f"Hungarian shift MAE should be ~0 for permuted-perfect pred; got {met}")
        # And canonical MAE should be larger (groups are swapped so shifts differ)
        if target["shifts"][0, 0] != target["shifts"][0, 1]:
            assert met["shift_mae_ppm"] >= met["h_shift_mae_ppm"], (
                "Hungarian MAE must be ≤ canonical MAE")
    print("Hungarian metrics (permuted-perfect):", {k: round(v, 4) for k, v in met.items()
                                                     if k.startswith("h_")})


def test_metrics_hungarian_leq_canonical():
    """Hungarian MAE is always ≤ canonical MAE (optimal permutation beats or ties identity)."""
    target, std, vocab = _make_target_batch(B=32)
    B, G = target["shifts"].shape
    C = len(vocab)

    # Random predictions (noisy, order may be wrong)
    rng2 = np.random.default_rng(99)
    pred = {
        "shifts":     target["shifts"] + rng2.standard_normal(target["shifts"].shape) * 0.5,
        "j_mag":      target["j_mag"]  + rng2.standard_normal(target["j_mag"].shape) * 0.3,
        "j_presence": rng2.standard_normal(target["j_presence"].shape),
        "deg_logits": rng2.standard_normal((B, G, C)),
    }
    met = M.compute_metrics(pred, target, std, vocab)
    if "h_shift_mae_ppm" in met:
        assert met["h_shift_mae_ppm"] <= met["shift_mae_ppm"] + 1e-6, (
            f"h_shift_mae_ppm={met['h_shift_mae_ppm']:.4f} > shift_mae_ppm={met['shift_mae_ppm']:.4f}")
    print("Hungarian ≤ canonical: OK")


def test_metrics_hungarian_equals_canonical_when_ordered():
    """When pred is already in the canonical order, Hungarian MAE ≈ canonical MAE."""
    target, std, vocab = _make_target_batch(B=16)
    B, G = target["shifts"].shape
    C = len(vocab)

    # Small perturbation, identity permutation is optimal
    pred = {
        "shifts":     target["shifts"] + 0.01,
        "j_mag":      target["j_mag"],
        "j_presence": np.where(target["j_presence"] > 0.5, 20.0, -20.0),
        "deg_logits": np.full((B, G, C), -20.0),
    }
    for b in range(B):
        for g in range(G):
            pred["deg_logits"][b, g, target["deg_class"][b, g]] = 20.0

    met = M.compute_metrics(pred, target, std, vocab)
    if "h_shift_mae_ppm" in met:
        # They should be very close (identity is optimal for a tiny uniform shift)
        assert abs(met["h_shift_mae_ppm"] - met["shift_mae_ppm"]) < 0.05, met
    print("Hungarian ≈ canonical (ordered pred): OK")


if __name__ == "__main__":
    test_schedules()
    test_metrics_perfect_and_perturbed()
    test_metrics_hungarian_perfect_permuted()
    test_metrics_hungarian_leq_canonical()
    test_metrics_hungarian_equals_canonical_when_ordered()
    print("\nALL TRAIN-INFRA CHECKS PASSED")
