"""Verify torch-free training infrastructure: schedules + decode/metrics.

Run: PYTHONPATH=<repo root> python3 ml_model/test_train_infra.py
"""
import numpy as np
from ml_model import schedules as SCH
from ml_model import metrics as M
from ml_model import targets as T

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


if __name__ == "__main__":
    test_schedules()
    test_metrics_perfect_and_perturbed()
    print("\nALL TRAIN-INFRA CHECKS PASSED")
