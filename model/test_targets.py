"""Verify model.targets (torch-free).

Run: PYTHONPATH=<repo root> python3 model/test_targets.py
"""
import numpy as np
from model import targets as T
from model.splits import canonical_order, reorder

rng = np.random.default_rng(7)
G = 8


def rand_mol():
    shifts = rng.uniform(0.5, 9.0, G)
    c = np.zeros((G, G))
    for i in range(G):
        for j in range(i + 1, G):
            if rng.random() < 0.4:
                c[i, j] = c[j, i] = round(float(rng.uniform(1, 12)), 2)
    deg = rng.choice(T.DEFAULT_DEG_VOCAB, size=G).astype(int)
    return dict(shifts=shifts, couplings=c, degeneracy=deg)


def main():
    vocab = T.DegeneracyVocab()

    # --- vocab round-trip ---
    deg = np.array([1, 3, 9, 2, 6, 1, 3, 4])
    idx = vocab.to_index(deg)
    assert np.array_equal(vocab.from_index(idx), deg), "vocab round-trip failed"

    # --- encode_target: canonical order, presence mask, shapes ---
    m = rand_mol()
    t = T.encode_target(m["shifts"], m["couplings"], m["degeneracy"], vocab)
    assert np.all(np.diff(t["shifts"]) <= 1e-6), "shifts not descending"
    assert t["j_mag"].shape == (G * (G - 1) // 2,)
    assert t["j_presence"].shape == (G * (G - 1) // 2,)
    # presence mask matches nonzero couplings under the same ordering
    s, c, d = reorder(m["shifts"], m["couplings"], m["degeneracy"], t["order"])
    iu = np.triu_indices(G, 1)
    assert np.array_equal(t["j_presence"] > 0, np.abs(c[iu]) > 1e-6), "presence mismatch"
    assert np.array_equal(vocab.from_index(t["deg_class"]), d), "deg_class mismatch"

    # --- Standardizer: fit on train, present-only J stats, inverse round-trip ---
    train = [rand_mol() for _ in range(300)]
    std = T.Standardizer().fit(train, vocab)
    # manual check of shift stats
    all_shifts = np.concatenate([T.encode_target(r["shifts"], r["couplings"],
                                 r["degeneracy"], vocab)["shifts"] for r in train])
    assert abs(std.shift_mean - all_shifts.mean()) < 1e-5
    tt = std.transform(t)
    # absent couplings must be exactly 0 after transform
    assert np.all(tt["j_mag"][t["j_presence"] == 0] == 0.0), "absent J not zeroed"
    # inverse of present entries recovers original magnitude
    present = t["j_presence"] > 0
    recov = std.inverse_j(tt["j_mag"][present])
    assert np.allclose(recov, t["j_mag"][present], atol=1e-4), "J inverse failed"
    recov_s = std.inverse_shifts(tt["shifts"])
    assert np.allclose(recov_s, t["shifts"], atol=1e-4), "shift inverse failed"

    # --- augmentation: preserves length and unit integral, stays non-negative ---
    P, ppm_to = 4096, 12.0
    spec = np.zeros(P); spec[[800, 1600, 2400]] = [1.0, 0.6, 0.3]
    dx = ppm_to / P
    spec = spec / (spec.sum() * dx)
    aug = T.augment_spectrum(spec, 0.0, ppm_to, rng=np.random.default_rng(1))
    assert aug.shape == (P,)
    assert aug.min() >= 0.0
    assert abs(aug.sum() * dx - 1.0) < 1e-5, "augmented spectrum not unit-integral"

    # --- bucket_key stable under input relabeling (permutation) ---
    perm = rng.permutation(G)
    m2 = dict(shifts=m["shifts"][perm],
              couplings=m["couplings"][np.ix_(perm, perm)],
              degeneracy=np.asarray(m["degeneracy"])[perm])
    assert T.bucket_key(**m) == T.bucket_key(**m2), "bucket_key not permutation-invariant"

    print("targets: vocab/encode/standardizer/augment/bucket_key  ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
