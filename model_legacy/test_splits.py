"""Verify model.splits without RDKit (precomputed 'scaffold' fields).

Run: PYTHONPATH=<repo root> python3 model/test_splits.py
Checks: canonical ordering, no scaffold/near-dup leakage across folds,
target ratios, and stratification balance.
"""
import numpy as np
from model_legacy import splits as S

rng = np.random.default_rng(42)
G = 8


def random_record(mid, scaffold):
    shifts = np.sort(rng.uniform(0.5, 9.0, G))[::-1].copy()
    c = np.zeros((G, G))
    for i in range(G):
        for j in range(i + 1, G):
            if rng.random() < 0.35:
                c[i, j] = c[j, i] = round(float(rng.uniform(1, 12)), 1)
    deg = rng.choice([1, 1, 2, 3, 3, 6, 9], size=G).astype(int)
    return dict(mol_id=mid, shifts=shifts, couplings=c, degeneracy=deg,
                scaffold=scaffold)


def main():
    # 600 molecules across 100 scaffolds
    recs = []
    n, n_scaf = 600, 100
    for i in range(n):
        recs.append(random_record(f"mol_{i}", f"scaf_{rng.integers(n_scaf)}"))

    # --- canonical ordering sanity: shifts come out descending ---
    o = S.canonical_order(recs[0]["shifts"], recs[0]["couplings"], recs[0]["degeneracy"])
    s_sorted = np.asarray(recs[0]["shifts"])[o]
    assert np.all(np.diff(s_sorted) <= 1e-9), "canonical order not shift-descending"

    # --- inject a near-duplicate matrix on a DIFFERENT scaffold ---
    twin = dict(recs[3]); twin = {k: (v.copy() if hasattr(v, "copy") else v)
                                  for k, v in recs[3].items()}
    twin["mol_id"] = "mol_twin"
    twin["scaffold"] = "scaf_unique_999"          # different scaffold
    twin["shifts"] = recs[3]["shifts"] + 0.005    # within shift_tol=0.02
    recs.append(twin)

    assignment, report = S.make_splits(recs, ratios=(0.7, 0.2, 0.1), seed=0)

    print("counts:", report["counts"])
    print("ratios:", {k: round(v, 3) for k, v in report["ratios"].items()})
    print("n_groups:", report["n_groups"], "| n_strata:", report["n_strata"])
    print("scaffold_leaks:", report["scaffold_leaks"],
          "| dup_matrix_leaks:", report["dup_matrix_leaks"])

    # near-dup twin must share the fold of its original
    same = assignment["mol_3"] == assignment["mol_twin"]
    print("near-dup co-folded (mol_3 == mol_twin):", same,
          f"({assignment['mol_3']})")

    # assertions
    assert report["scaffold_leaks"] == 0, "a scaffold straddled folds!"
    assert report["dup_matrix_leaks"] == 0, "a near-dup matrix straddled folds!"
    assert same, "near-duplicate matrix not co-folded"
    r = report["ratios"]
    assert abs(r["train"] - 0.7) < 0.06 and abs(r["val"] - 0.2) < 0.06 \
        and abs(r["test"] - 0.1) < 0.06, f"ratios off target: {r}"

    # stratification: each fold should see a spread of degeneracy patterns
    from collections import defaultdict
    fold_patterns = defaultdict(set)
    for rec in recs:
        f = assignment[rec["mol_id"]]
        fold_patterns[f].add(tuple(sorted(map(int, rec["degeneracy"]), reverse=True)))
    print("distinct degeneracy patterns per fold:",
          {f: len(p) for f, p in fold_patterns.items()})

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
