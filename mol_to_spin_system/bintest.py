"""mol_to_spin_system/bintest.py — visualize shift/coupling randomness.

Two "bin test" views of the augmentation (mol_to_spin_system.augment):

* ``mols``    — pick a few labelled molecules, sample each many times, and
                histogram every group's shift and every coupling's J.  Shows
                that the jitter is centered correctly and scaled by each
                environment's spread (shifts) / mechanism (couplings).
* ``dataset`` — aggregate over a whole per-count Task 2 dataset: the shift
                spread distribution, the resulting per-group sigma (vs
                floor/cap), coupling counts by type, and the global shift
                distribution (predicted means vs one augmented draw).

Usage
-----
::

    # per-molecule (needs the predictor + labelled XYZ input)
    python -m mol_to_spin_system.bintest mols generate/data/buckets/chembl_8spin.xyz.gz out.png [--n 2] [--draws 2000]

    # whole dataset (reads Task 2 JSON / JSON.gz; no predictor needed)
    python -m mol_to_spin_system.bintest dataset "mol_to_spin_system/data/buckets/spin_systems_chembl_*spin.json.gz" out.png [--cap 200000]
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter

import numpy as np

from mol_to_spin_system.augment import (
    COUPLING_SIGMA, DEFAULT_CAP, DEFAULT_FLOOR, sample_record, sample_shifts,
    shift_sigma,
)


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def bintest_mols(xyz_path: str, out_png: str, n: int = 2, draws: int = 2000) -> None:
    """Histogram per-group shifts and per-coupling J for *n* sampled molecules."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import RDLogger

    from mol_to_spin_system.xyz import entry_to_spin_system, iter_xyz_entries
    RDLogger.DisableLog("rdApp.*")

    recs = []
    for comment, atoms in iter_xyz_entries(xyz_path):
        try:
            r = entry_to_spin_system(comment, atoms).to_dict()
        except Exception:
            continue
        if r["couplings"]:
            recs.append(r)
        if len(recs) >= n:
            break

    fig, axes = plt.subplots(len(recs), 2, figsize=(15, 4.5 * len(recs)))
    axes = np.atleast_2d(axes)
    for row, rec in enumerate(recs):
        samp = [sample_record(rec, rng=np.random.default_rng(s)) for s in range(draws)]
        sg = np.array([[s["spin_groups"][k][0] for k in range(len(rec["labels"]))]
                       for s in samp])
        cj = np.array([[s["couplings"][k][2] for k in range(len(rec["couplings"]))]
                       for s in samp])
        ax = axes[row, 0]
        for k, lab in enumerate(rec["labels"]):
            lo, hi = rec["shift_range"][k]
            ax.hist(sg[:, k], bins=40, alpha=0.55,
                    label=f"{lab} μ={rec['spin_groups'][k][0]} [{lo},{hi}]")
        ax.set_title(f"{rec['chembl_id']}: shift distributions (n={draws})")
        ax.set_xlabel("δ (ppm)"); ax.invert_xaxis(); ax.legend(fontsize=6, ncol=2)
        ax = axes[row, 1]
        for k, (c, t) in enumerate(zip(rec["couplings"], rec["coupling_types"])):
            ax.hist(cj[:, k], bins=40, alpha=0.55, label=f"{c[0]}-{c[1]} {t} J={c[2]}")
        ax.set_title(f"{rec['chembl_id']}: coupling distributions (n={draws})")
        ax.set_xlabel("J (Hz)"); ax.legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()
    print(f"saved {out_png} ({len(recs)} molecules)")


def bintest_dataset(src_glob: str, out_png: str, cap: int = 200000) -> None:
    """Aggregate randomness view over a per-count Task 2 dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spreads, sigmas, means, sampled = [], [], [], []
    ctype: Counter = Counter()
    rng = np.random.default_rng(0); n = 0
    for f in sorted(glob.glob(src_glob)):
        for rec in json.loads(_open_text(f).read()):
            sr, sg = rec.get("shift_range"), rec["spin_groups"]
            if sr:
                for (m, _), (lo, hi) in zip(sg, sr):
                    spreads.append(hi - lo); sigmas.append(shift_sigma(lo, hi)); means.append(m)
                sampled.extend(sample_shifts([g[0] for g in sg], sr, rng=rng).tolist())
            for t in rec.get("coupling_types", []):
                ctype[t] += 1
            n += 1
            if n >= cap:
                break
        if n >= cap:
            break

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].hist(np.clip(spreads, 0, 8), bins=80); ax[0, 0].set_yscale("log")
    ax[0, 0].set_title(f"shift spread (max-min): {n:,} mols, {len(spreads):,} groups")
    ax[0, 0].set_xlabel("ppm")
    s = np.asarray(sigmas)
    ax[0, 1].hist(s, bins=60)
    ax[0, 1].axvline(DEFAULT_FLOOR, c="g", ls="--", label=f"floor {DEFAULT_FLOOR}")
    ax[0, 1].axvline(DEFAULT_CAP, c="r", ls="--", label=f"cap {DEFAULT_CAP}")
    ax[0, 1].set_title("resulting shift sigma (ppm)"); ax[0, 1].legend()
    ts = sorted(ctype)
    ax[1, 0].bar(ts, [ctype[t] for t in ts])
    for i, t in enumerate(ts):
        ax[1, 0].text(i, ctype[t], f"σ={COUPLING_SIGMA.get(t, '?')}", ha="center",
                      va="bottom", fontsize=8)
    ax[1, 0].set_title(f"couplings by type ({sum(ctype.values()):,})")
    ax[1, 0].tick_params(axis="x", rotation=20)
    ax[1, 1].hist(means, bins=120, alpha=0.6, density=True, label="predicted means")
    ax[1, 1].hist(sampled, bins=120, alpha=0.6, density=True, label="one augmented draw")
    ax[1, 1].set_title("global δ: means vs augmented"); ax[1, 1].set_xlabel("δ (ppm)")
    ax[1, 1].legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close()
    print(f"saved {out_png}")
    print(f"  records={n:,} groups={len(spreads):,} "
          f"spread median={np.median(spreads):.3f} | "
          f"sigma@floor={100*np.mean(s<=DEFAULT_FLOOR+1e-9):.0f}% "
          f"@cap={100*np.mean(s>=DEFAULT_CAP-1e-9):.0f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize shift/coupling augmentation randomness.")
    sub = p.add_subparsers(dest="mode", required=True)
    pm = sub.add_parser("mols", help="per-molecule histograms from a labelled XYZ")
    pm.add_argument("xyz"); pm.add_argument("out_png")
    pm.add_argument("--n", type=int, default=2); pm.add_argument("--draws", type=int, default=2000)
    pd = sub.add_parser("dataset", help="aggregate view over a Task 2 JSON dataset")
    pd.add_argument("src_glob"); pd.add_argument("out_png")
    pd.add_argument("--cap", type=int, default=200000)
    a = p.parse_args()
    if a.mode == "mols":
        bintest_mols(a.xyz, a.out_png, n=a.n, draws=a.draws)
    else:
        bintest_dataset(a.src_glob, a.out_png, cap=a.cap)


if __name__ == "__main__":
    main()
