"""
Verify the Task 4 pipeline against the real preliminary dataset (1072 molecules).

Run from repo root:
    PYTHONPATH=. python3 -m model.verify_real_data

Torch-free: exercises the adapter, splits, target encoding/standardizer, and the
numpy reference renderer vs the stored pyspin 90 MHz spectra. (Scaffold split
needs RDKit; here we use compute_scaffold=False — molecule + matrix-dedup.)
"""
from pathlib import Path

import numpy as np

from model.data_adapter import load_records, renderable_mask
from model.splits import make_splits
from model.targets import DegeneracyVocab, Standardizer, encode_target
from model import diff_renderer_ref as R

REPO = Path(__file__).resolve().parents[1]
JSON = REPO / "mol_to_matrix/data/spin_systems.json"
SPECTRA = REPO / "simulation/data/spectra"


def main():
    recs = load_records(JSON, SPECTRA, fields=(90, 600))
    print(f"loaded {len(recs)} records with spectra")

    # spectra sanity
    s0 = np.load(recs[0]["spec90_path"])
    ax = np.load(SPECTRA / "90MHz" / "ppm_axis.npy")
    dx = (ax[-1] - ax[0]) / (len(ax) - 1)
    print(f"spectrum shape {s0.shape}, dtype {s0.dtype}, integral {s0.sum()*dx:.4f}")

    rmask = renderable_mask(recs, max_spins=12)
    print(f"renderable (<=12 spins) for Stage-2: {sum(rmask)}/{len(recs)} "
          f"({100*sum(rmask)/len(recs):.0f}%)")

    # target encoding never KeyErrors (vocab covers all degeneracies)
    vocab = DegeneracyVocab()
    for r in recs:
        encode_target(r["shifts"], r["couplings"], r["degeneracy"], vocab)
    print("target encoding OK for all molecules (vocab covers every degeneracy)")

    # split + leakage checks
    assignment, report = make_splits(recs, ratios=(0.7, 0.2, 0.1), seed=0,
                                     compute_scaffold=False)
    print("split counts:", report["counts"], "ratios:",
          {k: round(v, 3) for k, v in report["ratios"].items()})
    print("leakage — scaffold:", report["scaffold_leaks"],
          "| near-dup matrix:", report["dup_matrix_leaks"],
          "| n_groups:", report["n_groups"])

    # standardizer on train fold
    train = [r for r in recs if assignment[r["mol_id"]] == "train"]
    std = Standardizer().fit(train, vocab)
    print(f"standardizer (train): shift {std.shift_mean:.2f}±{std.shift_std:.2f} ppm | "
          f"J {std.j_mean:.2f}±{std.j_std:.2f} Hz")

    # CONSISTENCY: numpy reference renderer vs stored pyspin 90 MHz spectra.
    # The oracle's dense broadening is memory-heavy at full res, so compare on a
    # coarser grid (downsample the stored 16384-pt spectrum to match) and on
    # smaller molecules. This validates the differentiable renderer's physics
    # against the actual training inputs.
    PTS = 1024
    MAX_SPINS = 8
    small = [r for r in recs if r["n_spins"] <= MAX_SPINS]
    npick = min(8, len(small))
    print(f"\nrenderer-vs-data consistency on {npick} small molecules "
          f"(n_spins<={MAX_SPINS}), compared at {PTS} pts:")
    coarse = np.linspace(0.0, 12.0, PTS)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(small), size=npick, replace=False)
    corrs = []
    for k in pick:
        r = small[k]
        ref_full = np.load(r["spec90_path"]).astype(float)
        ref = np.interp(coarse, ax, ref_full)            # downsample to PTS
        _, mine = R.simulate(r["shifts"], r["couplings"].tolist(),
                             r["degeneracy"].tolist(), 90.0, points=PTS,
                             ppm_from=0.0, ppm_to=12.0, linewidth_hz=1.0)
        c = float(np.corrcoef(ref, mine)[0, 1])
        corrs.append(c)
        print(f"  {r['mol_id']} n_spins={r['n_spins']:2d} couplings="
              f"{int((np.abs(np.triu(r['couplings'],1))>0).sum()):2d}  corr={c:.4f}")
    print(f"median corr vs stored spectra: {np.median(corrs):.4f}")

    print("\nREAL-DATA VERIFICATION COMPLETE")


if __name__ == "__main__":
    main()
