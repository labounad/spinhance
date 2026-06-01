"""
model.diagnostics.region_debug
==============================
Sanity-check the support-region tokenizer (IDEAS TaskSpec 3): overlay a spectrum
with its detected region spans + per-region integrals, and dump a JSON summary.
Use before a region-token training run to confirm the extractor finds sensible
spectral objects (and isn't over/under-segmenting).

    from model.diagnostics.region_debug import plot_support_regions
    plot_support_regions(spectrum, out_path="region_debug/mol_000000.png")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model.data.regions import N_SCALAR, extract_support_regions


def _regions_summary(feats, mask, ppm_to=12.0):
    rows = []
    for j in np.where(mask > 0)[0]:
        f = feats[j]
        rows.append({
            "center_ppm": round(float(f[0] * ppm_to), 3),
            "start_ppm": round(float(f[1] * ppm_to), 3),
            "end_ppm": round(float(f[2] * ppm_to), 3),
            "raw_integral": round(float(f[4]), 5),
            "rel_integral": round(float(f[5]), 4),
            "n_local_maxima": int(round(float(f[8]) * 20)),
        })
    return rows


def plot_support_regions(spectrum, ppm_from=0.0, ppm_to=12.0, out_path=None,
                         title=None, **region_kwargs):
    """Plot the spectrum with shaded detected regions + integral labels. Returns
    the region summary list. Saves a PNG if out_path is given."""
    spec = np.asarray(spectrum, dtype=np.float64)
    P = spec.shape[0]
    ppm = np.linspace(ppm_from, ppm_to, P)
    feats, mask = extract_support_regions(spec, ppm_from, ppm_to, **region_kwargs)
    summary = _regions_summary(feats, mask, ppm_to)

    if out_path is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(ppm, spec, color="#2563EB", lw=0.8)
        for r in summary:
            ax.axvspan(r["start_ppm"], r["end_ppm"], color="#F59E0B", alpha=0.18)
            ax.text(r["center_ppm"], spec.max() * 0.92,
                    f"{r['rel_integral']:.0%}", ha="center", fontsize=7, color="#B45309")
        ax.set_xlim(ppm_to, ppm_from)                       # NMR convention: high ppm left
        ax.set_xlabel("δ (ppm)"); ax.set_ylabel("intensity")
        ax.set_title(title or f"{len(summary)} support regions")
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110); plt.close(fig)
    return summary


def dump_region_summary(spectra, out_dir, ids=None, **region_kwargs):
    """Write per-spectrum region summaries (JSON) + plots for a small sample."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ids = ids or [f"mol_{i:06d}" for i in range(len(spectra))]
    allsum = {}
    for mid, spec in zip(ids, spectra):
        allsum[mid] = plot_support_regions(spec, out_path=out_dir / f"{mid}.png",
                                           title=mid, **region_kwargs)
    (out_dir / "region_summary.json").write_text(json.dumps(allsum, indent=2))
    return allsum
