"""
model.probes
=============
Fixed probe-set evaluator. Selects N molecules from the val set at run start
(stratified by spectral regime), runs inference every probe_every_epochs, and
saves per-molecule JSON + optional matrix-error PNG plots.

Artifacts are written to ``run_dir/probes/epoch_XXXX/``.  When ``run_dir``
is an ``s3://`` URI the files are uploaded directly to S3; otherwise they are
written to the local filesystem.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from model.metrics import decode, compute_metrics

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# ── Molecule selection ─────────────────────────────────────────────────────────

def _probe_indices(records: list[dict], n: int) -> list[int]:
    """Pick n molecules spread across aromatic/aliphatic × methyl/non-methyl bins."""
    if len(records) <= n:
        return list(range(len(records)))

    rng = np.random.default_rng(42)
    max_shifts = np.array([np.asarray(r["shifts"]).max() for r in records])
    max_degs   = np.array([np.asarray(r["degeneracy"]).max() for r in records])
    shift_med  = float(np.median(max_shifts))

    buckets: dict[tuple, list[int]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for i, (ms, md) in enumerate(zip(max_shifts, max_degs)):
        buckets[(int(ms > shift_med), int(md >= 3))].append(i)

    per = max(1, n // 4)
    chosen: list[int] = []
    for idxs in buckets.values():
        if idxs:
            k = min(per, len(idxs))
            chosen.extend(int(x) for x in rng.choice(idxs, k, replace=False))

    pool = [i for i in range(len(records)) if i not in set(chosen)]
    need = n - len(chosen)
    if need > 0 and pool:
        chosen.extend(int(x) for x in rng.choice(pool, min(need, len(pool)), replace=False))

    return chosen[:n]


# ── Matrix plot ────────────────────────────────────────────────────────────────

def _matrix_plot(true_mat: np.ndarray, pred_mat: np.ndarray,
                 path: Path, title: str = "") -> None:
    if not _MPL:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    err = np.abs(pred_mat - true_mat)
    vmax = max(float(np.abs(true_mat).max()), float(np.abs(pred_mat).max()), 1e-6)
    for ax, mat, label, cmap, lo in [
        (axes[0], true_mat, "True",      "RdBu_r", -vmax),
        (axes[1], pred_mat, "Predicted", "RdBu_r", -vmax),
        (axes[2], err,      "|Error|",   "hot_r",   0.0),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=lo,
                       vmax=vmax if cmap == "RdBu_r" else float(err.max()))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ── Evaluator ──────────────────────────────────────────────────────────────────

class ProbeEvaluator:
    """Run fixed probe molecules through the model and save diagnostic artifacts."""

    def __init__(
        self,
        val_records: list[dict],
        val_dataset,
        vocab,
        std,
        probe_count: int = 16,
        device: str = "cpu",
        run_dir: str | Path | None = None,
        save_plots: bool = True,
    ) -> None:
        self.vocab      = vocab
        self.std        = std
        self.device     = device
        self.run_dir    = str(run_dir) if run_dir is not None else None
        self.save_plots = save_plots and _MPL

        idxs = _probe_indices(val_records, probe_count)
        self.probe_records = [val_records[i] for i in idxs]
        self.mol_ids = [r["mol_id"] for r in self.probe_records]

        items = [val_dataset[i] for i in idxs]
        tensor_keys = ("spectrum", "spectrum_ref", "shifts", "j_mag",
                       "j_presence", "deg_class", "degeneracy")
        self.batch = {k: torch.stack([item[k] for item in items]) for k in tensor_keys}

    # ── Main entry ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, model, epoch: int, amp_ctx) -> dict[str, float]:
        if self.run_dir is None:
            return {}

        s3_mode      = self.run_dir.startswith("s3://")
        epoch_tag    = f"epoch_{epoch:04d}"
        epoch_prefix = f"{self.run_dir.rstrip('/')}/probes/{epoch_tag}"

        if not s3_mode:
            epoch_dir = Path(epoch_prefix)
            epoch_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        dev = {k: v.to(self.device) if torch.is_tensor(v) else v
               for k, v in self.batch.items()}
        with amp_ctx():
            pred = model(dev["spectrum"])

        pred_np = {k: pred[k].float().cpu().numpy() for k in pred}
        tgt_np  = {k: self.batch[k].numpy()
                   for k in ("shifts", "j_mag", "j_presence", "deg_class")}

        dec        = decode(pred_np, self.std, self.vocab)
        tgt_shifts = self.std.inverse_shifts(tgt_np["shifts"])
        tgt_pres   = tgt_np["j_presence"] > 0.5
        tgt_jmag   = self.std.inverse_j(tgt_np["j_mag"]) * tgt_pres

        G  = tgt_shifts.shape[1]
        iu = np.triu_indices(G, 1)

        per_mol: list[dict] = []
        for b, mol_id in enumerate(self.mol_ids):
            met = compute_metrics(
                {k: pred_np[k][b:b+1] for k in pred_np},
                {k: tgt_np[k][b:b+1]  for k in tgt_np},
                self.std, self.vocab,
            )
            true_jmat = np.zeros((G, G))
            true_jmat[iu[0], iu[1]] = tgt_jmag[b]
            true_jmat[iu[1], iu[0]] = tgt_jmag[b]
            true_mat = true_jmat.copy()
            np.fill_diagonal(true_mat, tgt_shifts[b])

            pred_jmat = dec["couplings"][b].copy()
            pred_mat  = pred_jmat.copy()
            np.fill_diagonal(pred_mat, dec["shifts"][b])

            per_mol.append({
                "mol_id": mol_id,
                "smiles": self.probe_records[b].get("smiles", ""),
                **met,
                "true_shifts":    tgt_shifts[b].tolist(),
                "pred_shifts":    dec["shifts"][b].tolist(),
                "true_deg":       self.probe_records[b]["degeneracy"].tolist(),
                "pred_deg":       dec["degeneracy"][b].tolist(),
                "true_couplings": true_jmat.tolist(),
                "pred_couplings": pred_jmat.tolist(),
            })

            if self.save_plots:
                title = (f"{mol_id}  shift_mae={met['shift_mae_ppm']:.3f}ppm  "
                         f"J_mae={met['j_mae_hz']:.2f}Hz")
                plot_name = f"matrix_{b:03d}_{mol_id}.png"
                if s3_mode:
                    from model import s3io
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
                    os.close(tmp_fd)
                    try:
                        _matrix_plot(true_mat, pred_mat, Path(tmp_path), title)
                        with open(tmp_path, "rb") as fh:
                            s3io.put_bytes(f"{epoch_prefix}/{plot_name}",
                                           fh.read(), "image/png")
                    finally:
                        os.unlink(tmp_path)
                else:
                    _matrix_plot(true_mat, pred_mat,
                                 Path(epoch_prefix) / plot_name, title)

        predictions_json = json.dumps(per_mol, indent=2)
        agg: dict[str, float] = {}
        for k in ("shift_mae_ppm", "j_mae_hz", "presence_f1", "deg_acc",
                  "h_shift_mae_ppm", "h_j_mae_hz"):
            vals = [m[k] for m in per_mol if k in m]
            if vals:
                agg[k] = float(np.mean(vals))
        probe_metrics_json = json.dumps(agg, indent=2)

        worst = sorted(per_mol, key=lambda m: m.get("shift_mae_ppm", 0.0), reverse=True)[:8]
        worst_json = json.dumps(worst, indent=2)

        if s3_mode:
            from model import s3io
            s3io.put_bytes(f"{epoch_prefix}/predictions.json",
                           predictions_json.encode())
            s3io.put_bytes(f"{epoch_prefix}/probe_metrics.json",
                           probe_metrics_json.encode())
            s3io.put_bytes(f"{epoch_prefix}/worst_cases.json",
                           worst_json.encode())
        else:
            (Path(epoch_prefix) / "predictions.json").write_text(predictions_json)
            (Path(epoch_prefix) / "probe_metrics.json").write_text(probe_metrics_json)
            (Path(epoch_prefix) / "worst_cases.json").write_text(worst_json)

        return agg
