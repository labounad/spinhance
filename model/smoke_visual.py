"""
model/smoke_visual.py
=====================
Graphical smoke test for the model training pipeline.

Runs 6 epochs of training on 96 synthetic molecules (CPU, no S3), reads
the resulting diagnostics, and renders a summary figure that is saved to
disk and opened in the system image viewer.

Usage:
    PYTHONPATH=. python model/smoke_visual.py
    PYTHONPATH=. python model/smoke_visual.py --out my_smoke.png
    PYTHONPATH=. python model/smoke_visual.py --no-open   # save only
"""
from __future__ import annotations

# Ensure the repo root is on sys.path so "from model.X import …" works whether
# this script is run as "python model/smoke_visual.py" or "-m model.smoke_visual".
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Force Agg before any pyplot import.  probes.py also calls matplotlib.use("Agg")
# at module level; keeping the same backend here avoids backend-switch warnings.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# ── Synthetic training run ─────────────────────────────────────────────────────

def _run(run_dir: Path) -> None:
    from model.splits import make_splits
    from model.train import TrainConfig, fit

    rng = np.random.default_rng(0)
    G, P = 8, 2048
    recs = []
    for i in range(96):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 10))
        recs.append(dict(
            mol_id=f"m{i}",
            shifts=rng.uniform(0.5, 9, G),
            couplings=c,
            degeneracy=rng.choice([1, 2, 3], size=G).astype(int),
            scaffold=f"s{i % 12}",
            spec90=rng.random(P).astype(np.float32),
        ))
    assignment, _ = make_splits(recs, seed=0)
    cfg = TrainConfig(
        points=P,
        batch_size=8,
        epochs=6,
        stage1_epochs=4,
        ramp_epochs=2,
        warmup_frac=0.1,
        device="cpu",
        amp_dtype="none",
        patience=10,
        ckpt_path="",
        run_dir=str(run_dir),
        log_every_steps=2,
        probe_every_epochs=2,
        probe_count=8,
        save_probe_plots=True,
        save_failure_tables=True,
    )
    fit(recs, assignment, cfg)


# ── Diagnostics readers ────────────────────────────────────────────────────────

def _jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


# ── Figure ─────────────────────────────────────────────────────────────────────

_BG     = "#0d0d14"
_PANEL  = "#15151f"
_GRID   = "#1e1e30"
_TEXT   = "#d4d4e8"
_BLUE   = "#60a5fa"
_GREEN  = "#4ade80"
_AMBER  = "#fbbf24"
_PURPLE = "#c084fc"
_PINK   = "#f472b6"
_RED    = "#f87171"


def _style_ax(ax):
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    ax.grid(True, color=_GRID, linewidth=0.6, zorder=0)


def _render(run_dir: Path, save_path: Path) -> bool:
    rows    = _jsonl(run_dir / "metrics.jsonl")
    summary = _json(run_dir / "summary.json", {})
    status  = _json(run_dir / "status.json",  {})

    val_rows   = [r for r in rows if r.get("split") == "val"]
    train_rows = [r for r in rows if r.get("split") == "train_step"]

    probes_dir  = run_dir / "probes"
    probe_dirs  = sorted(probes_dir.iterdir()) if probes_dir.exists() else []
    latest_ep   = probe_dirs[-1] if probe_dirs else None

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.48, wspace=0.32,
                            left=0.06, right=0.97, top=0.90, bottom=0.07)

    def ax(r, c, colspan=1):
        a = fig.add_subplot(gs[r, c:c+colspan])
        _style_ax(a)
        return a

    # ── (0,0) Train-step loss ─────────────────────────────────────────────────
    a00 = ax(0, 0)
    if train_rows:
        xs = [r.get("step", i) for i, r in enumerate(train_rows)]
        ys = [r.get("metrics", {}).get("loss_total", np.nan) for r in train_rows]
        a00.plot(xs, ys, color=_BLUE, linewidth=1.6, zorder=3)
        a00.fill_between(xs, ys, alpha=0.12, color=_BLUE, zorder=2)
    a00.set_title("Train-step loss", fontsize=9, fontweight="bold")
    a00.set_xlabel("step", fontsize=8)
    a00.set_ylabel("loss", fontsize=8)

    # ── (0,1) Val shift + J MAE ───────────────────────────────────────────────
    a01 = ax(0, 1)
    if val_rows:
        ep  = [r.get("epoch", i) for i, r in enumerate(val_rows)]
        s   = [r.get("metrics", {}).get("shift_mae_ppm", np.nan) for r in val_rows]
        j   = [r.get("metrics", {}).get("j_mae_hz",      np.nan) for r in val_rows]
        a01.plot(ep, s, color=_GREEN, linewidth=1.6, label="shift MAE (ppm)", zorder=3)
        a01.plot(ep, j, color=_AMBER, linewidth=1.6, label="J MAE (Hz)",
                 linestyle="--", zorder=3)
        a01.legend(fontsize=7.5, facecolor=_PANEL, labelcolor=_TEXT, framealpha=0.8,
                   edgecolor=_GRID)
    a01.set_title("Validation — shift & J", fontsize=9, fontweight="bold")
    a01.set_xlabel("epoch", fontsize=8)

    # ── (0,2) Presence F1 + deg acc ──────────────────────────────────────────
    a02 = ax(0, 2)
    if val_rows:
        f1   = [r.get("metrics", {}).get("presence_f1", np.nan) for r in val_rows]
        dacc = [r.get("metrics", {}).get("deg_acc",     np.nan) for r in val_rows]
        a02.plot(ep, f1,   color=_PURPLE, linewidth=1.6, label="presence F1", zorder=3)
        a02.plot(ep, dacc, color=_PINK,   linewidth=1.6, label="deg acc",
                 linestyle="--", zorder=3)
        a02.set_ylim(-0.05, 1.1)
        a02.legend(fontsize=7.5, facecolor=_PANEL, labelcolor=_TEXT, framealpha=0.8,
                   edgecolor=_GRID)
    a02.set_title("Validation — coupling & degeneracy", fontsize=9, fontweight="bold")
    a02.set_xlabel("epoch", fontsize=8)

    # ── (1,0) Probe matrix PNG ────────────────────────────────────────────────
    a10 = ax(1, 0)
    a10.axis("off")
    if latest_ep:
        pngs = sorted(latest_ep.glob("matrix_*.png"))
        if pngs:
            img = plt.imread(str(pngs[0]))
            a10.imshow(img, aspect="auto")
            a10.set_title(f"Probe matrix  ({latest_ep.name})",
                          fontsize=9, fontweight="bold")
        else:
            a10.text(0.5, 0.5, "no matrix plots saved", ha="center", va="center",
                     color=_TEXT, fontsize=9, transform=a10.transAxes)
            a10.set_title("Probe matrix", fontsize=9, fontweight="bold")
    else:
        a10.text(0.5, 0.5, "no probe epochs", ha="center", va="center",
                 color=_TEXT, fontsize=9, transform=a10.transAxes)
        a10.set_title("Probe matrix", fontsize=9, fontweight="bold")

    # ── (1,1) Failure distribution ────────────────────────────────────────────
    a11 = ax(1, 1)
    failure_dist: dict = {}
    if latest_ep:
        fs = _json(latest_ep / "failure_summary.json", {})
        failure_dist = fs.get("failure_distribution", {})
    if failure_dist:
        labels = list(failure_dist.keys())
        counts = [failure_dist[k] for k in labels]
        colors = [_GREEN if l == "ok" else _RED for l in labels]
        short  = [l.replace("_", "\n") for l in labels]
        a11.bar(range(len(labels)), counts, color=colors, alpha=0.85, zorder=3)
        a11.set_xticks(range(len(labels)))
        a11.set_xticklabels(short, fontsize=6.5)
        a11.set_ylabel("molecules", fontsize=8)
    else:
        a11.text(0.5, 0.5, "no failure data", ha="center", va="center",
                 color=_TEXT, fontsize=9, transform=a11.transAxes)
    a11.set_title("Failure distribution", fontsize=9, fontweight="bold")

    # ── (1,2) PASS / FAIL summary ─────────────────────────────────────────────
    a12 = ax(1, 2)
    a12.axis("off")
    a12.set_title("Summary", fontsize=9, fontweight="bold")

    best_score = summary.get("best_score")
    passed     = isinstance(best_score, (int, float)) and not np.isnan(best_score)
    badge_col  = _GREEN if passed else _RED
    badge_lbl  = "PASS" if passed else "FAIL"

    a12.text(0.5, 0.80, badge_lbl,
             ha="center", va="center", fontsize=40, fontweight="bold",
             color=badge_col, transform=a12.transAxes)

    bm = summary.get("best_metrics", {})
    lines = [
        f"state      : {summary.get('state', status.get('state', '?'))}",
        f"best epoch : {summary.get('best_epoch', '?')}",
        (f"best score : {best_score:.4f}"
         if isinstance(best_score, float) else f"best score : {best_score}"),
    ]
    for k, label in [
        ("shift_mae_ppm", "shift MAE "),
        ("j_mae_hz",      "J MAE     "),
        ("presence_f1",   "pres F1   "),
        ("deg_acc",       "deg acc   "),
    ]:
        if k in bm:
            lines.append(f"{label}: {bm[k]:.3f}")

    a12.text(0.5, 0.36, "\n".join(lines),
             ha="center", va="center", fontsize=8.5,
             color=_TEXT, family="monospace",
             transform=a12.transAxes,
             bbox=dict(boxstyle="round,pad=0.55", facecolor="#0a0a12",
                       edgecolor=_GRID, alpha=0.95))

    # ── Title + save ──────────────────────────────────────────────────────────
    fig.suptitle("SpinHance  ·  model/ smoke test",
                 fontsize=13, fontweight="bold", color=_TEXT, y=0.96)

    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return passed


# ── System image viewer ────────────────────────────────────────────────────────

def _open_image(path: Path) -> None:
    """Open the PNG in the OS image viewer (best-effort)."""
    for cmd in (["xdg-open"], ["open"], ["start"]):
        try:
            subprocess.Popen(cmd + [str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue
    print(f"(no image viewer found — open manually: {path})")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="SpinHance model/ graphical smoke test")
    ap.add_argument("--out",     default="",
                    help="PNG output path (default: smoke_YYYYMMDD_HHMMSS.png)")
    ap.add_argument("--no-open", action="store_true",
                    help="skip opening the PNG in the system viewer")
    args = ap.parse_args()

    save_path = Path(args.out) if args.out else \
        Path(f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    with tempfile.TemporaryDirectory(prefix="spinhance_smoke_") as tmp:
        run_dir = Path(tmp)

        print("Running smoke training (6 epochs · 96 synthetic molecules · CPU)…")
        _run(run_dir)

        print("Training complete. Rendering figure…")
        passed = _render(run_dir, save_path)

    print(f"Saved  → {save_path.resolve()}")

    if not args.no_open:
        _open_image(save_path)

    verdict = "PASS ✓" if passed else "FAIL ✗"
    print(f"Smoke  {verdict}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
