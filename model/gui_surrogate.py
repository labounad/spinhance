"""
model/gui_surrogate.py — Surrogate renderer evaluation viewer
=============================================================

Visually evaluate the trained differentiable surrogate (matrix -> spectrum,
Branch 5): for a held-out test molecule, overlay the SURROGATE-rendered spectrum
against the pyspin GROUND TRUTH at 90 and 600 MHz, with W1 / cosine.

    AWS_PROFILE=hack-scripps streamlit run model/gui_surrogate.py

Self-contained: ground truth is simulated on the fly with pyspin (no stored
spectra needed); the surrogate checkpoint is pulled from an S3 training session
(or a local path). Uses the same test split the surrogate trained against.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

AWS_PROFILE = "hack-scripps"
AWS_REGION = "us-west-2"
S3_TRAINING = "s3://spinhance-data/training"
CACHE_DIR = Path(tempfile.gettempdir()) / "spinhance_surrogate_viewer"
DEF_JSON = str(REPO / "mol_to_spin_system/data/spin_systems_chembl_8spin_randomized.json")

st.set_page_config(page_title="SpinHance Surrogate Viewer", layout="wide",
                   initial_sidebar_state="expanded")


# ── S3 / checkpoint ────────────────────────────────────────────────────────────

def _s3_ls(prefix):
    r = subprocess.run(["aws", "s3", "ls", prefix.rstrip("/") + "/",
                        "--profile", AWS_PROFILE, "--region", AWS_REGION],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.splitlines() if r.returncode == 0 else []


@st.cache_data(ttl=60)
def _list_sessions():
    return sorted((p.strip().split()[-1].rstrip("/") for p in _s3_ls(S3_TRAINING)
                   if p.strip().startswith("PRE")), reverse=True)


@st.cache_data(ttl=60)
def _resolve_run(session):
    base = f"{S3_TRAINING}/{session}"
    runs = sorted(p.strip().split()[-1].rstrip("/") for p in _s3_ls(f"{base}/runs")
                  if p.strip().startswith("PRE"))
    return f"{base}/runs/{runs[-1]}" if runs else base


def _download_ckpt(session, run_prefix, which):
    local = CACHE_DIR / session / f"{which}.pt"
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        r = subprocess.run(["aws", "s3", "cp", f"{run_prefix}/checkpoints/{which}.pt",
                            str(local), "--profile", AWS_PROFILE, "--region", AWS_REGION],
                           capture_output=True, timeout=300)
        if r.returncode != 0:
            local.unlink(missing_ok=True)
            raise RuntimeError(r.stderr.decode()[:300])
    return local


@st.cache_resource(show_spinner="Loading surrogate checkpoint…")
def _load_surrogate(ckpt_path: str):
    import torch
    from model.renderers import build_renderer
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = {k: v for k, v in (ckpt.get("cfg", {}).get("model", {}) or {}).items() if k != "name"}
    model = build_renderer("surrogate", **mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("cfg", {}), ckpt.get("metrics", {})


# ── molecules + split (mirrors training) ─────────────────────────────────────

@st.cache_data(show_spinner="Loading molecules…")
def _test_records(json_path: str, seed: int):
    from simulation.graph_io import read_spin_systems, record_to_arrays
    from model.data.splits import make_splits
    recs = []
    for idx, rec in read_spin_systems(json_path):
        _, shifts, couplings, deg = record_to_arrays(rec)
        recs.append({"mol_id": f"mol_{idx:06d}", "smiles": rec.get("smiles"),
                     "chembl_id": rec.get("chembl_id"),
                     "shifts": np.asarray(shifts, float), "couplings": np.asarray(couplings, float),
                     "degeneracy": np.asarray(deg, int)})
    assignment, _ = make_splits(recs, seed=seed, compute_scaffold=False)
    return [r for r in recs if assignment.get(r["mol_id"]) == "test"]


@st.cache_data(show_spinner="Simulating ground truth…")
def _ground_truth(shifts_t, couplings_t, deg_t, field, points):
    from simulation.pyspin.composite import simulate_spectrum_composite
    ppm, y = simulate_spectrum_composite(np.array(shifts_t), np.array(couplings_t),
                                         list(deg_t), float(field), points=points)
    return ppm.astype(np.float64), y.astype(np.float64)


def _render_surrogate(model, shifts, couplings, deg, field, points):
    import torch
    with torch.no_grad():
        spec = model(torch.tensor(shifts, dtype=torch.float32)[None],
                     torch.tensor(couplings, dtype=torch.float32)[None],
                     torch.tensor(deg, dtype=torch.float32)[None], float(field))[0]
    ppm = np.linspace(0, 12, points)
    return ppm, spec.numpy().astype(np.float64)


def _w1_cos(a, b):
    import torch
    from model.evaluation.spectral_metrics import wasserstein1, cosine_similarity
    ta = torch.tensor(a, dtype=torch.float64)[None]
    tb = torch.tensor(b, dtype=torch.float64)[None]
    dx = 12.0 / len(a)
    return float(wasserstein1(ta, tb, dx=dx)[0]), float(cosine_similarity(ta, tb)[0])


# ── plotting ──────────────────────────────────────────────────────────────────

def _fig_overlay(gt_ppm, gt, su_ppm, su, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gt_ppm, y=gt, mode="lines", name="ground truth (pyspin)",
                             line=dict(color="#2563EB", width=1.4)))
    fig.add_trace(go.Scatter(x=su_ppm, y=su, mode="lines", name="surrogate",
                             line=dict(color="#DC2626", width=1.4)))
    fig.update_layout(title=dict(text=title, font=dict(size=13)),
                      xaxis=dict(title="δ (ppm)", autorange="reversed", showgrid=True, gridcolor="#e5e7eb"),
                      yaxis=dict(title="intensity", showgrid=True, gridcolor="#e5e7eb"),
                      height=340, margin=dict(l=50, r=20, t=46, b=44), plot_bgcolor="white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    return fig


# ── app ────────────────────────────────────────────────────────────────────────

def main():
    st.title("SpinHance — Surrogate Renderer Evaluation")
    st.caption("Surrogate-rendered spectrum vs pyspin ground truth (matrix → spectrum).")

    with st.sidebar:
        st.header("Checkpoint")
        sessions = _list_sessions()
        if sessions:
            session = st.selectbox("Session", sessions)
            which = st.selectbox("Checkpoint", ["best", "last"], index=0)
            run_prefix = _resolve_run(session)
            local_ckpt = None
        else:
            local_ckpt = st.text_input("Local checkpoint .pt", "")
            session = which = run_prefix = None
        json_path = st.text_input("Molecules JSON", DEF_JSON)

    try:
        ckpt = local_ckpt if local_ckpt else str(_download_ckpt(session, run_prefix, which))
        model, cfg, metrics = _load_surrogate(ckpt)
    except Exception as exc:
        st.error(f"Could not load checkpoint: {exc}")
        return

    # the surrogate's output grid is fixed at its trained resolution; match GT to it
    points = int(getattr(model, "points", 16384))

    if metrics:
        cols = st.columns(min(6, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i % len(cols)].metric(k, f"{v:.4f}" if isinstance(v, float) else str(v))

    seed = int((cfg.get("training", {}) or {}).get("seed", 0))
    try:
        test = _test_records(json_path, seed)
    except Exception as exc:
        st.error(f"Could not build test split: {exc}")
        return
    if not test:
        st.warning("No test molecules found.")
        return
    st.caption(f"Held-out test set: **{len(test):,}** molecules · split seed={seed}")

    mol_idx = st.selectbox("Molecule", range(len(test)),
                           format_func=lambda i: f"{test[i]['mol_id']}"
                           + (f" · {test[i]['chembl_id']}" if test[i].get("chembl_id") else ""))
    rec = test[mol_idx]
    if rec.get("smiles"):
        st.code(rec["smiles"], language=None)

    s = rec["shifts"].tolist()
    c = tuple(map(tuple, rec["couplings"].tolist()))
    d = rec["degeneracy"].tolist()
    for field in (90, 600):
        gx, gy = _ground_truth(tuple(s), c, tuple(d), field, points)
        sx, sy = _render_surrogate(model, s, rec["couplings"].tolist(), d, field, points)
        w1, cos = _w1_cos(gy / (gy.sum() + 1e-12), sy / (sy.sum() + 1e-12))
        st.plotly_chart(_fig_overlay(gx, gy, sx, sy,
                        f"{field} MHz   ·   W1 {w1:.4f}   ·   cosine {cos:.3f}"),
                        use_container_width=True)


main()
