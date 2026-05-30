"""
model/gui.py — SpinHance model evaluation dashboard.

Run from the repo root:
    conda run -n spinhance streamlit run model/gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEF_JSON = str(REPO / "mol_to_spin_system/data/spin_systems.json")
DEF_CKPT = str(REPO / "model/checkpoints/spinhance.pt")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="SpinHance Evaluator", layout="wide",
                   initial_sidebar_state="expanded")
st.title("SpinHance — Model Evaluation Dashboard")
st.caption("ResNet-1D encoder · ¹H spectrum (90/600 MHz) → 8×9 spin-shift matrix")


# ─── Session state init ───────────────────────────────────────────────────────
for key in ("model", "std", "vocab", "n_params", "ckpt_loaded"):
    st.session_state.setdefault(key, None)
st.session_state.setdefault("ckpt_loaded", False)


# ─── Helpers (cached) ─────────────────────────────────────────────────────────

def _rebuild_model(sd: dict):
    """Reconstruct SpinHanceModel from a state-dict without knowing the config."""
    from model.model import SpinHanceModel, ResNet1DEncoder

    stem_c = sd["encoder.stem.0.weight"].shape[0]
    head_hidden = sd["shift_head.0.weight"].shape[0]
    G = sd["shift_head.3.weight"].shape[0]
    n_deg = sd["deg_head.3.weight"].shape[0] // G

    if stem_c <= 24:   # "--small" encoder
        enc = ResNet1DEncoder(stem_channels=24, stage_channels=(32, 64, 128, 192),
                              blocks_per_stage=(1, 1, 1, 1))
    else:              # default encoder
        enc = ResNet1DEncoder()

    model = SpinHanceModel(n_groups=G, n_deg_classes=n_deg,
                           encoder=enc, head_hidden=head_hidden, dropout=0.0)
    model.load_state_dict(sd)
    model.eval()
    return model


def _load_ckpt(path: str):
    import torch
    from model.targets import DegeneracyVocab, Standardizer

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model"].items()}
    std_d = ckpt["standardizer"]

    std = Standardizer()
    std.shift_mean, std.shift_std = float(std_d["shift_mean"]), float(std_d["shift_std"])
    std.j_mean, std.j_std = float(std_d["j_mean"]), float(std_d["j_std"])

    vocab = DegeneracyVocab()
    model = _rebuild_model(sd)
    n_params = sum(p.numel() for p in model.parameters())
    return model, std, vocab, n_params


@st.cache_data(show_spinner="Simulating NMR spectrum…")
def _simulate(shifts_t: tuple, couplings_t: tuple, degeneracy_t: tuple,
              field_mhz: int) -> tuple[np.ndarray, np.ndarray]:
    from simulation.pyspin.composite import simulate_spectrum_composite
    shifts = np.array(shifts_t)
    couplings = np.array(couplings_t)
    degeneracy = list(degeneracy_t)
    ppm_axis, intensity = simulate_spectrum_composite(
        shifts, couplings, degeneracy, float(field_mhz))
    return ppm_axis.astype(np.float64), intensity.astype(np.float64)


@st.cache_data(show_spinner="Loading molecules…")
def _load_records(json_path: str) -> list[dict]:
    from simulation.graph_io import read_spin_systems, record_to_arrays
    from model.splits import canonical_order, reorder
    records = []
    for idx, rec in read_spin_systems(json_path):
        _, shifts, couplings, degeneracy = record_to_arrays(rec)
        shifts = np.array(shifts, dtype=float)
        couplings = np.array(couplings, dtype=float)
        degeneracy = np.array(degeneracy, dtype=int)
        # Apply canonical ordering so GT display matches the model's output order
        order = canonical_order(shifts, couplings, degeneracy)
        shifts, couplings, degeneracy = reorder(shifts, couplings, degeneracy, order)
        records.append({
            "idx": idx,
            "mol_id": f"mol_{idx:06d}",
            "chembl_id": rec.get("chembl_id", ""),
            "smiles": rec.get("smiles", ""),
            "shifts": shifts,
            "couplings": couplings,
            "degeneracy": degeneracy,
        })
    return records


def _run_inference(model, intensity: np.ndarray, std, vocab) -> dict:
    import torch
    from model.metrics import decode

    x = torch.from_numpy(intensity.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
    pred_np = {k: v.float().cpu().numpy() for k, v in pred.items()}
    return decode(pred_np, std, vocab)


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _fig_spectrum(ppm_axis: np.ndarray, intensity: np.ndarray,
                  title: str = "", color: str = "#2563EB") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=ppm_axis, y=intensity, mode="lines",
        line=dict(color=color, width=1.5)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="δ (ppm)", autorange="reversed",
                   showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(title="Intensity (norm.)", showgrid=True, gridcolor="#e5e7eb"),
        height=240, margin=dict(l=50, r=20, t=36, b=40),
        plot_bgcolor="white", showlegend=False,
    )
    return fig


def _fig_matrix(shifts: np.ndarray, couplings: np.ndarray, degeneracy: np.ndarray,
                title: str = "") -> go.Figure:
    G = len(shifts)
    labels = [f"G{i+1}" for i in range(G)]

    # Off-diagonal: J couplings; diagonal zeroed for coloring, shown as annotation
    z = couplings.copy()
    np.fill_diagonal(z, 0.0)

    text = []
    for r in range(G):
        row = []
        for c in range(G):
            if r == c:
                row.append(f"<b>{shifts[r]:.2f}</b><br>n={degeneracy[r]}")
            elif abs(couplings[r, c]) > 0.01:
                row.append(f"{couplings[r, c]:.1f}")
            else:
                row.append("")
        text.append(row)

    max_j = max(np.abs(z).max(), 1.0)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-max_j, zmax=max_j,
        text=text, texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="J (Hz)", thickness=14, len=0.8),
    ))
    # Highlight diagonal cells with a neutral tint
    for i in range(G):
        fig.add_shape(type="rect",
                      x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      fillcolor="rgba(200,200,200,0.3)", line=dict(width=0))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=380, margin=dict(l=60, r=60, t=36, b=50),
        xaxis=dict(title="Spin group"), yaxis=dict(title="Spin group"),
    )
    return fig


def _per_group_df(gt_shifts, gt_deg, pred_shifts, pred_deg) -> pd.DataFrame:
    G = len(gt_shifts)
    return pd.DataFrame({
        "Group": [f"G{i+1}" for i in range(G)],
        "GT shift (ppm)": [f"{gt_shifts[i]:.3f}" for i in range(G)],
        "Pred shift (ppm)": [f"{pred_shifts[i]:.3f}" for i in range(G)],
        "|Δ shift|": [f"{abs(pred_shifts[i] - gt_shifts[i]):.3f}" for i in range(G)],
        "GT deg": [int(gt_deg[i]) for i in range(G)],
        "Pred deg": [int(pred_deg[i]) for i in range(G)],
        "Deg match": ["✓" if pred_deg[i] == gt_deg[i] else "✗" for i in range(G)],
    })


def _metrics(pred_dec: dict, gt: dict) -> dict:
    G = len(gt["shifts"])
    iu = np.triu_indices(G, 1)

    shift_mae = float(np.abs(pred_dec["shifts"][0] - gt["shifts"]).mean())

    gt_j = gt["couplings"][iu]
    pred_j = pred_dec["couplings"][0][iu]
    gt_pres = np.abs(gt_j) > 0.01
    j_mae = (float(np.abs(pred_j[gt_pres] - gt_j[gt_pres]).mean())
             if gt_pres.any() else float("nan"))

    pred_pres = pred_dec["presence"][0] > 0.5
    tp = float((pred_pres & gt_pres).sum())
    fp = float((pred_pres & ~gt_pres).sum())
    fn = float((~pred_pres & gt_pres).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    pred_deg = pred_dec["degeneracy"][0]
    deg_acc = float((pred_deg == gt["degeneracy"]).mean())

    return {"shift_mae": shift_mae, "j_mae": j_mae, "presence_f1": f1,
            "deg_acc": deg_acc}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    ckpt_path = st.text_input("Checkpoint (.pt)", value=DEF_CKPT,
                               help="Path to a model checkpoint saved by train.py")
    if st.button("Load Model", type="primary", use_container_width=True):
        if not Path(ckpt_path).exists():
            st.error(f"Not found: {ckpt_path}")
        else:
            with st.spinner("Loading…"):
                try:
                    m, s, v, n = _load_ckpt(ckpt_path)
                    st.session_state.update(
                        model=m, std=s, vocab=v, n_params=n, ckpt_loaded=True)
                    st.success(f"Loaded ({n / 1e6:.2f}M params)")
                except Exception as exc:
                    st.error(f"Load failed: {exc}")

    if st.session_state["ckpt_loaded"]:
        std = st.session_state["std"]
        st.success(f"Model ready · {st.session_state['n_params'] / 1e6:.2f}M params")
        st.caption(
            f"Shift: μ={std.shift_mean:.2f} σ={std.shift_std:.2f} ppm  "
            f"J: μ={std.j_mean:.2f} σ={std.j_std:.2f} Hz"
        )
    else:
        st.warning("No checkpoint loaded — inference disabled")

    st.divider()
    json_path = st.text_input("spin_systems.json", value=DEF_JSON)
    field_mhz = st.radio("Simulation field", [90, 600], index=0, horizontal=True,
                          help="Field strength used to simulate the input spectrum")


# ─── Data ─────────────────────────────────────────────────────────────────────
if not Path(json_path).exists():
    st.error(f"spin_systems.json not found: {json_path}")
    st.stop()

records = _load_records(json_path)
st.caption(f"{len(records)} molecules · canonical shift order (δ descending)")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["Single Molecule Inspector", "Batch Evaluation"])


# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Single molecule
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    mol_idx = st.slider("Molecule", 0, len(records) - 1, 0,
                        help="Index into spin_systems.json (canonical shift order)")
    rec = records[mol_idx]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mol ID", rec["mol_id"])
    c2.metric("ChEMBL", rec["chembl_id"] or "—")
    c3.metric("Spin groups (G)", len(rec["shifts"]))
    c4.metric("Total spins", int(rec["degeneracy"].sum()))

    if rec["smiles"]:
        with st.expander("SMILES", expanded=False):
            st.code(rec["smiles"], language=None)

    st.divider()

    # Generate spectrum
    ppm_axis, intensity = _simulate(
        tuple(rec["shifts"].tolist()),
        tuple(map(tuple, rec["couplings"].tolist())),
        tuple(rec["degeneracy"].tolist()),
        field_mhz,
    )

    col_spec, col_gt = st.columns(2)
    with col_spec:
        st.plotly_chart(
            _fig_spectrum(ppm_axis, intensity,
                          title=f"{field_mhz} MHz input spectrum"),
            use_container_width=True)
    with col_gt:
        st.plotly_chart(
            _fig_matrix(rec["shifts"], rec["couplings"], rec["degeneracy"],
                        title="Ground truth matrix (diagonal=δ ppm, off-diag=J Hz)"),
            use_container_width=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    if not st.session_state["ckpt_loaded"]:
        st.info("Load a checkpoint in the sidebar to run inference on this molecule.")
    else:
        if st.button("▶  Run Inference", type="primary"):
            with st.spinner("Running inference…"):
                dec = _run_inference(
                    st.session_state["model"], intensity,
                    st.session_state["std"], st.session_state["vocab"])
            st.session_state["pred_dec"] = dec
            st.session_state["pred_mol_idx"] = mol_idx

        if (st.session_state.get("pred_dec") is not None
                and st.session_state.get("pred_mol_idx") == mol_idx):
            dec = st.session_state["pred_dec"]
            pred_shifts = dec["shifts"][0]
            pred_couplings = dec["couplings"][0]
            pred_deg = dec["degeneracy"][0]

            col_pred, col_metrics = st.columns(2)

            with col_pred:
                st.plotly_chart(
                    _fig_matrix(pred_shifts, pred_couplings, pred_deg,
                                title="Predicted matrix"),
                    use_container_width=True)

            with col_metrics:
                m = _metrics(dec, rec)
                st.subheader("Metrics vs ground truth")
                mc1, mc2 = st.columns(2)
                mc1.metric("Shift MAE", f"{m['shift_mae']:.3f} ppm")
                mc2.metric("J MAE (present)", f"{m['j_mae']:.2f} Hz"
                           if not np.isnan(m["j_mae"]) else "no couplings")
                mc3, mc4 = st.columns(2)
                mc3.metric("Presence F1", f"{m['presence_f1']:.3f}")
                mc4.metric("Degeneracy acc", f"{m['deg_acc']:.3f}")

            st.dataframe(
                _per_group_df(rec["shifts"], rec["degeneracy"],
                              pred_shifts, pred_deg),
                use_container_width=True, hide_index=True)

            # Overlay predicted spectrum
            pred_ppm, pred_int = _simulate(
                tuple(pred_shifts.tolist()),
                tuple(map(tuple, pred_couplings.tolist())),
                tuple(pred_deg.tolist()),
                field_mhz,
            )
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(
                x=ppm_axis, y=intensity, mode="lines",
                line=dict(color="#2563EB", width=1.5), name="Input (GT)"))
            fig_ov.add_trace(go.Scatter(
                x=pred_ppm, y=pred_int, mode="lines",
                line=dict(color="#DC2626", width=1.5, dash="dash"),
                name="Rendered (pred)"))
            fig_ov.update_layout(
                title="Spectral overlay — ground truth (blue) vs rendered predicted (red dashed)",
                xaxis=dict(title="δ (ppm)", autorange="reversed",
                           showgrid=True, gridcolor="#e5e7eb"),
                yaxis=dict(title="Intensity", showgrid=True, gridcolor="#e5e7eb"),
                height=280, margin=dict(l=50, r=20, t=36, b=40),
                plot_bgcolor="white", legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_ov, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Batch evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    if not st.session_state["ckpt_loaded"]:
        st.info("Load a checkpoint in the sidebar to run batch evaluation.")
    else:
        col_n, col_seed = st.columns([2, 1])
        n_eval = col_n.number_input(
            "Molecules to evaluate", min_value=1, max_value=len(records),
            value=min(50, len(records)), step=10)
        seed = col_seed.number_input("Random seed", value=42, min_value=0)

        if st.button("▶  Run Batch Evaluation", type="primary"):
            rng = np.random.default_rng(int(seed))
            idxs = rng.choice(len(records), size=int(n_eval), replace=False)

            rows = []
            shift_gt_all, shift_pred_all = [], []
            j_gt_all, j_pred_all = [], []

            prog = st.progress(0.0, text="Evaluating…")
            for k, i in enumerate(idxs):
                r = records[i]
                pa, intens = _simulate(
                    tuple(r["shifts"].tolist()),
                    tuple(map(tuple, r["couplings"].tolist())),
                    tuple(r["degeneracy"].tolist()),
                    field_mhz,
                )
                dec = _run_inference(
                    st.session_state["model"], intens,
                    st.session_state["std"], st.session_state["vocab"])
                m = _metrics(dec, r)
                rows.append({
                    "mol_id": r["mol_id"],
                    "shift_mae (ppm)": round(m["shift_mae"], 4),
                    "j_mae (Hz)": round(m["j_mae"], 3) if not np.isnan(m["j_mae"]) else None,
                    "presence_f1": round(m["presence_f1"], 4),
                    "deg_acc": round(m["deg_acc"], 4),
                })

                G = len(r["shifts"])
                shift_gt_all.extend(r["shifts"].tolist())
                shift_pred_all.extend(dec["shifts"][0].tolist())
                iu = np.triu_indices(G, 1)
                j_gt_all.extend(r["couplings"][iu].tolist())
                j_pred_all.extend(dec["couplings"][0][iu].tolist())

                prog.progress((k + 1) / len(idxs), text=f"{r['mol_id']} ({k+1}/{len(idxs)})")

            prog.empty()
            st.session_state["batch"] = {
                "rows": rows,
                "shifts": (shift_gt_all, shift_pred_all),
                "j": (j_gt_all, j_pred_all),
            }

        if "batch" in st.session_state:
            b = st.session_state["batch"]
            df = pd.DataFrame(b["rows"])

            # ── Aggregate ─────────────────────────────────────────────────────
            st.subheader("Aggregate metrics")
            num_cols = ["shift_mae (ppm)", "j_mae (Hz)", "presence_f1", "deg_acc"]
            agg = df[num_cols].apply(pd.to_numeric, errors="coerce").describe().loc[
                ["mean", "std", "min", "25%", "50%", "75%", "max"]]
            st.dataframe(agg.style.format("{:.4f}"), use_container_width=True)

            # ── Scatter plots ─────────────────────────────────────────────────
            sg, sp = b["shifts"]
            jg, jp = b["j"]
            sg, sp = np.array(sg), np.array(sp)
            jg, jp = np.array(jg), np.array(jp)

            sc1, sc2 = st.columns(2)
            with sc1:
                lo, hi = min(sg.min(), sp.min()), max(sg.max(), sp.max())
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    x=sg.tolist(), y=sp.tolist(), mode="markers",
                    marker=dict(size=3, color="#2563EB", opacity=0.4), name="groups"))
                fig_s.add_trace(go.Scatter(
                    x=[lo, hi], y=[lo, hi], mode="lines",
                    line=dict(color="#DC2626", dash="dash", width=1), name="ideal"))
                fig_s.update_layout(
                    title="Shift: GT vs Predicted (ppm)",
                    xaxis_title="GT (ppm)", yaxis_title="Predicted (ppm)",
                    height=350, margin=dict(l=50, r=20, t=36, b=40),
                    showlegend=False, plot_bgcolor="white")
                st.plotly_chart(fig_s, use_container_width=True)

            with sc2:
                nonzero = np.abs(jg) > 0.1
                if nonzero.any():
                    lo, hi = jg[nonzero].min(), jg[nonzero].max()
                    fig_j = go.Figure()
                    fig_j.add_trace(go.Scatter(
                        x=jg[nonzero].tolist(), y=jp[nonzero].tolist(),
                        mode="markers",
                        marker=dict(size=3, color="#D97706", opacity=0.4)))
                    fig_j.add_trace(go.Scatter(
                        x=[lo, hi], y=[lo, hi], mode="lines",
                        line=dict(color="#DC2626", dash="dash", width=1)))
                    fig_j.update_layout(
                        title="J coupling: GT vs Predicted (Hz, nonzero only)",
                        xaxis_title="GT (Hz)", yaxis_title="Predicted (Hz)",
                        height=350, margin=dict(l=50, r=20, t=36, b=40),
                        showlegend=False, plot_bgcolor="white")
                    st.plotly_chart(fig_j, use_container_width=True)
                else:
                    st.info("No nonzero GT couplings in this sample.")

            # ── Per-molecule table ─────────────────────────────────────────────
            with st.expander("Per-molecule results", expanded=False):
                st.dataframe(df, use_container_width=True, hide_index=True)
