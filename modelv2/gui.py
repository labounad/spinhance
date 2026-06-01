"""
modelv2.gui — training-session viewer & diagnostic dashboard
============================================================
A single Streamlit app for visualizing, diagnosing, and exploring model quality
— the human element for designing the network's training (DESIGN.md calls this
essential). It reuses the layout and feature set of the old model/gui.py and
model/live_dashboard.py, retargeted at the modelv2 artifact tree.

Run:
    PYTHONPATH=. streamlit run modelv2/gui.py

Storage
-------
Models train remotely and write to ``s3://spinhance-data/training/sessionXXX/``;
a local session directory (e.g. ``modelv2/runs/...``) works identically. Every
artifact is pulled from the session root. The molecule viewer and per-epoch
diagnostics run on the held-out ``diagnostic_set.json`` (+ ``diagnostic_spectra.
npy``) — 500 molecules the model never saw — so "held out" is true by
construction, not re-derived from the split.

Two pages:
  1. Session browser — pick a session under the configured root.
  2. Session analysis — status, training curves, the diagnostics that reveal
     mean-collapse vs. capacity vs. information limits, a per-epoch validation
     bar chart, the probe/failure tables, and a molecule inspector that runs the
     selected epoch's checkpoint on the held-out spectra.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the ``modelv2`` namespace package importable when run via
# ``streamlit run modelv2/gui.py`` (no -m, so the repo root may be off sys.path).
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from modelv2 import data as D
from modelv2 import train as T

DEFAULT_ROOT = "s3://spinhance-data/training"

st.set_page_config(page_title="modelv2 viewer", layout="wide",
                   initial_sidebar_state="expanded")


# =============================================================================
# Storage — read a session root that is local or s3://
# =============================================================================

def _full(root, name):
    return f"{root.rstrip('/')}/{name}"


def read_bytes(root, name):
    if T.is_s3(root):
        return T.s3_get_bytes(_full(root, name))
    p = Path(root) / name
    return p.read_bytes() if p.exists() else None


def read_json(root, name, default=None):
    b = read_bytes(root, name)
    if b is None:
        return default
    try:
        return json.loads(b)
    except Exception:
        return default


def read_jsonl(root, name):
    b = read_bytes(root, name)
    if not b:
        return []
    rows = []
    for line in b.decode().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def read_npy(root, name):
    b = read_bytes(root, name)
    return np.load(io.BytesIO(b)) if b is not None else None


def list_sessions(root):
    if T.is_s3(root):
        try:
            return sorted(T.s3_list_prefixes(root), reverse=True)
        except Exception as e:
            st.error(f"Cannot list S3 sessions: {e}")
            return []
    p = Path(root)
    return sorted([d.name for d in p.iterdir() if d.is_dir()], reverse=True) if p.exists() else []


def list_epochs(root):
    """Epoch numbers that have a probe checkpoint, ascending."""
    probes = f"{root.rstrip('/')}/probes"
    names = []
    if T.is_s3(root):
        try:
            names = T.s3_list_prefixes(probes)
        except Exception:
            names = []
    else:
        p = Path(probes)
        names = [d.name for d in p.iterdir() if d.is_dir()] if p.exists() else []
    out = []
    for n in names:
        if n.startswith("epoch_"):
            try:
                out.append(int(n[6:]))
            except ValueError:
                pass
    return sorted(out)


# =============================================================================
# Cached loaders
# =============================================================================

@st.cache_data(show_spinner=False)
def load_metrics(root, _bust=0):
    """Flatten metrics.jsonl val rows into a per-epoch DataFrame."""
    rows = read_jsonl(root, "metrics.jsonl")
    flat = []
    for r in rows:
        if r.get("split") != "val":
            continue
        m = r.get("metrics", {})
        rec = {"epoch": r.get("epoch")}
        for k, v in m.items():
            if isinstance(v, (int, float)):
                rec[k] = v
        if "shift_mae_ppm" in rec and "j_mae_hz" in rec:
            rec["score"] = rec["shift_mae_ppm"] + rec["j_mae_hz"] / 10.0
        flat.append(rec)
    if not flat:
        return pd.DataFrame(columns=["epoch"])
    df = pd.DataFrame(flat)
    if "epoch" not in df.columns:
        return pd.DataFrame(columns=["epoch"])
    return df.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_events(root, _bust=0):
    rows = read_jsonl(root, "events.jsonl")
    steps = [r for r in rows if r.get("event") == "train_step"]
    return pd.json_normalize(steps) if steps else pd.DataFrame()


@st.cache_data(show_spinner=True)
def load_diagnostic_set(root):
    return read_json(root, "diagnostic_set.json", [])


@st.cache_resource(show_spinner="Loading checkpoint…")
def load_model(root, epoch):
    """Reconstruct (model, std, vocab) from a probe/best checkpoint (EMA weights)."""
    import torch
    from modelv2.model import SpinHanceModel
    name = (f"probes/epoch_{epoch:04d}/checkpoint.pt" if epoch is not None
            else "checkpoints/best.pt")
    b = read_bytes(root, name) or read_bytes(root, "checkpoints/best.pt")
    if b is None:
        raise FileNotFoundError(f"no checkpoint at {name}")
    ckpt = torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
    model = SpinHanceModel(**ckpt["model_build"])
    sd = dict(ckpt["model_state"])
    if ckpt.get("ema_state"):                       # overlay shadow params for eval
        sd.update({k: v for k, v in ckpt["ema_state"].items() if k in sd})
    model.load_state_dict(sd)
    model.eval()
    std = D.Standardizer.from_state(ckpt["standardizer"])
    vocab = D.DegeneracyVocab(tuple(ckpt["vocab"]))
    return model, std, vocab, ckpt.get("cfg", {})


# =============================================================================
# Plotting (reused from the old viewer)
# =============================================================================

def fig_spectrum(intensity, ppm_from=0.0, ppm_to=12.0, title="", color="#2563EB"):
    ppm = np.linspace(ppm_from, ppm_to, len(intensity))
    fig = go.Figure(go.Scatter(x=ppm, y=intensity, mode="lines",
                               line=dict(color=color, width=1.3)))
    fig.update_layout(title=dict(text=title, font=dict(size=12)),
                      xaxis=dict(title="δ (ppm)", autorange="reversed"),
                      yaxis=dict(title="intensity"), height=220,
                      margin=dict(l=40, r=10, t=32, b=36),
                      plot_bgcolor="white", showlegend=False)
    return fig


def fig_matrix(shifts, couplings, degeneracy, title=""):
    G = len(shifts)
    labels = [f"G{i+1}" for i in range(G)]
    z = np.array(couplings, dtype=float).copy()
    np.fill_diagonal(z, 0.0)
    text = []
    for r in range(G):
        row = []
        for c in range(G):
            if r == c:
                row.append(f"<b>{shifts[r]:.2f}</b><br>n={int(degeneracy[r])}")
            elif abs(z[r, c]) > 0.01:
                row.append(f"{z[r, c]:.1f}")
            else:
                row.append("")
        text.append(row)
    max_j = max(float(np.abs(z).max()), 1.0)
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale="RdBu", zmid=0,
                               zmin=-max_j, zmax=max_j, text=text,
                               texttemplate="%{text}", textfont=dict(size=9),
                               colorbar=dict(title="J (Hz)", thickness=12, len=0.8)))
    for i in range(G):
        fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      fillcolor="rgba(200,200,200,0.3)", line=dict(width=0))
    fig.update_layout(title=dict(text=title, font=dict(size=12)), height=340,
                      margin=dict(l=50, r=50, t=34, b=40),
                      xaxis=dict(title="spin group"), yaxis=dict(title="spin group"))
    return fig


def line(df, x, ys, title, height=260):
    fig = go.Figure()
    for y in (ys if isinstance(ys, (list, tuple)) else [ys]):
        if y in df.columns:
            fig.add_scatter(x=df[x], y=df[y], mode="lines+markers", name=y)
    fig.update_layout(title=dict(text=title, font=dict(size=12)), height=height,
                      margin=dict(l=40, r=10, t=34, b=34), plot_bgcolor="white")
    return fig


# =============================================================================
# Page 1 — session browser
# =============================================================================

def page_select():
    st.title("modelv2 — training session viewer")
    root = st.session_state.get("root", DEFAULT_ROOT)
    root = st.text_input("Session root (local path or s3:// URI)", value=root)
    st.session_state["root"] = root
    cols = st.columns([1, 5])
    if cols[0].button("↺ Refresh"):
        st.cache_data.clear()
        st.rerun()
    sessions = list_sessions(root)
    if not sessions:
        st.warning(f"No sessions found under `{root}`.")
        return
    sel = st.selectbox("Session", sessions)
    if st.button("Open session →", type="primary"):
        st.session_state.update(session=sel, page="analysis")
        for k in ("sel_epoch", "pred"):
            st.session_state.pop(k, None)
        st.rerun()


# =============================================================================
# Page 2 — session analysis
# =============================================================================

def page_analysis():
    root = st.session_state["root"]
    session = st.session_state["session"]
    sroot = _full(root, session)

    nav, title = st.columns([1, 8])
    if nav.button("← Sessions"):
        st.session_state["page"] = "select"
        st.rerun()
    title.title(f"Session: `{session}`")

    if st.sidebar.button("↺ Reload data"):
        st.cache_data.clear()
        st.rerun()

    status = read_json(sroot, "status.json", {})
    summary = read_json(sroot, "summary.json", {})
    cfg = read_json(sroot, "config.json", {})

    state = status.get("state", "—")
    icon = {"running": "🟢", "finished": "🔵"}.get(state, "🔴")
    c = st.columns(6)
    c[0].metric("State", f"{icon} {state}")
    c[1].metric("Epoch", f"{status.get('epoch', '?')}/{status.get('epochs', '?')}")
    best = status.get("best_score")
    c[2].metric("Best score", f"{best:.4f}" if isinstance(best, (int, float)) else "—")
    c[3].metric("Best epoch", status.get("best_epoch", "?"))
    c[4].metric("Device", status.get("device", "?"))
    c[5].metric("Step", status.get("global_step", "?"))

    metrics = load_metrics(sroot)
    if metrics.empty:
        st.info("No validation metrics yet — training may be warming up.")
        return

    # ── Validation score across epochs (click a bar to load that checkpoint) ──
    st.subheader("Validation score across epochs")
    st.caption("score = shift_MAE (ppm) + J_MAE (Hz)/10 · lower is better · click a bar")
    valid = metrics.dropna(subset=["score"])
    best_row = valid.loc[valid["score"].idxmin()]
    best_epoch = int(best_row["epoch"])
    sel_epoch = st.session_state.get("sel_epoch", best_epoch)

    colors = ["#DC2626" if int(e) == best_epoch else
              ("#EAB308" if int(e) == sel_epoch else "#93C5FD")
              for e in valid["epoch"]]
    bar = go.Figure(go.Bar(
        x=valid["epoch"], y=valid["score"], marker_color=colors,
        customdata=np.stack([valid["shift_mae_ppm"], valid["j_mae_hz"]], -1),
        hovertemplate="ep%{x}<br>score %{y:.4f}<br>shift %{customdata[0]:.3f} ppm"
                      "<br>J %{customdata[1]:.2f} Hz<extra></extra>"))
    bar.update_layout(height=300, margin=dict(l=50, r=20, t=10, b=40),
                      plot_bgcolor="white", showlegend=False, dragmode=False,
                      xaxis_title="epoch", yaxis_title="score")
    ev = st.plotly_chart(bar, use_container_width=True, on_select="rerun",
                         selection_mode=("points",), key="score_bar",
                         config={"displayModeBar": False})
    pts = ev.selection.points if hasattr(ev, "selection") else []
    if pts:
        clicked = int(pts[0]["x"])
        if clicked != sel_epoch:
            st.session_state["sel_epoch"] = clicked
            st.rerun()

    # ── Training curves + diagnostics ─────────────────────────────────────────
    tab_curves, tab_diag, tab_probe, tab_mol = st.tabs(
        ["Curves", "Diagnostics", "Probes / failures", "Molecule inspector"])

    with tab_curves:
        cc = st.columns(3)
        with cc[0]:
            st.plotly_chart(line(metrics, "epoch",
                                 ["shift_mae_ppm", "h_shift_mae_ppm", "baseline_shift_mae_ppm"],
                                 "Shift MAE (ppm) · incl. Hungarian + baseline"),
                            use_container_width=True)
        with cc[1]:
            st.plotly_chart(line(metrics, "epoch",
                                 ["j_mae_hz", "h_j_mae_hz", "baseline_j_mae_hz"],
                                 "J MAE (Hz) · incl. Hungarian + baseline"),
                            use_container_width=True)
        with cc[2]:
            st.plotly_chart(line(metrics, "epoch",
                                 ["presence_f1", "deg_acc_balanced"],
                                 "Presence F1 / deg acc (balanced)"),
                            use_container_width=True)
        events = load_events(sroot)
        if not events.empty:
            ec = st.columns(3)
            with ec[0]:
                st.plotly_chart(line(events, "step", "loss_total", "Total train loss"),
                                use_container_width=True)
            with ec[1]:
                st.plotly_chart(line(events, "step",
                                     ["loss_shift", "loss_jmag", "loss_presence", "loss_deg"],
                                     "Per-term loss (raw)"), use_container_width=True)
            with ec[2]:
                gcols = [c for c in events.columns if c.startswith("gradnorm_")]
                st.plotly_chart(line(events, "step", gcols, "Per-head gradient norms"),
                                use_container_width=True)

    with tab_diag:
        st.caption("Why does it plateau? Var(pred)/Var(target) ≪ 1 and Pearson r ≈ 0 "
                   "flag mean-collapse; a large train−val gap points at "
                   "generalization rather than capacity.")
        dc = st.columns(3)
        with dc[0]:
            st.plotly_chart(line(metrics, "epoch",
                                 ["var_ratio_shift_mean", "var_ratio_j_mean"],
                                 "Var(pred)/Var(target)  (1.0 = matched)"),
                            use_container_width=True)
        with dc[1]:
            st.plotly_chart(line(metrics, "epoch",
                                 ["pearson_shift_mean", "pearson_j_mean"],
                                 "Pearson r (pred vs target)"),
                            use_container_width=True)
        with dc[2]:
            st.plotly_chart(line(metrics, "epoch",
                                 ["pred_logvar_shift_mean", "pred_logvar_j_mean"],
                                 "Predicted log σ² (aleatoric)"),
                            use_container_width=True)
        gc = st.columns(2)
        with gc[0]:
            st.plotly_chart(line(metrics, "epoch", ["gap_shift_mae_ppm", "gap_j_mae_hz"],
                                 "Train − val gap"), use_container_width=True)
        with gc[1]:
            # per-cell variance ratio at the selected epoch (if stored)
            row = metrics[metrics["epoch"] == sel_epoch]
            st.write(f"Per-slot diagnostics at epoch {sel_epoch}")
            rows_full = read_jsonl(sroot, "metrics.jsonl")
            cell = next((r["metrics"] for r in rows_full
                         if r.get("split") == "val" and r.get("epoch") == sel_epoch), {})
            if cell.get("var_ratio_shift"):
                vr = pd.DataFrame({"slot": [f"G{i+1}" for i in range(len(cell["var_ratio_shift"]))],
                                   "var_ratio": cell["var_ratio_shift"],
                                   "pearson": cell.get("pearson_shift", [])})
                st.bar_chart(vr.set_index("slot"))

    with tab_probe:
        epochs = list_epochs(sroot)
        if not epochs:
            st.info("No probe epochs saved yet.")
        else:
            pe = st.selectbox("Probe epoch", epochs, index=len(epochs) - 1)
            pprefix = f"probes/epoch_{pe:04d}"
            pm = read_json(sroot, f"{pprefix}/probe_metrics.json", {})
            if pm:
                mc = st.columns(len(pm))
                for i, (k, v) in enumerate(pm.items()):
                    mc[i].metric(k.replace("_", " "), f"{v:.3f}")
            fs = read_json(sroot, f"{pprefix}/failure_summary.json", {})
            if fs:
                st.info(f"Dominant failure: **{fs.get('dominant_failure', '—')}** · "
                        f"{fs.get('recommendation', '')}")
                fd = fs.get("failure_distribution", {})
                if fd:
                    st.bar_chart(pd.Series(fd, name="count"))
            worst = read_json(sroot, f"{pprefix}/worst_shift_cases.json", [])
            if worst:
                st.write("Worst cases by shift MAE")
                st.dataframe(pd.DataFrame([{
                    "mol_id": w["mol_id"], "shift_mae": round(w["shift_mae_ppm"], 3),
                    "j_mae": round(w["j_mae_hz"], 2), "f1": round(w["presence_f1"], 2),
                    "deg_acc": round(w["deg_acc"], 2), "failure": w.get("failure_type", ""),
                } for w in worst[:12]]), use_container_width=True, hide_index=True)
            # matrix PNGs
            keys = []
            try:
                if T.is_s3(sroot):
                    keys = [k for k in T.s3_list_keys(f"{sroot}/{pprefix}")
                            if k.endswith(".png") and "matrix_" in k]
                else:
                    d = Path(sroot) / pprefix
                    keys = [p.name for p in d.glob("matrix_*.png")] if d.exists() else []
            except Exception:
                keys = []
            if keys:
                st.write(f"Matrix plots ({len(keys)}; showing 6)")
                icols = st.columns(3)
                for i, k in enumerate(sorted(keys)[:6]):
                    b = read_bytes(sroot, f"{pprefix}/{k}")
                    if b:
                        icols[i % 3].image(b)

    with tab_mol:
        _molecule_inspector(sroot, sel_epoch, best_epoch, cfg)


def _simulate_spectrum(shifts, couplings, degeneracy, ppm_from=0.0, ppm_to=12.0):
    """Return a 16384-point float32 spectrum array, or None on failure."""
    try:
        from simulation.pyspin.composite import simulate_spectrum_composite
        _, spec = simulate_spectrum_composite(
            shifts, couplings, [int(d) for d in degeneracy], 90.0,
            ppm_from=ppm_from, ppm_to=ppm_to)
        return spec.astype(np.float32)
    except Exception as e:
        st.warning(f"Spectrum simulation failed: {e}")
        return None


def _molecule_inspector(sroot, sel_epoch, best_epoch, cfg):
    recs = load_diagnostic_set(sroot)
    if not recs:
        st.info("No diagnostic_set.json found for this session.")
        return
    st.caption(f"Held-out diagnostic set: **{len(recs)}** molecules the model never "
               f"saw · inference uses epoch **{sel_epoch}** (click a bar above to change).")

    idx = st.selectbox("Molecule", range(len(recs)),
                       format_func=lambda i: (recs[i]["mol_id"]
                                              + (f" · {recs[i].get('chembl_id')}" if recs[i].get("chembl_id") else "")
                                              + (f" · {str(recs[i].get('smiles'))[:40]}" if recs[i].get("smiles") else "")))
    rec = recs[idx]
    shifts = np.asarray(rec["shifts"], float)
    coup = np.asarray(rec["couplings"], float)
    deg = np.asarray(rec["degeneracy"])
    ppm_from = float(cfg.get("ppm_from", 0.0))
    ppm_to = float(cfg.get("ppm_to", 12.0))

    with st.spinner("Simulating ground-truth spectrum…"):
        gt_spec = _simulate_spectrum(shifts, coup, deg, ppm_from, ppm_to)

    run = st.button("▶ Run inference", type="primary")
    col_gt, col_pred = st.columns(2)

    with col_gt:
        st.markdown("#### Ground truth")
        st.plotly_chart(fig_matrix(shifts, coup, deg,
                                   "GT matrix (diag δ ppm · off-diag J Hz)"),
                        use_container_width=True)
        if gt_spec is not None:
            st.plotly_chart(fig_spectrum(gt_spec.astype(float), ppm_from, ppm_to,
                                         "GT spectrum (90 MHz, simulated)"),
                            use_container_width=True)

    if run:
        try:
            model, std, vocab, _ = load_model(sroot, sel_epoch)
        except Exception as e:
            st.error(f"Could not load checkpoint: {e}")
            return
        import torch
        x_np = gt_spec if gt_spec is not None else np.zeros(cfg.get("points", 16384), np.float32)
        x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pred = model(x)
        pred_np = {k: v.float().cpu().numpy() for k, v in pred.items()}
        dec = T.decode(pred_np, std, vocab)
        tgt = std.transform(D.encode_target(
            {"shifts": shifts, "couplings": coup, "degeneracy": deg}, vocab))
        met = T.compute_metrics(pred_np, {k: tgt[k][None] for k in
                                          ("shifts", "j_mag", "j_presence", "deg_class")},
                                std, vocab)
        with st.spinner("Simulating predicted spectrum…"):
            pred_spec = _simulate_spectrum(dec["shifts"][0], dec["couplings"][0],
                                           dec["degeneracy"][0], ppm_from, ppm_to)
        st.session_state["pred"] = {
            "mol": rec["mol_id"], "dec": dec, "met": met, "pred_spec": pred_spec,
        }

    pred_state = st.session_state.get("pred")
    with col_pred:
        st.markdown("#### Model prediction")
        if pred_state and pred_state["mol"] == rec["mol_id"]:
            dec = pred_state["dec"]
            met = pred_state["met"]
            pred_spec = pred_state.get("pred_spec")
            st.plotly_chart(fig_matrix(dec["shifts"][0], dec["couplings"][0],
                                       dec["degeneracy"][0], "Predicted matrix"),
                            use_container_width=True)
            if pred_spec is not None:
                st.plotly_chart(fig_spectrum(pred_spec.astype(float), ppm_from, ppm_to,
                                             "Predicted spectrum (90 MHz, simulated)"),
                                use_container_width=True)
            m = st.columns(4)
            m[0].metric("shift MAE", f"{met['shift_mae_ppm']:.3f} ppm")
            m[1].metric("J MAE", f"{met['j_mae_hz']:.2f} Hz")
            m[2].metric("presence F1", f"{met['presence_f1']:.2f}")
            m[3].metric("deg acc", f"{met['deg_acc']:.2f}")
            st.caption(f"Hungarian-matched: shift {met['h_shift_mae_ppm']:.3f} ppm · "
                       f"J {met['h_j_mae_hz']:.2f} Hz · deg {met['h_deg_acc']:.2f}")
        else:
            st.info("Press **▶ Run inference** to see the predicted matrix and metrics.")


# =============================================================================
# Router
# =============================================================================

if st.session_state.get("page", "select") == "select":
    page_select()
else:
    page_analysis()
