"""
model.live_dashboard
====================
Streamlit live dashboard for monitoring active or completed training runs.
Reads status.json + metrics.jsonl from a run directory produced by
model/diagnostics.py; auto-refreshes every 5 s when live mode is on.

Usage:
    streamlit run model/live_dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO / "model" / "runs"

st.set_page_config(page_title="SpinHance Training", layout="wide")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def _list_runs() -> list[str]:
    if not RUNS_ROOT.exists():
        return []
    return sorted([d.name for d in RUNS_ROOT.iterdir() if d.is_dir()], reverse=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("SpinHance Training")
    run_options = _list_runs()
    if run_options:
        run_name = st.selectbox("Run", run_options)
        run_dir  = RUNS_ROOT / run_name
    else:
        raw = st.text_input("Run directory", str(RUNS_ROOT / "latest"))
        run_dir = Path(raw)
    live     = st.toggle("Auto-refresh (5 s)", value=True)
    interval = "5s" if live else None
    st.caption(f"`{run_dir.name}`")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _line(df: pd.DataFrame, x: str, y: str, title: str):
    try:
        import plotly.express as px
        fig = px.line(df, x=x, y=y, title=title, markers=True)
        fig.update_layout(height=260, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart(df.set_index(x)[[y]])


# ── Main dashboard ────────────────────────────────────────────────────────────

def _dashboard():
    status     = _read_json(run_dir / "status.json", {})
    metrics_df = _read_jsonl(run_dir / "metrics.jsonl")

    # Header
    state = status.get("state", "—")
    icon  = {"running": "🟢", "finished": "🔵"}.get(state, "🔴")
    st.subheader(f"{icon}  {run_dir.name}")

    c = st.columns(6)
    c[0].metric("State",   state)
    c[1].metric("Epoch",   f"{status.get('epoch','?')} / {status.get('epochs','?')}")
    c[2].metric("Stage",   status.get("stage", "?"))
    best = status.get("best_score")
    c[3].metric("Best",    f"{best:.4f}" if isinstance(best, (int, float)) else "—")
    c[4].metric("Device",  status.get("device", "?"))
    c[5].metric("Step",    status.get("global_step", "?"))

    if metrics_df.empty:
        st.info("Waiting for training metrics…")
        return

    # Filter splits
    def _split(name):
        if "split" not in metrics_df.columns:
            return pd.DataFrame()
        return metrics_df[metrics_df["split"] == name].copy()

    val_df   = _split("val")
    train_df = _split("train_step")

    # Validation curves
    if not val_df.empty:
        st.subheader("Validation")
        key_metrics = [
            ("metrics.shift_mae_ppm",     "Shift MAE (ppm)"),
            ("metrics.j_mae_hz",          "J MAE (Hz)"),
            ("metrics.h_shift_mae_ppm",   "Hungarian shift MAE (ppm)"),
            ("metrics.presence_f1",       "Presence F1"),
            ("metrics.deg_acc_balanced",  "Degeneracy acc (balanced)"),
        ]
        available = [(col, lbl) for col, lbl in key_metrics if col in val_df.columns]
        cols = st.columns(min(3, max(1, len(available))))
        for i, (col, lbl) in enumerate(available):
            with cols[i % len(cols)]:
                _line(val_df, "epoch", col, lbl)

    # Train step loss
    if not train_df.empty and "metrics.loss_total" in train_df.columns:
        st.subheader("Training loss")
        col_a, col_b = st.columns(2)
        with col_a:
            _line(train_df, "step", "metrics.loss_total", "Total loss")
        if "metrics.lr" in train_df.columns:
            with col_b:
                _line(train_df, "step", "metrics.lr", "Learning rate")

    # Stage / curriculum weights
    if "metrics.w_spec" in train_df.columns:
        st.subheader("Curriculum weights")
        col_a, col_b = st.columns(2)
        with col_a:
            _line(train_df, "step", "metrics.w_mat", "w_mat (matrix anchor)")
        with col_b:
            _line(train_df, "step", "metrics.w_spec", "w_spec (spectral loss)")

    # GPU health
    if "metrics.cuda_allocated_gb" in train_df.columns:
        st.subheader("GPU memory")
        cols = st.columns(2)
        with cols[0]:
            _line(train_df, "step", "metrics.cuda_allocated_gb", "Allocated (GB)")
        with cols[1]:
            _line(train_df, "step", "metrics.cuda_reserved_gb",  "Reserved (GB)")

    # Best metrics table
    summary = _read_json(run_dir / "summary.json")
    if summary and "best_metrics" in summary:
        st.subheader("Best metrics")
        bm = summary["best_metrics"]
        st.dataframe(
            pd.DataFrame([{"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                          for k, v in bm.items()]),
            use_container_width=True, hide_index=True,
        )
        if summary.get("failure_summary", {}).get("dominant_failure"):
            st.info(f"Dominant failure: **{summary['failure_summary']['dominant_failure']}**  "
                    f"— {summary.get('recommendation', '')}")

    # Probe inspector
    probes_dir = run_dir / "probes"
    if probes_dir.exists():
        epoch_dirs = sorted([d for d in probes_dir.iterdir() if d.is_dir()], reverse=True)
        if epoch_dirs:
            st.subheader("Probe diagnostics")
            sel = st.selectbox("Probe epoch", [d.name for d in epoch_dirs])
            ep_dir = probes_dir / sel

            pm = _read_json(ep_dir / "probe_metrics.json", {})
            if pm:
                pc = st.columns(len(pm))
                for i, (k, v) in enumerate(pm.items()):
                    pc[i].metric(k.replace("_", " "), f"{v:.3f}")

            worst = _read_json(ep_dir / "worst_cases.json", [])
            if worst:
                st.write("Worst probe cases (by shift MAE)")
                wdf = pd.DataFrame([{
                    "mol_id":    m["mol_id"],
                    "shift_mae": f"{m.get('shift_mae_ppm', 0):.3f}",
                    "j_mae":     f"{m.get('j_mae_hz', 0):.2f}",
                    "pres_f1":   f"{m.get('presence_f1', 0):.2f}",
                    "deg_acc":   f"{m.get('deg_acc', 0):.2f}",
                } for m in worst[:8]])
                st.dataframe(wdf, use_container_width=True, hide_index=True)

            imgs = sorted(ep_dir.glob("matrix_*.png"))
            if imgs:
                n_show = min(6, len(imgs))
                st.write(f"Matrix plots ({len(imgs)} probes, showing {n_show})")
                img_cols = st.columns(3)
                for i, img in enumerate(imgs[:n_show]):
                    img_cols[i % 3].image(str(img), width="stretch")

            fsummary = _read_json(ep_dir / "failure_summary.json")
            if fsummary:
                st.subheader("Failure analysis")
                col_a, col_b = st.columns(2)
                col_a.metric("Dominant failure", fsummary.get("dominant_failure", "—"))
                col_b.metric("OK molecules", fsummary.get("n_ok", "?"))
                fd = fsummary.get("failure_distribution", {})
                if fd:
                    st.bar_chart(pd.Series(fd, name="count"))


# ── Fragment (auto-refresh if supported) ──────────────────────────────────────

try:
    @st.fragment(run_every=interval)  # type: ignore[call-arg]
    def _live():
        _dashboard()
    _live()
except TypeError:
    # Older Streamlit — run_every not supported
    _dashboard()
    if live:
        st.caption("⚠ Auto-refresh requires Streamlit ≥ 1.33 — showing static view.")
