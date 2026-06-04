"""
model.diagnostics.live_dashboard
================================
Streamlit dashboard for monitoring the SpinHance training fleet. Reads the
canonical run layout (status.json / metrics.jsonl / summary.json / probes/)
directly from a **runs directory** (default ``model/runs/``) via
``autoai.run_reader`` — so it works on the HPC's GPFS runs (rsync them locally
or run with an SSH port-forward) as well as S3 sessions.

Multi-run by design: a fleet comparison table + overlaid learning curves across
every matching run, plus a per-run detail view. Styled to match the project
website (dark palette, accent gradient, mono numerals).

Usage:
    streamlit run model/diagnostics/live_dashboard.py
    SPINHANCE_RUNS=/path/to/runs streamlit run model/diagnostics/live_dashboard.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autoai import run_reader as rr  # noqa: E402

st.set_page_config(page_title="SpinHance · training fleet", layout="wide",
                   initial_sidebar_state="expanded")

# ── Website palette (dark theme) ────────────────────────────────────────────────
BG, PANEL, PANEL2 = "#07090d", "#0f131b", "#121826"
INK, INK_SOFT, INK_FAINT = "#eef1f6", "#aab2c2", "#6a7385"
LINE, LINE_STRONG = "#1c2330", "#2a3344"
ACCENT, ACCENT2, ACCENT3 = "#5b8cff", "#34e3c4", "#b07bff"
# categorical run colors (mirror the website's learning-curve series)
PALETTE = [ACCENT, ACCENT3, ACCENT2, "#e0a44d", "#7ee06b", "#e06b6b",
           "#6bd0e0", "#c46be0", "#f5a623", "#9aa0a6"]
MONO = '"SF Mono","JetBrains Mono",ui-monospace,Menlo,Consolas,monospace'

st.markdown(f"""
<style>
  .stApp {{ background:{BG}; }}
  /* gradient page title */
  .sh-title {{ font:700 26px/1.1 {MONO}; letter-spacing:-.01em;
    background:linear-gradient(90deg,{ACCENT},{ACCENT3});
    -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent; }}
  .sh-eyebrow {{ font:600 12px {MONO}; letter-spacing:.18em; text-transform:uppercase; color:{ACCENT2}; }}
  .sh-sub {{ color:{INK_FAINT}; font:13px {MONO}; }}
  h1,h2,h3 {{ font-family:{MONO}; color:{INK}; }}
  /* metric cards */
  [data-testid="stMetric"] {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px;
    padding:10px 14px; }}
  [data-testid="stMetricValue"] {{ font-family:{MONO}; color:{INK}; }}
  [data-testid="stMetricLabel"] {{ color:{INK_FAINT}; }}
  /* fleet comparison table (mirrors the website .cmp) */
  table.cmp {{ width:100%; border-collapse:collapse; font:13px {MONO}; }}
  table.cmp th {{ text-align:left; padding:7px 10px; border-bottom:2px solid {LINE_STRONG};
    color:{INK_SOFT}; font-weight:700; white-space:nowrap; }}
  table.cmp td {{ padding:7px 10px; border-bottom:1px solid {LINE}; color:{INK_SOFT}; }}
  table.cmp td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  table.cmp td.best {{ font-weight:800; background:linear-gradient(90deg,{ACCENT},{ACCENT3});
    -webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent; }}
  table.cmp td.run {{ color:{INK}; }}
  .chip {{ display:inline-block; padding:1px 8px; border-radius:999px; font:600 11px {MONO}; }}
  .chip.running {{ color:{ACCENT2}; border:1px solid {ACCENT2}55; background:{ACCENT2}14; }}
  .chip.finished {{ color:{ACCENT}; border:1px solid {ACCENT}55; background:{ACCENT}14; }}
  .chip.failed,.chip.unknown {{ color:#e06b6b; border:1px solid #e06b6b55; background:#e06b6b14; }}
</style>
""", unsafe_allow_html=True)


def themed(fig, height=300):
    fig.update_layout(
        height=height, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(color=INK_SOFT, family="SF Mono, monospace", size=12),
        margin=dict(t=42, b=28, l=48, r=14), hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h",
                    y=1.12, x=0),
        title=dict(font=dict(color=INK, size=13)))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE_STRONG)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE_STRONG)
    return fig


# ── run helpers ─────────────────────────────────────────────────────────────────

def _val_series(run_dir, key):
    """(epochs, values) for a val metric across a run's metrics.jsonl."""
    xs, ys = [], []
    for r in rr.read_metrics(run_dir):
        if r.get("split") != "val":
            continue
        m = r.get("metrics", {})
        if key in m and m[key] is not None:
            xs.append(r.get("epoch", len(xs)))
            ys.append(m[key])
    return xs, ys


def _train_series(run_dir, key):
    xs, ys = [], []
    for r in rr.read_metrics(run_dir):
        if r.get("split") != "train_step":
            continue
        m = r.get("metrics", {})
        if key in m and m[key] is not None:
            xs.append(r.get("step", len(xs)))
            ys.append(m[key])
    return xs, ys


def _label(run_id: str) -> str:
    """Strip the timestamp prefix + 6-hex hash suffix → e.g. v2_026_500k."""
    s = re.sub(r"^\d+_(?:\d+_)?", "", run_id)     # drop leading timestamp(s)
    s = re.sub(r"_[0-9a-f]{6}$", "", s)           # drop trailing run hash
    return s or run_id


# ── sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sh-eyebrow">SpinHance</div>', unsafe_allow_html=True)
    default_root = os.environ.get("SPINHANCE_RUNS", str(rr.RUNS_ROOT))
    runs_root = st.text_input("Runs directory", default_root,
                              help="Local path to the runs/ dir (rsync from HPC, or an s3:// session).")
    runs = rr.list_runs(runs_root)
    flt = st.text_input("Filter", "rebuild", help="Substring match on run id (blank = all). "
                        "Defaults to the corrected-data 'rebuild_*' fleet; use 'v2_' for the "
                        "pre-regeneration runs.")
    if flt.strip():
        runs = [d for d in runs if flt.strip() in d.name]
    hide_cancelled = st.toggle("Hide cancelled", value=True,
                               help="Drop runs whose status is cancelled (superseded / "
                                    "scancel'd re-runs whose frozen curves would otherwise show).")
    if hide_cancelled:   # filter BEFORE collapse so a cancelled 'latest' falls back to a live run
        runs = [d for d in runs if rr.read_status(d).get("state") != "cancelled"]
    collapse = st.toggle("Latest run per config", value=True,
                         help="Show only the most recent run dir for each config·tier "
                              "(hides superseded / cancelled re-runs).")
    if collapse:
        # keep the lexically-greatest dir name per label = the newest timestamp;
        # robust to filesystem mtime (rsync'd copies) since the name carries the stamp.
        best = {}
        for d in runs:
            lab = _label(d.name)
            if lab not in best or d.name > best[lab].name:
                best[lab] = d
        runs = sorted(best.values(), key=lambda d: d.name, reverse=True)
    labels = {str(d): _label(d.name) for d in runs}
    st.caption(f"{len(runs)} run(s)")
    live = st.toggle("Auto-refresh (10 s)", value=True)
    interval = "10s" if live else None


# ── dashboard ───────────────────────────────────────────────────────────────────

VAL_METRICS = [
    ("shift_mae_ppm", "Shift MAE (ppm)", True),
    ("j_mae_hz", "J MAE (Hz)", True),
    ("presence_f1", "Presence F1", False),
    ("deg_acc_balanced", "Degeneracy acc (bal.)", False),
]


def _fleet_table(runs):
    rows = []
    for d in runs:
        a = rr.analyze_run(d)
        bm = a.get("best_metrics", {}) or {}
        rows.append({
            "run": labels[str(d)], "state": a.get("state", "unknown"),
            "epoch": rr.read_status(d).get("epoch", "—"),
            "shift": bm.get("shift_mae_ppm"), "j": bm.get("j_mae_hz"),
            "f1": bm.get("presence_f1"), "deg": bm.get("deg_acc_balanced"),
        })
    if not rows:
        st.info("No runs found. Point the sidebar at a runs directory (rsync the HPC's model/runs/).")
        return
    # best per metric column
    def _best(k, lower):
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        return (min if lower else max)(vals) if vals else None
    best = {"shift": _best("shift", True), "j": _best("j", True),
            "f1": _best("f1", False), "deg": _best("deg", False)}
    hdr = "<tr><th>run</th><th>state</th><th>epoch</th><th>shift MAE</th><th>J MAE</th><th>F1</th><th>deg-bal</th></tr>"
    body = ""
    for r in rows:
        def cell(k, fmt):
            v = r[k]
            if not isinstance(v, (int, float)):
                return '<td class="num">—</td>'
            cls = "num best" if best[k] is not None and abs(v - best[k]) < 1e-9 else "num"
            return f'<td class="{cls}">{fmt.format(v)}</td>'
        body += (f'<tr><td class="run">{r["run"]}</td>'
                 f'<td><span class="chip {r["state"]}">{r["state"]}</span></td>'
                 f'<td class="num">{r["epoch"]}</td>'
                 f'{cell("shift","{:.3f}")}{cell("j","{:.2f}")}'
                 f'{cell("f1","{:.3f}")}{cell("deg","{:.3f}")}</tr>')
    st.markdown(f'<table class="cmp">{hdr}{body}</table>', unsafe_allow_html=True)


def _heldout_table(runs):
    """Standardized eval: every model scored on the SAME global held-out test pool
    (eval_heldout.py → <run>/heldout_eval.json). Directly comparable across tiers."""
    rows = []
    for d in runs:
        h = rr.read_heldout_eval(d)
        if h and h.get("metrics"):
            m = h["metrics"]
            rows.append({"run": labels[str(d)], "n": h.get("n_test", "—"),
                         "shift": m.get("shift_mae_ppm"), "j": m.get("j_mae_hz"),
                         "f1": m.get("presence_f1"), "deg": m.get("deg_acc_balanced")})
    if not rows:
        st.markdown(
            f'<span class="sh-sub">No held-out eval yet. After a model finishes, score it on the '
            f'shared 315k test pool:</span>', unsafe_allow_html=True)
        st.code("python -m model.experiments.eval_heldout --run-dir model/runs/<id> \\\n"
                "  --test-records $REBUILD/records_3M_test.json.gz --parts $REBUILD/parts --device cuda",
                language="bash")
        return
    sizes = {r["n"] for r in rows if isinstance(r["n"], int)}
    note = f"all scored on the shared held-out test pool" + (f" (n={max(sizes)})" if sizes else "")
    st.markdown(f'<span class="sh-sub">{note}</span>', unsafe_allow_html=True)

    def _best(k, lower):
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        return (min if lower else max)(vals) if vals else None
    best = {"shift": _best("shift", True), "j": _best("j", True),
            "f1": _best("f1", False), "deg": _best("deg", False)}
    hdr = "<tr><th>run</th><th>n</th><th>shift MAE</th><th>J MAE</th><th>F1</th><th>deg-bal</th></tr>"
    body = ""
    for r in rows:
        def cell(k, fmt):
            v = r[k]
            if not isinstance(v, (int, float)):
                return '<td class="num">—</td>'
            cls = "num best" if best[k] is not None and abs(v - best[k]) < 1e-9 else "num"
            return f'<td class="{cls}">{fmt.format(v)}</td>'
        body += (f'<tr><td class="run">{r["run"]}</td><td class="num">{r["n"]}</td>'
                 f'{cell("shift","{:.3f}")}{cell("j","{:.2f}")}'
                 f'{cell("f1","{:.3f}")}{cell("deg","{:.3f}")}</tr>')
    st.markdown(f'<table class="cmp">{hdr}{body}</table>', unsafe_allow_html=True)


def _curves(runs):
    import plotly.graph_objects as go
    color = {str(d): PALETTE[i % len(PALETTE)] for i, d in enumerate(runs)}
    grid = st.columns(2)
    for i, (key, title, _lower) in enumerate(VAL_METRICS):
        fig = go.Figure()
        any_data = False
        for d in runs:
            xs, ys = _val_series(d, key)
            if xs:
                any_data = True
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                              name=labels[str(d)], line=dict(color=color[str(d)], width=2),
                              marker=dict(size=4)))
        if not any_data:
            continue
        fig.update_layout(title=title)
        grid[i % 2].plotly_chart(themed(fig), width="stretch")


def _detail(d):
    import plotly.graph_objects as go
    status = rr.read_status(d)
    st.markdown(f"#### {labels[str(d)]}  ·  <span class='sh-sub'>{Path(d).name}</span>",
                unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("State", status.get("state", "—"))
    c[1].metric("Epoch", f"{status.get('epoch','?')} / {status.get('epochs','?')}")
    c[2].metric("Stage", status.get("stage", "?"))
    bs = status.get("best_score")
    c[3].metric("Best score", f"{bs:.4f}" if isinstance(bs, (int, float)) else "—")
    c[4].metric("Step", f"{status.get('global_step','?')}")
    c[5].metric("Device", status.get("device", "?"))

    # training loss + LR
    lx, ly = _train_series(d, "loss_total")
    if lx:
        cols = st.columns(2)
        f = go.Figure(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=ACCENT, width=1.6)))
        f.update_layout(title="Training loss"); cols[0].plotly_chart(themed(f, 240), width="stretch")
        rx, ry = _train_series(d, "lr")
        if rx:
            g = go.Figure(go.Scatter(x=rx, y=ry, mode="lines", line=dict(color=ACCENT3, width=1.6)))
            g.update_layout(title="Learning rate"); cols[1].plotly_chart(themed(g, 240), width="stretch")
    else:
        st.info("Waiting for training metrics… (run may still be loading/preloading).")

    # failure analysis (latest probe epoch)
    fs = rr.read_failure_summary(d)
    if fs:
        st.markdown("###### Failure analysis")
        dom = fs.get("dominant_failure", "—")
        nmol = fs.get("n_molecules", 0) or 0
        n_ok = fs.get("n_ok", 0)
        cc = st.columns(3)
        # "healthy" = no failures; show the leading failure's share otherwise
        if dom in ("healthy", "none", "ok"):
            cc[0].metric("Dominant failure", "none ✓")
        else:
            frac = fs.get("dominant_failure_frac")
            cc[0].metric("Dominant failure", dom,
                         delta=(f"{frac:.0%} of mols" if frac else None), delta_color="off")
        ok_pct = f"{(n_ok / nmol):.0%}" if nmol else "—"
        cc[1].metric("Healthy molecules", f"{n_ok}/{nmol}", delta=ok_pct, delta_color="off")
        cc[2].metric("Failing", fs.get("n_failing", "?"))
        # distribution chart: drop the (dominant) "ok" bar so failures are visible
        fd = {k: v for k, v in fs.get("failure_distribution", {}).items() if k != "ok"}
        if fd:
            f = go.Figure(go.Bar(x=list(fd.keys()), y=list(fd.values()),
                          marker_color=ACCENT))
            f.update_layout(title="Failure modes (excl. healthy)")
            st.plotly_chart(themed(f, 240), width="stretch")


def _dashboard():
    st.markdown('<div class="sh-eyebrow">training fleet</div>'
                '<div class="sh-title">SpinHance · run monitor</div>', unsafe_allow_html=True)
    if not runs:
        st.info("No runs match. Set the runs directory + filter in the sidebar.")
        return
    st.markdown("### Fleet")
    _fleet_table(runs)
    st.markdown("### Standardized held-out eval")
    _heldout_table(runs)
    st.markdown("### Learning curves")
    _curves(runs)
    st.markdown("### Run detail")
    sel = st.selectbox("Run", [str(d) for d in runs], format_func=lambda s: labels[s])
    _detail(Path(sel) if not str(sel).startswith("s3://") else sel)


try:
    @st.fragment(run_every=interval)  # type: ignore[call-arg]
    def _live():
        _dashboard()
    _live()
except TypeError:
    _dashboard()
    if live:
        st.caption("⚠ Auto-refresh needs Streamlit ≥ 1.33 — static view.")
