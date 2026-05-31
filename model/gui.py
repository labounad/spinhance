"""
model/gui.py — SpinHance training session viewer
=================================================

A two-page Streamlit dashboard for inspecting EC2 training runs stored on S3.

Quick start
-----------
Run from the repo root (requires the `spinhance` conda env):

    conda run -n spinhance streamlit run model/gui.py

The app opens at http://localhost:8501 in your browser.

AWS credentials
---------------
The app uses the `hack-scripps` SSO profile.  On first launch it writes the
profile stanzas to ~/.aws/config automatically.  If the token has expired you
will see a login gate — click "Login with AWS SSO" (a browser window opens)
or run the helper script manually:

    bash context/setup_aws_login.sh

Tokens last ~8 hours; re-run the script or click the login button when you see
"Token has expired and refresh failed."

Page 1 — Session browser
-------------------------
Lists all training sessions at s3://spinhance-data/training/.  Sessions are
named session001, session002, … in the order they were launched by train.sh.
Select one and click "Open session →".

Page 2 — Session analysis
--------------------------
Epoch bar chart
  Shows the composite validation score (shift_MAE ppm + J_MAE Hz / 10) for
  every saved epoch.  Lower is better; the best epoch is highlighted in red.
  Epoch metrics are cached to /tmp/spinhance_viewer/<session>/epoch_metrics.json
  after the first load so subsequent visits are instant.  Use "↺ Reload metrics"
  to force a fresh pull (e.g. while a run is still in progress).

Best-epoch molecule inspector
  Loads the best epoch's model weights, reconstructs the 70/20/10 test split
  using the training seed stored in the checkpoint, and lets you browse test-set
  molecules.  For each molecule:
    • JSMol rotating 3D structure (requires internet for the St. Olaf CDN)
    • Ground-truth spin matrix and simulated ¹H NMR spectrum
    • Click "▶ Run Inference" to see the model's predicted matrix and spectrum
      side-by-side with ground truth

Sidebar options (Page 2)
  spin_systems.json  path to the molecule dataset (default: mol_to_spin_system/data/)
  Field (MHz)        90 or 600 MHz for spectrum simulation

Notes
-----
* Checkpoint files are downloaded to /tmp/spinhance_viewer/<session>/ and reused
  on subsequent loads — no repeated S3 traffic.
* The JSMol widget needs internet access to load from chemapps.stolaf.edu.
* If RDKit is unavailable the 3D widget is replaced with a plain text fallback.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

S3_TRAINING = "s3://spinhance-data/training"
CACHE_DIR = Path(tempfile.gettempdir()) / "spinhance_viewer"
DEF_JSON = str(REPO / "mol_to_spin_system/data/spin_systems.json")

# ── AWS SSO constants (mirrors context/setup_aws_login.sh) ───────────────────
AWS_PROFILE    = "hack-scripps"
AWS_REGION     = "us-west-2"
SSO_SESSION    = "scripps-hackathon"
SSO_START_URL  = "https://d-9267e96a16.awsapps.com/start"
ACCOUNT_ID     = "127696279288"

METRICS_CACHE_V = 2  # bump to invalidate stale disk caches

st.set_page_config(page_title="SpinHance Viewer", layout="wide",
                   initial_sidebar_state="collapsed")


# ─── AWS credential helpers ───────────────────────────────────────────────────

def _ensure_aws_config() -> None:
    """Write the SSO profile stanzas to ~/.aws/config if they are absent."""
    config = Path.home() / ".aws" / "config"
    config.parent.mkdir(exist_ok=True)
    text = config.read_text() if config.exists() else ""

    additions = ""
    if f"[sso-session {SSO_SESSION}]" not in text:
        additions += f"""
[sso-session {SSO_SESSION}]
sso_start_url = {SSO_START_URL}
sso_region = {AWS_REGION}
sso_registration_scopes = sso:account:access
"""
    if f"[profile {AWS_PROFILE}]" not in text:
        additions += f"""
[profile {AWS_PROFILE}]
sso_session = {SSO_SESSION}
sso_account_id = {ACCOUNT_ID}
sso_role_name = Hackathon
region = {AWS_REGION}
output = json
"""
    if additions:
        with open(config, "a") as f:
            f.write(additions)


def _aws_ok() -> bool:
    """Return True if the SSO token for AWS_PROFILE is currently valid."""
    r = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--profile", AWS_PROFILE],
        capture_output=True, timeout=10)
    return r.returncode == 0


# ─── S3 helpers ───────────────────────────────────────────────────────────────

def _s3_ls(prefix: str) -> list[str]:
    r = subprocess.run(
        ["aws", "s3", "ls", prefix.rstrip("/") + "/",
         "--profile", AWS_PROFILE, "--region", AWS_REGION],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "aws s3 ls returned non-zero")
    return r.stdout.splitlines()


@st.cache_data(ttl=60)
def _list_sessions() -> list[str]:
    lines = _s3_ls(S3_TRAINING)
    sessions = []
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0] == "PRE":
            sessions.append(parts[1].rstrip("/"))
    return sorted(sessions, reverse=True)


@st.cache_data(ttl=60)
def _list_epoch_numbers(session: str) -> list[int]:
    lines = _s3_ls(f"{S3_TRAINING}/{session}")
    epochs = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        fname = parts[-1]
        if fname.startswith("epoch_") and fname.endswith(".pt"):
            try:
                epochs.append(int(fname[6:-3]))
            except ValueError:
                pass
    return sorted(epochs)


def _epoch_local(session: str, epoch: int) -> Path:
    d = CACHE_DIR / session
    d.mkdir(parents=True, exist_ok=True)
    return d / f"epoch_{epoch:03d}.pt"


def _download_epoch(session: str, epoch: int) -> Path:
    local = _epoch_local(session, epoch)
    if not local.exists():
        s3_uri = f"{S3_TRAINING}/{session}/epoch_{epoch:03d}.pt"
        r = subprocess.run(
            ["aws", "s3", "cp", s3_uri, str(local),
             "--profile", AWS_PROFILE, "--region", AWS_REGION],
            capture_output=True, timeout=300)
        if r.returncode != 0:
            local.unlink(missing_ok=True)
            raise RuntimeError(r.stderr.decode()[:300])
    return local


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _rebuild_model(sd: dict):
    from model.model import SpinHanceModel, ResNet1DEncoder
    stem_c = sd["encoder.stem.0.weight"].shape[0]
    head_hidden = sd["shift_head.0.weight"].shape[0]
    G = sd["shift_head.3.weight"].shape[0]
    n_deg = sd["deg_head.3.weight"].shape[0] // G
    if stem_c <= 24:
        enc = ResNet1DEncoder(stem_channels=24, stage_channels=(32, 64, 128, 192),
                              blocks_per_stage=(1, 1, 1, 1))
    else:
        enc = ResNet1DEncoder()
    model = SpinHanceModel(n_groups=G, n_deg_classes=n_deg,
                           encoder=enc, head_hidden=head_hidden, dropout=0.0)
    model.load_state_dict(sd)
    model.eval()
    return model


def _metrics_cache_path(session: str) -> Path:
    return CACHE_DIR / session / "epoch_metrics.json"


def _load_session_metrics(session: str) -> tuple[list[dict], dict[int, str]]:
    """Return (rows, errors).

    rows  — list of {epoch, score, shift_mae_ppm, j_mae_hz, presence_f1, deg_acc}
            sorted by epoch.  NaN values are float('nan').
    errors — {epoch: error_string} for any epochs that failed.

    Strategy:
      1. Read from disk cache if present (instant).
      2. Otherwise download all epoch checkpoints in parallel (6 workers),
         extract just the metrics dict, then write a compact disk cache so the
         next call is instant.
    """
    import torch

    cache = _metrics_cache_path(session)

    # ── Fast path: disk cache ─────────────────────────────────────────────────
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("v") == METRICS_CACHE_V:
                rows = [
                    {k: (float("nan") if v is None else v) for k, v in r.items()}
                    for r in data["rows"]
                ]
                return rows, {}
        except Exception:
            cache.unlink(missing_ok=True)

    # ── Slow path: parallel S3 download ──────────────────────────────────────
    epoch_list = _list_epoch_numbers(session)
    if not epoch_list:
        return [], {}

    def _fetch(ep: int) -> dict:
        local = _download_epoch(session, ep)
        ckpt = torch.load(str(local), map_location="cpu", weights_only=False)
        m = ckpt.get("metrics") or {}
        shift_mae = float(m.get("shift_mae_ppm", float("nan")))
        j_mae     = float(m.get("j_mae_hz",      float("nan")))
        score = (shift_mae + j_mae / 10.0
                 if not (math.isnan(shift_mae) or math.isnan(j_mae))
                 else float("nan"))
        return {
            "epoch":        ep,
            "score":        score,
            "shift_mae_ppm": shift_mae,
            "j_mae_hz":      j_mae,
            "presence_f1":   float(m.get("presence_f1",   float("nan"))),
            "deg_acc":       float(m.get("deg_acc_balanced",
                                         m.get("deg_acc", float("nan")))),
        }

    rows_map: dict[int, dict] = {}
    errors:   dict[int, str]  = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch, ep): ep for ep in epoch_list}
        for fut in as_completed(futures):
            ep = futures[fut]
            try:
                rows_map[ep] = fut.result()
            except Exception as exc:
                errors[ep] = str(exc)

    sorted_rows = [rows_map[ep] for ep in sorted(rows_map)]

    # ── Write compact disk cache (NaN → null for JSON) ────────────────────────
    def _nan_safe(v):
        return None if isinstance(v, float) and math.isnan(v) else v

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "v": METRICS_CACHE_V,
        "rows": [{k: _nan_safe(v) for k, v in r.items()} for r in sorted_rows],
    }))

    return sorted_rows, errors


@st.cache_resource(show_spinner="Loading model weights…")
def _load_model(session: str, epoch: int):
    """Load model + standardizer from a checkpoint. Returns (model, std, vocab, cfg)."""
    import torch
    from model.targets import DegeneracyVocab, Standardizer
    local = _download_epoch(session, epoch)
    ckpt = torch.load(str(local), map_location="cpu", weights_only=False)
    std_d = ckpt["standardizer"]
    std = Standardizer()
    std.shift_mean, std.shift_std = float(std_d["shift_mean"]), float(std_d["shift_std"])
    std.j_mean, std.j_std = float(std_d["j_mean"]), float(std_d["j_std"])
    vocab = DegeneracyVocab()
    model = _rebuild_model(ckpt["model"])
    return model, std, vocab, ckpt.get("cfg") or {}


# ─── Data helpers ─────────────────────────────────────────────────────────────

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
        order = canonical_order(shifts, couplings, degeneracy)
        shifts, couplings, degeneracy = reorder(shifts, couplings, degeneracy, order)
        records.append({
            "mol_id": f"mol_{idx:06d}",
            "chembl_id": rec.get("chembl_id", ""),
            "smiles": rec.get("smiles", ""),
            "shifts": shifts,
            "couplings": couplings,
            "degeneracy": degeneracy,
        })
    return records


@st.cache_data(show_spinner="Computing test split…")
def _test_records(json_path: str, seed: int) -> list[dict]:
    from model.splits import make_splits
    records = _load_records(json_path)
    assignment, _ = make_splits(records, seed=seed, compute_scaffold=False)
    return [r for r in records if assignment.get(r["mol_id"]) == "test"]


@st.cache_data(show_spinner="Simulating spectrum…")
def _simulate(shifts_t: tuple, couplings_t: tuple, degeneracy_t: tuple,
              field_mhz: int) -> tuple[np.ndarray, np.ndarray]:
    from simulation.pyspin.composite import simulate_spectrum_composite
    ppm_axis, intensity = simulate_spectrum_composite(
        np.array(shifts_t), np.array(couplings_t), list(degeneracy_t), float(field_mhz))
    return ppm_axis.astype(np.float64), intensity.astype(np.float64)


def _run_inference(model, intensity: np.ndarray, std, vocab) -> dict:
    import torch
    from model.metrics import decode
    x = torch.from_numpy(intensity.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
    return decode({k: v.float().cpu().numpy() for k, v in pred.items()}, std, vocab)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _fig_spectrum(ppm_axis: np.ndarray, intensity: np.ndarray,
                  title: str = "", color: str = "#2563EB") -> go.Figure:
    fig = go.Figure(go.Scatter(x=ppm_axis, y=intensity, mode="lines",
                               line=dict(color=color, width=1.5)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis=dict(title="δ (ppm)", autorange="reversed",
                   showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        height=220, margin=dict(l=40, r=10, t=32, b=36),
        plot_bgcolor="white", showlegend=False)
    return fig


def _fig_matrix(shifts: np.ndarray, couplings: np.ndarray,
                degeneracy: np.ndarray, title: str = "") -> go.Figure:
    G = len(shifts)
    labels = [f"G{i+1}" for i in range(G)]
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
    max_j = max(float(np.abs(z).max()), 1.0)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-max_j, zmax=max_j,
        text=text, texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="J (Hz)", thickness=12, len=0.8)))
    for i in range(G):
        fig.add_shape(type="rect",
                      x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      fillcolor="rgba(200,200,200,0.3)", line=dict(width=0))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        height=320, margin=dict(l=50, r=50, t=32, b=40),
        xaxis=dict(title="Spin group"), yaxis=dict(title="Spin group"))
    return fig


# ─── JSMol ────────────────────────────────────────────────────────────────────

def _jsmol_html(smiles: str, width: int = 230, height: int = 230) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol_block_json = json.dumps(Chem.MolToMolBlock(mol))
    except Exception as exc:
        return (f"<p style='color:#888;font-size:11px;padding:8px;'>"
                f"3D unavailable: {exc}</p>")

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<script src='https://chemapps.stolaf.edu/jmol/jmol.js'></script>
</head>
<body style='margin:0;padding:0;background:#f3f4f6;'>
<div id='jd'></div>
<script>
Jmol.setDocument(false);
var md = {mol_block_json};
var Info = {{
  width:{width}, height:{height},
  script: "data 'mol'\\n" + md + "\\nend 'mol'\\nspin on; background [243,244,246];",
  use: "HTML5",
  j2sPath: "https://chemapps.stolaf.edu/jmol/j2s",
  disableJ2SLoadMonitor: true, disableInitialConsole: true
}};
document.getElementById('jd').innerHTML = Jmol.getAppletHtml("jsmolApp0", Info);
</script></body></html>"""


# ─── Page 1 — Session browser ─────────────────────────────────────────────────

def _page_select() -> None:
    st.title("SpinHance — Training Session Viewer")
    st.caption(f"S3 prefix: `{S3_TRAINING}`")

    col_btn, _ = st.columns([1, 5])
    if col_btn.button("↺  Refresh"):
        _list_sessions.clear()
        st.rerun()

    try:
        sessions = _list_sessions()
    except Exception as exc:
        st.error(f"Cannot list S3 sessions: {exc}")
        st.info("Ensure AWS credentials are configured and `aws` CLI is on PATH.")
        return

    if not sessions:
        st.warning(f"No sessions found at `{S3_TRAINING}`.")
        return

    selected = st.selectbox("Training session", sessions)

    if st.button("Open session →", type="primary"):
        st.session_state["session"] = selected
        st.session_state["page"] = "analysis"
        st.session_state.pop("mol_pred", None)
        st.rerun()


# ─── Page 2 — Session analysis ────────────────────────────────────────────────

def _page_analysis() -> None:
    session: str = st.session_state.get("session", "")
    if not session:
        st.session_state["page"] = "select"
        st.rerun()

    # ── Nav + title ───────────────────────────────────────────────────────────
    nav_col, title_col = st.columns([1, 8])
    if nav_col.button("← Sessions"):
        st.session_state["page"] = "select"
        st.rerun()
    title_col.title(f"Session: `{session}`")

    with st.sidebar:
        st.header("Data")
        json_path = st.text_input("spin_systems.json", value=DEF_JSON)
        field_mhz = st.radio("Field (MHz)", [90, 600], index=0, horizontal=True)
        st.divider()
        st.caption("Checkpoints cached to\n`" + str(CACHE_DIR / session) + "`")

    # ── Epoch list ────────────────────────────────────────────────────────────
    try:
        epoch_list = _list_epoch_numbers(session)
    except Exception as exc:
        st.error(f"Cannot list epochs: {exc}")
        return

    if not epoch_list:
        st.warning("No `epoch_XXX.pt` files found in this session.")
        return

    # ── Load metrics (disk cache → parallel S3 download) ─────────────────────
    cached = _metrics_cache_path(session).exists()
    spinner_msg = ("Reading cached metrics…" if cached
                   else f"Downloading {len(epoch_list)} epoch checkpoints in parallel…")
    with st.spinner(spinner_msg):
        try:
            rows, errors = _load_session_metrics(session)
        except Exception as exc:
            st.error(f"Failed to load session metrics: {exc}")
            return

    n_ep = len(epoch_list)
    n_loaded = len(rows)
    cache_note = " (cached)" if cached else f" — {n_loaded}/{n_ep} loaded"
    st.caption(f"{n_ep} epoch checkpoints: ep{epoch_list[0]} → ep{epoch_list[-1]}{cache_note}")

    reload_col, _ = st.columns([1, 6])
    if reload_col.button("↺  Reload metrics"):
        _metrics_cache_path(session).unlink(missing_ok=True)
        st.rerun()

    if errors:
        with st.expander(f"{len(errors)} epoch error(s)", expanded=False):
            st.text("\n".join(f"ep{ep}: {err}" for ep, err in sorted(errors.items())))

    valid_rows = [r for r in rows if not np.isnan(r["score"])]
    if not valid_rows:
        st.error("No epochs with valid validation metrics could be loaded.")
        return

    best = min(valid_rows, key=lambda r: r["score"])
    best_epoch: int = best["epoch"]

    # ── Epoch bar chart ───────────────────────────────────────────────────────
    st.subheader("Validation score across epochs")
    st.caption("Score = shift_MAE (ppm) + J_MAE (Hz) / 10  ·  lower is better  ·  "
               "best epoch highlighted in red")

    xs = [r["epoch"] for r in valid_rows]
    ys = [r["score"] for r in valid_rows]
    bar_colors = ["#DC2626" if r["epoch"] == best_epoch else "#93C5FD"
                  for r in valid_rows]

    fig_bar = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=bar_colors,
        text=[f"{s:.3f}" for s in ys],
        textposition="outside", textfont=dict(size=8)))
    fig_bar.add_annotation(
        x=best_epoch, y=best["score"],
        text=f"Best ep {best_epoch}",
        showarrow=True, arrowhead=2, arrowsize=1, ax=0, ay=-44,
        font=dict(color="#DC2626", size=11))
    fig_bar.update_layout(
        xaxis=dict(title="Epoch", dtick=max(1, len(valid_rows) // 20)),
        yaxis=dict(title="Score (lower = better)"),
        height=320, margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Best-epoch summary ────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Best epoch: {best_epoch}")
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Score", f"{best['score']:.4f}")
    bm2.metric("Shift MAE", f"{best['shift_mae_ppm']:.3f} ppm")
    bm3.metric("J MAE", f"{best['j_mae_hz']:.2f} Hz")
    bm4.metric("Presence F1", f"{best['presence_f1']:.3f}"
               if not np.isnan(best['presence_f1']) else "—")

    # ── Load best-epoch model ─────────────────────────────────────────────────
    try:
        model, std, vocab, cfg_dict = _load_model(session, best_epoch)
    except Exception as exc:
        st.error(f"Failed to load epoch {best_epoch} model: {exc}")
        return

    # ── Test set ──────────────────────────────────────────────────────────────
    if not Path(json_path).exists():
        st.error(f"spin_systems.json not found: {json_path}")
        return

    seed = int(cfg_dict.get("seed", 0))
    try:
        test_recs = _test_records(json_path, seed)
    except Exception as exc:
        st.error(f"Failed to build test split: {exc}")
        return

    if not test_recs:
        st.warning("No test molecules found.")
        return

    st.caption(f"Test set: {len(test_recs)} molecules (split seed={seed})")

    # ── Molecule selector row: [JSMol | dropdown + info + button] ─────────────
    jsmol_col, sel_col = st.columns([1, 3])

    with sel_col:
        mol_idx = st.selectbox(
            "Select molecule (SMILES)",
            range(len(test_recs)),
            format_func=lambda i: (
                f"{test_recs[i]['mol_id']}"
                + (f"  ·  {test_recs[i]['chembl_id']}" if test_recs[i]["chembl_id"] else "")
                + (f"  ·  {test_recs[i]['smiles'][:55]}…" if len(test_recs[i].get("smiles", "")) > 55
                   else f"  ·  {test_recs[i]['smiles']}" if test_recs[i].get("smiles") else "")
            ),
            key="mol_selector",
        )
        rec = test_recs[mol_idx]
        info_parts = [
            f"**Spin groups:** {len(rec['shifts'])}",
            f"**Total spins:** {int(rec['degeneracy'].sum())}",
        ]
        if rec.get("chembl_id"):
            info_parts.insert(0, f"**ChEMBL:** `{rec['chembl_id']}`")
        st.caption("  ·  ".join(info_parts))
        if rec.get("smiles"):
            st.code(rec["smiles"], language=None)

        if st.button("▶  Run Inference", type="primary", key="run_inf"):
            with st.spinner("Running inference…"):
                ppm, intens = _simulate(
                    tuple(rec["shifts"].tolist()),
                    tuple(map(tuple, rec["couplings"].tolist())),
                    tuple(rec["degeneracy"].tolist()),
                    field_mhz)
                dec = _run_inference(model, intens, std, vocab)
            st.session_state["mol_pred"] = {
                "dec": dec, "ppm": ppm, "intens": intens,
                "mol_id": rec["mol_id"], "field": field_mhz,
            }

    with jsmol_col:
        if rec.get("smiles"):
            components.html(_jsmol_html(rec["smiles"]), height=242, scrolling=False)
        else:
            st.info("No SMILES available")

    # ── Ground truth + prediction panes ──────────────────────────────────────
    rec = test_recs[mol_idx]  # re-read in case it changed
    pred_state = st.session_state.get("mol_pred")
    pred_ready = (pred_state is not None
                  and pred_state["mol_id"] == rec["mol_id"]
                  and pred_state["field"] == field_mhz)

    ppm, intens = _simulate(
        tuple(rec["shifts"].tolist()),
        tuple(map(tuple, rec["couplings"].tolist())),
        tuple(rec["degeneracy"].tolist()),
        field_mhz)

    col_gt, col_pred = st.columns(2)

    with col_gt:
        st.markdown("#### Ground truth")
        st.plotly_chart(
            _fig_matrix(rec["shifts"], rec["couplings"], rec["degeneracy"],
                        title="GT matrix  (diag = δ ppm · off-diag = J Hz)"),
            use_container_width=True)
        st.plotly_chart(
            _fig_spectrum(ppm, intens, title=f"GT spectrum  ({field_mhz} MHz)"),
            use_container_width=True)

    with col_pred:
        st.markdown("#### Model prediction")
        if pred_ready:
            dec = pred_state["dec"]
            p_shifts = dec["shifts"][0]
            p_coup = dec["couplings"][0]
            p_deg = dec["degeneracy"][0]
            pred_ppm, pred_int = _simulate(
                tuple(p_shifts.tolist()),
                tuple(map(tuple, p_coup.tolist())),
                tuple(p_deg.tolist()),
                field_mhz)
            st.plotly_chart(
                _fig_matrix(p_shifts, p_coup, p_deg,
                            title="Predicted matrix"),
                use_container_width=True)
            st.plotly_chart(
                _fig_spectrum(pred_ppm, pred_int,
                              title=f"Predicted spectrum  ({field_mhz} MHz)",
                              color="#DC2626"),
                use_container_width=True)
        else:
            st.info("Press **▶ Run Inference** to see the model output.")


# ─── Page 0 — AWS login gate ─────────────────────────────────────────────────

def _page_aws_login() -> None:
    st.title("AWS Login Required")
    st.warning(
        f"The `{AWS_PROFILE}` SSO session has expired or is not yet active.  "
        "Click **Login** to open the Scripps SSO browser flow, then come back "
        "and click **Refresh**.")

    col_login, col_refresh, _ = st.columns([1, 1, 4])

    if col_login.button("🔑  Login with AWS SSO", type="primary"):
        with st.spinner("Starting SSO login — a browser window will open…"):
            r = subprocess.run(
                ["aws", "sso", "login", "--profile", AWS_PROFILE],
                capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Login command failed:")
            st.code(r.stderr or r.stdout, language=None)

    if col_refresh.button("↺  Refresh"):
        st.rerun()

    with st.expander("Manual login", expanded=False):
        st.code(
            f"# Run this in a terminal, then click Refresh above\n"
            f"bash {REPO / 'context' / 'setup_aws_login.sh'}",
            language="bash")


# ─── Router ───────────────────────────────────────────────────────────────────

# Bootstrap: write the SSO config stanzas to ~/.aws/config if absent.
# This is always safe and requires no user interaction.
_ensure_aws_config()

# Gate: if the SSO token is expired, show the login page instead.
if not _aws_ok():
    _page_aws_login()
    st.stop()

if st.session_state.get("page", "select") == "select":
    _page_select()
else:
    _page_analysis()
