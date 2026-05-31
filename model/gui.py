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
  Molecules JSON     must be spin_systems_chembl.json (default: mol_to_spin_system/data/spin_systems_chembl.json)
  Spectra root       directory containing 90MHz/mol_*.npy (default: simulation/data/spectra)
  Field (MHz)        90 or 600 MHz for spectrum simulation

Data pipeline correctness
  The test split is reconstructed by running exactly the same pipeline as training:
    data_adapter.load_records(json, spectra_root, fields=(90,), require_spectra=True)
    make_splits(records, seed=checkpoint_seed, compute_scaffold=False)
  Only molecules that HAD a 90 MHz spectrum file during training are included, so
  the split is identical to what the model was evaluated on.  If spin_systems_chembl.json
  is missing locally, a download button appears in the sidebar.

Notes
-----
* Checkpoint files are downloaded to /tmp/spinhance_viewer/<session>/ and reused
  on subsequent loads — no repeated S3 traffic.
* The JSMol widget needs internet access to load from chemapps.stolaf.edu.
* If RDKit is unavailable the 3D widget is replaced with a plain text fallback.
"""
from __future__ import annotations

import base64
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

S3_TRAINING  = "s3://spinhance-data/training"
S3_JSON_60K  = "s3://spinhance-data/spin_systems_chembl.json"
CACHE_DIR    = Path(tempfile.gettempdir()) / "spinhance_viewer"
DEF_JSON     = str(REPO / "mol_to_spin_system/data/spin_systems_chembl.json")
DEF_SPECTRA  = str(REPO / "simulation/data/spectra")

# ── AWS SSO constants (mirrors context/setup_aws_login.sh) ───────────────────
AWS_PROFILE    = "hack-scripps"
AWS_REGION     = "us-west-2"
SSO_SESSION    = "scripps-hackathon"
SSO_START_URL  = "https://d-9267e96a16.awsapps.com/start"
ACCOUNT_ID     = "127696279288"

METRICS_CACHE_V = 2  # bump to invalidate stale disk caches

st.set_page_config(page_title="SpinHance Viewer", layout="wide",
                   initial_sidebar_state="collapsed")

# Suppress Streamlit's whole-page fade/dim during reruns; individual spinners
# placed near active controls serve as loading indicators instead.
st.markdown("""
<style>
[data-stale] { opacity: 1 !important; transition: none !important; }
[data-stale] * { transition: none !important; }
</style>
""", unsafe_allow_html=True)


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
# IMPORTANT: must mirror the training pipeline exactly.
# Training (model/train.sh + model/run_experiment.py) uses:
#   data_adapter.load_records(json, spectra_root, fields=(90,), require_spectra=True)
#   make_splits(records, seed=cfg.seed, compute_scaffold=False)
# Any deviation from this produces a different molecule set → different split.

@st.cache_data(show_spinner="Loading molecules (filtering by 90 MHz spectra)…")
def _load_all_records(json_path: str, spectra_root: str) -> list[dict]:
    """Load records using the same pipeline as training: requires 90 MHz .npy files."""
    from model.data_adapter import load_records
    from model.splits import canonical_order, reorder
    raw = load_records(json_path, spectra_root, fields=(90,), require_spectra=True)
    records = []
    for r in raw:
        order = canonical_order(r["shifts"], r["couplings"], r["degeneracy"])
        shifts, couplings, deg = reorder(r["shifts"], r["couplings"], r["degeneracy"], order)
        records.append({**r, "shifts": shifts, "couplings": couplings, "degeneracy": deg})
    return records


@st.cache_data(show_spinner="Computing test split…")
def _test_records(json_path: str, spectra_root: str, seed: int) -> list[dict]:
    from model.splits import make_splits
    records = _load_all_records(json_path, spectra_root)
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


# ─── Molecule structure viewers ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _mol2d_svg(smiles: str, width: int = 200, height: int = 200) -> str:
    """ChemDraw-style 2D structure as an SVG string."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 1.8
        opts.padding = 0.14
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        return f"<p style='color:#888;font-size:11px;padding:4px;'>2D error: {exc}</p>"


@st.cache_data(show_spinner=False)
def _mol3d_html(smiles: str, width: int = 200, height: int = 200) -> str:
    """Spinning 3D ball-and-stick via py3Dmol (uses 3Dmol.js CDN)."""
    try:
        import py3Dmol
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        view = py3Dmol.view(width=width, height=height)   # keyword args required
        view.addModel(mol_block, "mol")
        view.setStyle({"stick": {"colorscheme": "grayCarbon", "radius": 0.12},
                       "sphere": {"colorscheme": "grayCarbon", "radius": 0.28}})
        view.setBackgroundColor("#f3f4f6")
        view.spin(True)
        view.zoomTo()
        return view._make_html()   # _repr_html_ returns None outside Jupyter
    except Exception as exc:
        return (f"<p style='color:#888;font-size:11px;padding:4px;'>"
                f"3D error: {exc}</p>")


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
        for k in ("mol_pred", "epoch_sel", "loaded_epoch"):
            st.session_state.pop(k, None)
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
        for k in ("epoch_sel", "loaded_epoch"):
            st.session_state.pop(k, None)
        st.rerun()
    title_col.title(f"Session: `{session}`")

    with st.sidebar:
        st.header("Data")

        json_path = st.text_input("Molecules JSON", value=DEF_JSON,
                                   help="Must be spin_systems_chembl.json — the same file used during training")
        spectra_root = st.text_input("Spectra root", value=DEF_SPECTRA,
                                      help="Directory containing 90MHz/mol_*.npy files used to filter the training set")
        field_mhz = st.radio("Simulation field (MHz)", [90, 600], index=0, horizontal=True)

        # ── Data availability checks ──────────────────────────────────────────
        json_ok = Path(json_path).exists()
        spectra_ok = (Path(spectra_root) / "90MHz").exists()

        if not json_ok:
            st.error(f"`{Path(json_path).name}` not found locally.")
            if st.button("⬇  Download from S3", key="dl_json"):
                with st.spinner("Downloading spin_systems_chembl.json…"):
                    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
                    r = subprocess.run(
                        ["aws", "s3", "cp", S3_JSON_60K, json_path,
                         "--profile", AWS_PROFILE, "--region", AWS_REGION],
                        capture_output=True, timeout=120)
                if r.returncode == 0:
                    st.success("Downloaded.")
                    st.rerun()
                else:
                    st.error(r.stderr.decode()[:200])

        if not spectra_ok:
            st.warning("90MHz spectra directory not found — split may not match training.")

        if json_ok and spectra_ok:
            st.success("Data paths OK")

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
    valid_ep_set = {r["epoch"] for r in valid_rows}
    epoch_by_num = {r["epoch"]: r for r in valid_rows}

    # Resolve selected and loaded epochs
    if st.session_state.get("epoch_sel") not in valid_ep_set:
        st.session_state.pop("epoch_sel", None)
    sel_epoch: int = st.session_state.get("epoch_sel", best_epoch)
    loaded_epoch = st.session_state.get("loaded_epoch")
    is_loading = (sel_epoch != loaded_epoch)

    # ── Epoch bar chart (click-only; all pan/zoom/select disabled) ────────────
    st.subheader("Validation score across epochs")
    st.caption("Score = shift_MAE (ppm) + J_MAE (Hz) / 10  ·  lower is better  ·  "
               "click a bar to load that checkpoint")

    xs = [r["epoch"] for r in valid_rows]
    ys = [r["score"] for r in valid_rows]

    # Color scheme:
    #   best (not sel) → red      sel loading → pale yellow      sel loaded → gold
    #   other          → blue
    bar_colors = []
    for r in valid_rows:
        ep = r["epoch"]
        if ep == sel_epoch and is_loading:
            bar_colors.append("#FEF08A")   # pale yellow — clicked, loading
        elif ep == sel_epoch:
            bar_colors.append("#EAB308")   # gold — loaded / active
        elif ep == best_epoch:
            bar_colors.append("#DC2626")   # red — best (not currently selected)
        else:
            bar_colors.append("#93C5FD")   # blue — other

    hover = [
        f"<b>ep{r['epoch']}"
        + (" ★best" if r["epoch"] == best_epoch else "")
        + "</b><br>"
        + f"Score: {r['score']:.4f}<br>"
        + f"Shift MAE: {r['shift_mae_ppm']:.3f} ppm<br>"
        + f"J MAE: {r['j_mae_hz']:.2f} Hz"
        for r in valid_rows
    ]

    fig_bar = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=bar_colors,
        text=[f"{s:.3f}" for s in ys],
        textposition="outside", textfont=dict(size=8),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover,
    ))
    fig_bar.add_annotation(
        x=best_epoch, y=best["score"],
        text=f"★ best ep{best_epoch}",
        showarrow=True, arrowhead=2, arrowsize=1, ax=0, ay=-44,
        font=dict(color="#DC2626", size=11))
    if sel_epoch != best_epoch:
        sel_row = epoch_by_num[sel_epoch]
        label = "loading…" if is_loading else "loaded"
        fig_bar.add_annotation(
            x=sel_epoch, y=sel_row["score"],
            text=f"ep{sel_epoch} ({label})",
            showarrow=True, arrowhead=2, arrowsize=1, ax=0, ay=-28,
            font=dict(color="#854D0E", size=11))
    fig_bar.update_layout(
        xaxis=dict(title="Epoch", dtick=max(1, len(valid_rows) // 20)),
        yaxis=dict(title="Score (lower = better)"),
        height=320, margin=dict(l=50, r=20, t=20, b=40),
        plot_bgcolor="white", showlegend=False,
        dragmode=False,   # disable pan / zoom / box-select / lasso
    )

    event = st.plotly_chart(
        fig_bar, use_container_width=True,
        on_select="rerun", selection_mode=("points",), key="epoch_chart",
        config={"displayModeBar": False, "scrollZoom": False},
    )
    # Apply click immediately (in this render pass) so the rest of the page uses
    # the new sel_epoch; bar colours update on the next rerun.
    pts = event.selection.points if hasattr(event, "selection") else []
    if pts:
        clicked = int(pts[0]["x"])
        if clicked in valid_ep_set and clicked != sel_epoch:
            st.session_state["epoch_sel"] = clicked
            sel_epoch = clicked
            is_loading = (sel_epoch != loaded_epoch)

    sel = epoch_by_num[sel_epoch]

    # ── Lower half: blank out (spinner) while the checkpoint downloads ─────────
    st.divider()
    if is_loading:
        with st.spinner(f"Loading checkpoint for epoch {sel_epoch}…"):
            try:
                model, std, vocab, cfg_dict = _load_model(session, sel_epoch)
            except Exception as exc:
                st.error(f"Failed to load epoch {sel_epoch}: {exc}")
                return
        st.session_state["loaded_epoch"] = sel_epoch
        st.rerun()   # re-render so bar chart turns gold and content appears

    # Model is loaded; fetch from cache (instant)
    model, std, vocab, cfg_dict = _load_model(session, sel_epoch)

    # ── Selected-epoch summary ────────────────────────────────────────────────
    header = f"Epoch {sel_epoch}"
    if sel_epoch == best_epoch:
        header += "  ★ best"
    st.subheader(header)
    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Score", f"{sel['score']:.4f}",
               delta=f"{sel['score'] - best['score']:+.4f}" if sel_epoch != best_epoch else None,
               delta_color="inverse")
    bm2.metric("Shift MAE", f"{sel['shift_mae_ppm']:.3f} ppm")
    bm3.metric("J MAE", f"{sel['j_mae_hz']:.2f} Hz")
    bm4.metric("Presence F1", f"{sel['presence_f1']:.3f}"
               if not np.isnan(sel['presence_f1']) else "—")

    # ── Test set ──────────────────────────────────────────────────────────────
    if not Path(json_path).exists():
        st.error(f"Molecules JSON not found: {json_path}")
        return

    if not Path(spectra_root).exists():
        st.error(f"Spectra root not found: {spectra_root}  — cannot reconstruct training split.")
        return

    seed = int(cfg_dict.get("seed", 0))
    try:
        all_recs = _load_all_records(json_path, spectra_root)
        test_recs = _test_records(json_path, spectra_root, seed)
    except Exception as exc:
        st.error(f"Failed to build test split: {exc}")
        return

    if not test_recs:
        st.warning("No test molecules found.")
        return

    n_all = len(all_recs)
    n_test = len(test_recs)
    expected_pct = 10.0
    actual_pct = 100 * n_test / n_all if n_all else 0
    st.caption(
        f"Test set: **{n_test:,}** / {n_all:,} molecules  ({actual_pct:.1f}% — "
        f"expected ~{expected_pct:.0f}%)  ·  split seed={seed}"
    )

    # ── Molecule selector row: [2D] [3D] | [dropdown + info + button] ──────────
    draw_col, mol3d_col, sel_col = st.columns([1, 1, 3])

    with sel_col:
        mol_idx = st.selectbox(
            "Select molecule",
            range(len(test_recs)),
            format_func=lambda i: (
                f"{test_recs[i]['mol_id']}"
                + (f"  ·  {test_recs[i]['chembl_id']}" if test_recs[i]["chembl_id"] else "")
                + (f"  ·  {test_recs[i]['smiles'][:50]}…" if len(test_recs[i].get("smiles", "")) > 50
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

    with draw_col:
        if rec.get("smiles"):
            svg = _mol2d_svg(rec["smiles"], width=210, height=210)
            b64 = base64.b64encode(svg.encode()).decode()
            st.markdown(
                f'<div style="background:white;border-radius:8px;padding:4px;">'
                f'<img src="data:image/svg+xml;base64,{b64}" '
                f'style="width:100%;display:block;"/></div>',
                unsafe_allow_html=True)
        else:
            st.info("No SMILES")

    with mol3d_col:
        if rec.get("smiles"):
            with st.spinner(""):
                html3d = _mol3d_html(rec["smiles"], width=210, height=210)
            components.html(html3d, height=222, scrolling=False)
        else:
            st.info("No SMILES")

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
