"""
model/gui.py — SpinHance training-session viewer (rebuilt-trainer edition)
==========================================================================

Two-page Streamlit dashboard for inspecting EC2 training runs on S3, adapted to
the rebuilt `model/` package and its run-directory artifact contract.

    AWS_PROFILE=hack-scripps streamlit run model/gui.py
    # (or just `streamlit run model/gui.py` — the app writes the SSO profile and
    #  shows a login gate if the token is expired)

Differences from the legacy viewer:
  * Reads the epoch score curve from `metrics.jsonl` (the rebuilt trainer saves
    only best.pt / last.pt, not per-epoch checkpoints) — fast, one small file.
  * Rebuilds the model from the checkpoint's stored config via the architecture
    registry (handles resnet1d / resnet1d_attention_pool / any future model).
  * Resolves both run layouts: legacy flat `<session>/...` and rebuilt nested
    `<session>/runs/<run_id>/...`.

Page 1 lists sessions; Page 2 shows the validation-score curve and a test-set
molecule inspector (2D/3D structure, GT vs predicted matrix + spectrum) using the
best (or last) checkpoint.
"""
from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

S3_TRAINING = "s3://spinhance-data/training"
S3_JSON = "s3://spinhance-data/spin_systems_chembl.json"
CACHE_DIR = Path(tempfile.gettempdir()) / "spinhance_viewer"
DEF_JSON = str(REPO / "mol_to_spin_system/data/spin_systems_chembl.json")
DEF_SPECTRA = str(REPO / "simulation/data/spectra")

AWS_PROFILE = "hack-scripps"
AWS_REGION = "us-west-2"
SSO_SESSION = "scripps-hackathon"
SSO_START_URL = "https://d-9267e96a16.awsapps.com/start"
ACCOUNT_ID = "127696279288"

st.set_page_config(page_title="SpinHance Viewer", layout="wide",
                   initial_sidebar_state="collapsed")


# ─── AWS credential helpers ───────────────────────────────────────────────────

def _ensure_aws_config() -> None:
    config = Path.home() / ".aws" / "config"
    config.parent.mkdir(exist_ok=True)
    text = config.read_text() if config.exists() else ""
    additions = ""
    if f"[sso-session {SSO_SESSION}]" not in text:
        additions += (f"\n[sso-session {SSO_SESSION}]\n"
                      f"sso_start_url = {SSO_START_URL}\nsso_region = {AWS_REGION}\n"
                      f"sso_registration_scopes = sso:account:access\n")
    if f"[profile {AWS_PROFILE}]" not in text:
        additions += (f"\n[profile {AWS_PROFILE}]\nsso_session = {SSO_SESSION}\n"
                      f"sso_account_id = {ACCOUNT_ID}\nsso_role_name = Hackathon\n"
                      f"region = {AWS_REGION}\noutput = json\n")
    if additions:
        with open(config, "a") as f:
            f.write(additions)


def _aws_ok() -> bool:
    r = subprocess.run(["aws", "sts", "get-caller-identity", "--profile", AWS_PROFILE],
                       capture_output=True, timeout=10)
    return r.returncode == 0


# ─── S3 helpers ───────────────────────────────────────────────────────────────

def _s3_ls(prefix: str) -> list[str]:
    r = subprocess.run(["aws", "s3", "ls", prefix.rstrip("/") + "/",
                        "--profile", AWS_PROFILE, "--region", AWS_REGION],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "aws s3 ls returned non-zero")
    return r.stdout.splitlines()


def _s3_cat(uri: str) -> str:
    """Stream an S3 object to stdout (small text files: status/metrics/summary)."""
    r = subprocess.run(["aws", "s3", "cp", uri, "-",
                        "--profile", AWS_PROFILE, "--region", AWS_REGION],
                       capture_output=True, text=True, timeout=60)
    return r.stdout if r.returncode == 0 else ""


@st.cache_data(ttl=60)
def _list_sessions() -> list[str]:
    sessions = []
    for line in _s3_ls(S3_TRAINING):
        parts = line.strip().split()
        if parts and parts[0] == "PRE":
            sessions.append(parts[1].rstrip("/"))
    return sorted(sessions, reverse=True)


@st.cache_data(ttl=60)
def _resolve_run_prefix(session: str) -> str:
    """Return the S3 prefix that actually holds status.json/metrics.jsonl/checkpoints.

    Legacy sessions are flat (<session>/...); rebuilt sessions nest under
    <session>/runs/<run_id>/. Picks the newest run dir for the nested case.
    """
    base = f"{S3_TRAINING}/{session}"
    flat = subprocess.run(["aws", "s3", "ls", f"{base}/status.json",
                           "--profile", AWS_PROFILE, "--region", AWS_REGION],
                          capture_output=True, text=True, timeout=30)
    if flat.returncode == 0 and flat.stdout.strip():
        return base
    try:
        run_ids = sorted(p.strip().split()[-1].rstrip("/")
                         for p in _s3_ls(f"{base}/runs") if p.strip().startswith("PRE"))
        if run_ids:
            return f"{base}/runs/{run_ids[-1]}"
    except Exception:
        pass
    return base


@st.cache_data(ttl=30)
def _load_val_metrics(run_prefix: str) -> list[dict]:
    """Validation score curve from metrics.jsonl (no per-epoch checkpoints needed)."""
    rows = []
    for line in _s3_cat(f"{run_prefix}/metrics.jsonl").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("split") != "val":
            continue
        m = d.get("metrics", {})
        shift = float(m.get("shift_mae_ppm", float("nan")))
        j = float(m.get("j_mae_hz", float("nan")))
        score = (shift + j / 10.0 if not (math.isnan(shift) or math.isnan(j)) else float("nan"))
        rows.append({
            "epoch": int(d.get("epoch", -1)), "score": score,
            "shift_mae_ppm": shift, "j_mae_hz": j,
            "h_shift_mae_ppm": float(m.get("h_shift_mae_ppm", float("nan"))),
            "presence_f1": float(m.get("presence_f1", float("nan"))),
            "deg_acc": float(m.get("deg_acc_balanced", m.get("deg_acc", float("nan")))),
        })
    return sorted(rows, key=lambda r: r["epoch"])


@st.cache_data(ttl=30)
def _list_checkpoints(run_prefix: str) -> list[str]:
    """Available checkpoint names under the run (best, last, epoch_NNNN), ordered."""
    try:
        names = [p.strip().split()[-1][:-3] for p in _s3_ls(f"{run_prefix}/checkpoints")
                 if p.strip().endswith(".pt")]
    except Exception:
        return ["best", "last"]
    epochs = sorted(n for n in names if n.startswith("epoch_"))
    head = [n for n in ("best", "last") if n in names]
    return head + epochs


@st.cache_data(ttl=20)
def _ckpt_version(run_prefix: str, which: str) -> str:
    """S3 identity (date/time/size) of the checkpoint — a cache-buster so that a
    checkpoint REWRITTEN during a still-running job invalidates both the locally
    downloaded file and the cached model object (otherwise the prediction tool
    serves stale weights that disagree with the live val metrics)."""
    r = subprocess.run(["aws", "s3", "ls", f"{run_prefix}/checkpoints/{which}.pt",
                        "--profile", AWS_PROFILE, "--region", AWS_REGION],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip() or "missing"        # "<date> <time> <size> <name>"


def _download_ckpt(session: str, run_prefix: str, which: str, version: str) -> Path:
    local = CACHE_DIR / session / f"{which}.pt"
    ver_file = local.with_suffix(".ver")
    local.parent.mkdir(parents=True, exist_ok=True)
    cached = ver_file.read_text() if ver_file.exists() else None
    if (not local.exists()) or cached != version:        # re-download when S3 changed
        r = subprocess.run(["aws", "s3", "cp", f"{run_prefix}/checkpoints/{which}.pt",
                            str(local), "--profile", AWS_PROFILE, "--region", AWS_REGION],
                           capture_output=True, timeout=300)
        if r.returncode != 0:
            local.unlink(missing_ok=True)
            raise RuntimeError(r.stderr.decode()[:300])
        ver_file.write_text(version)
    return local


@st.cache_resource(show_spinner="Loading model weights…")
def _load_model(session: str, run_prefix: str, which: str, version: str):
    """Rebuild model + standardizer from a checkpoint via the architecture registry.
    ``version`` is part of the cache key so a new checkpoint invalidates the model."""
    import torch
    from model.architectures import build_architecture
    from model.data.standardization import DegeneracyVocab, Standardizer

    local = _download_ckpt(session, run_prefix, which, version)
    ckpt = torch.load(str(local), map_location="cpu", weights_only=False)
    vocab = DegeneracyVocab()
    std = Standardizer().load_state_dict(ckpt["standardizer"])
    mcfg = dict(ckpt.get("cfg", {}).get("model", {"name": "resnet1d_attention_pool", "size": "small"}))
    name = mcfg.pop("name")
    model = build_architecture(name, n_deg_classes=len(vocab), **mcfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, std, vocab, ckpt.get("cfg", {})


# ─── Data helpers — mirror the training pipeline ──────────────────────────────

@st.cache_data(show_spinner="Loading molecules (filtering by 90 MHz spectra)…")
def _load_all_records(json_path: str, spectra_root: str, field: int) -> list[dict]:
    from model.data.records import load_records
    from model.data.splits import canonical_order, reorder
    raw = load_records(json_path, spectra_root, fields=(field,), require_spectra=True)
    records = []
    for r in raw:
        order = canonical_order(r["shifts"], r["couplings"], r["degeneracy"])
        s, c, d = reorder(r["shifts"], r["couplings"], r["degeneracy"], order)
        records.append({**r, "shifts": s, "couplings": c, "degeneracy": d})
    return records


@st.cache_data(show_spinner="Computing test split…")
def _test_records(json_path: str, spectra_root: str, field: int,
                  seed: int, compute_scaffold: bool) -> list[dict]:
    from model.data.splits import make_splits
    records = _load_all_records(json_path, spectra_root, field)
    assignment, _ = make_splits(records, seed=seed, compute_scaffold=compute_scaffold)
    return [r for r in records if assignment.get(r["mol_id"]) == "test"]


@st.cache_data(show_spinner="Simulating spectrum…")
def _simulate(shifts_t: tuple, couplings_t: tuple, degeneracy_t: tuple,
              field_mhz: int) -> tuple[np.ndarray, np.ndarray]:
    from simulation.pyspin.composite import simulate_spectrum_composite
    ppm_axis, intensity = simulate_spectrum_composite(
        np.array(shifts_t), np.array(couplings_t), list(degeneracy_t), float(field_mhz))
    return ppm_axis.astype(np.float64), intensity.astype(np.float64)


def _run_inference(model, intensity: np.ndarray, std, vocab, region_tokens: bool = False) -> dict:
    """Run the model on one 90 MHz spectrum. If the model was trained WITH support
    -region tokens, extract + pass them too (otherwise its region branch is starved
    and predictions are garbage — a train/serve mismatch)."""
    import torch
    from model.evaluation.metrics import decode, _np_pred
    inten = intensity.astype(np.float32)
    spec = torch.from_numpy(inten).unsqueeze(0)
    if region_tokens:
        from model.data.regions import extract_support_regions
        from model.schemas import SpinBatch
        from model.schemas.batch import RegionTokenBatch
        feats, mask = extract_support_regions(inten, 0.0, 12.0, max_regions=48)
        G = 8
        x = SpinBatch(
            spectrum=spec, spectrum_ref=spec,
            shifts=torch.zeros(1, G), couplings=torch.zeros(1, G, G),
            coupling_mask=torch.zeros(1, G, G),
            degeneracy_classes=torch.zeros(1, G, dtype=torch.long),
            degeneracy_values=torch.ones(1, G, dtype=torch.long),
            molecule_ids=["q"],
            region_tokens=RegionTokenBatch(features=torch.from_numpy(feats)[None],
                                           mask=torch.from_numpy(mask)[None]))
    else:
        x = spec
    with torch.no_grad():
        out = model(x)
    return decode(_np_pred(out), std, vocab)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _fig_spectrum(ppm_axis, intensity, title="", color="#2563EB") -> go.Figure:
    fig = go.Figure(go.Scatter(x=ppm_axis, y=intensity, mode="lines",
                               line=dict(color=color, width=1.5)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        xaxis=dict(title="δ (ppm)", autorange="reversed", showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(showgrid=True, gridcolor="#e5e7eb"),
        height=220, margin=dict(l=40, r=10, t=32, b=36),
        plot_bgcolor="white", showlegend=False)
    return fig


def _fig_overlay(traces, title="", height=360) -> go.Figure:
    """Overlay multiple (ppm, intensity, name, color) spectra on one set of axes."""
    fig = go.Figure()
    for ppm, inten, name, color in traces:
        fig.add_trace(go.Scatter(x=ppm, y=inten, mode="lines", name=name,
                                 line=dict(color=color, width=1.4)))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="δ (ppm)", autorange="reversed", showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(title="intensity", showgrid=True, gridcolor="#e5e7eb"),
        height=height, margin=dict(l=50, r=20, t=48, b=44), plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1))
    return fig


def _fig_matrix(shifts, couplings, degeneracy, title="") -> go.Figure:
    G = len(shifts)
    labels = [f"G{i+1}" for i in range(G)]
    z = couplings.copy()
    np.fill_diagonal(z, 0.0)
    text = []
    for r in range(G):
        row = []
        for c in range(G):
            if r == c:
                row.append(f"<b>{shifts[r]:.2f}</b><br>n={int(degeneracy[r])}")
            elif abs(couplings[r, c]) > 0.01:
                row.append(f"{couplings[r, c]:.1f}")
            else:
                row.append("")
        text.append(row)
    max_j = max(float(np.abs(z).max()), 1.0)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, colorscale="RdBu", zmid=0, zmin=-max_j, zmax=max_j,
        text=text, texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="J (Hz)", thickness=12, len=0.8)))
    for i in range(G):
        fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      fillcolor="rgba(200,200,200,0.3)", line=dict(width=0))
    fig.update_layout(title=dict(text=title, font=dict(size=12)),
                      height=320, margin=dict(l=50, r=50, t=32, b=40),
                      xaxis=dict(title="Spin group"), yaxis=dict(title="Spin group"))
    return fig


# ─── Molecule structure viewers ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _mol2d_svg(smiles: str, width=200, height=200) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        o = drawer.drawOptions()
        o.addStereoAnnotation = True; o.bondLineWidth = 1.8; o.padding = 0.14
        drawer.DrawMolecule(mol); drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        return f"<p style='color:#888;font-size:11px;padding:4px;'>2D error: {exc}</p>"


@st.cache_data(show_spinner=False)
def _mol3d_html(smiles: str, width=200, height=200) -> str:
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
        view = py3Dmol.view(width=width, height=height)
        view.addModel(Chem.MolToMolBlock(mol), "mol")
        view.setStyle({"stick": {"colorscheme": "grayCarbon", "radius": 0.12},
                       "sphere": {"colorscheme": "grayCarbon", "radius": 0.28}})
        view.setBackgroundColor("#f3f4f6"); view.spin(True); view.zoomTo()
        return view._make_html()
    except Exception as exc:
        return f"<p style='color:#888;font-size:11px;padding:4px;'>3D error: {exc}</p>"


# ─── Page 1 — Session browser ─────────────────────────────────────────────────

def _page_select() -> None:
    st.title("SpinHance — Training Session Viewer")
    st.caption(f"S3 prefix: `{S3_TRAINING}`")
    if st.columns([1, 5])[0].button("↺  Refresh"):
        _list_sessions.clear()
        st.rerun()
    try:
        sessions = _list_sessions()
    except Exception as exc:
        st.error(f"Cannot list S3 sessions: {exc}")
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
    session = st.session_state.get("session", "")
    if not session:
        st.session_state["page"] = "select"
        st.rerun()

    nav, title = st.columns([1, 8])
    if nav.button("← Sessions"):
        st.session_state["page"] = "select"
        st.rerun()
    title.title(f"Session: `{session}`")

    run_prefix = _resolve_run_prefix(session)
    st.caption(f"run: `{run_prefix.replace(S3_TRAINING + '/', '')}`")

    with st.sidebar:
        st.header("Data")
        json_path = st.text_input("Molecules JSON", value=DEF_JSON)
        spectra_root = st.text_input("Spectra root", value=DEF_SPECTRA)
        ckpt_opts = _list_checkpoints(run_prefix)
        which = st.selectbox("Checkpoint", ckpt_opts,
                             index=ckpt_opts.index("best") if "best" in ckpt_opts else 0,
                             help="best / last / any saved epoch_NNNN")

        json_ok = Path(json_path).exists()
        spectra_ok = (Path(spectra_root) / "90MHz").exists()
        if not json_ok:
            st.error(f"`{Path(json_path).name}` not found locally.")
            if st.button("⬇  Download from S3", key="dl_json"):
                Path(json_path).parent.mkdir(parents=True, exist_ok=True)
                r = subprocess.run(["aws", "s3", "cp", S3_JSON, json_path,
                                    "--profile", AWS_PROFILE, "--region", AWS_REGION],
                                   capture_output=True, timeout=120)
                st.rerun() if r.returncode == 0 else st.error(r.stderr.decode()[:200])
        if not spectra_ok:
            st.warning("90MHz spectra dir not found — split may not match training.")
        if json_ok and spectra_ok:
            st.success("Data paths OK")

    # ── Validation score curve (from metrics.jsonl) ───────────────────────────
    rows = _load_val_metrics(run_prefix)
    if not rows:
        st.warning("No validation metrics found yet for this run.")
        return
    valid = [r for r in rows if not math.isnan(r["score"])]
    if not valid:
        st.error("No epochs with valid validation metrics.")
        return
    best = min(valid, key=lambda r: r["score"])

    st.subheader("Validation score across epochs")
    st.caption("Score = shift_MAE (ppm) + J_MAE (Hz) / 10 · lower is better")
    xs = [r["epoch"] for r in valid]
    ys = [r["score"] for r in valid]
    colors = ["#DC2626" if r["epoch"] == best["epoch"] else "#93C5FD" for r in valid]
    hover = [f"<b>ep{r['epoch']}</b><br>Score {r['score']:.4f}<br>"
             f"shift {r['shift_mae_ppm']:.3f} ppm<br>J {r['j_mae_hz']:.2f} Hz" for r in valid]
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=colors,
                           hovertemplate="%{customdata}<extra></extra>", customdata=hover))
    fig.add_annotation(x=best["epoch"], y=best["score"], text=f"★ best ep{best['epoch']}",
                       showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="#DC2626", size=11))
    fig.update_layout(xaxis=dict(title="Epoch"), yaxis=dict(title="Score (lower = better)"),
                      height=300, margin=dict(l=50, r=20, t=20, b=40),
                      plot_bgcolor="white", showlegend=False, dragmode=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best epoch", best["epoch"])
    c2.metric("Shift MAE", f"{best['shift_mae_ppm']:.3f} ppm")
    c3.metric("J MAE", f"{best['j_mae_hz']:.2f} Hz")
    c4.metric("Presence F1", f"{best['presence_f1']:.3f}" if not math.isnan(best["presence_f1"]) else "—")

    # ── Load model (best/last) ────────────────────────────────────────────────
    st.divider()
    version = _ckpt_version(run_prefix, which)       # cache-buster: re-loads if S3 changed
    try:
        model, std, vocab, cfg = _load_model(session, run_prefix, which, version)
    except Exception as exc:
        st.error(f"Failed to load {which}.pt: {exc}")
        return
    st.caption(f"Loaded `{which}.pt` · {version.rsplit(' ', 1)[0] if version != 'missing' else version}")

    if not Path(json_path).exists() or not (Path(spectra_root) / "90MHz").exists():
        st.info("Set a valid Molecules JSON + spectra root in the sidebar to browse the test set.")
        return

    seed = int(cfg.get("training", {}).get("seed", 0))
    compute_scaffold = (cfg.get("data", {}).get("split", "none") == "scaffold")
    use_regions = bool(cfg.get("data", {}).get("region_tokens", False))   # model trained w/ region tokens?
    if use_regions:
        st.caption("ℹ️ This model uses **support-region tokens** — the inspector extracts + feeds them.")
    try:
        all_recs = _load_all_records(json_path, spectra_root, 90)
        test_recs = _test_records(json_path, spectra_root, 90, seed, compute_scaffold)
    except Exception as exc:
        st.error(f"Failed to build test split: {exc}")
        return
    if not test_recs:
        st.warning("No test molecules found.")
        return
    st.caption(f"Test set: **{len(test_recs):,}** / {len(all_recs):,} molecules · "
               f"split seed={seed} · scaffold={compute_scaffold} · using `{which}.pt`")

    # ── Molecule inspector ────────────────────────────────────────────────────
    draw_col, mol3d_col, sel_col = st.columns([1, 1, 3])
    with sel_col:
        mol_idx = st.selectbox(
            "Select molecule", range(len(test_recs)),
            format_func=lambda i: test_recs[i]["mol_id"]
            + (f" · {test_recs[i]['chembl_id']}" if test_recs[i].get("chembl_id") else ""),
            key="mol_selector")
        rec = test_recs[mol_idx]
        if rec.get("chembl_id"):
            st.caption(f"**ChEMBL:** `{rec['chembl_id']}` · spins {int(rec['degeneracy'].sum())}")
        if rec.get("smiles"):
            st.code(rec["smiles"], language=None)
        if st.button("▶  Run Inference", type="primary", key="run_inf"):
            with st.spinner("Running inference…"):
                # the model always takes the 90 MHz spectrum as input
                _, in90 = _simulate(tuple(rec["shifts"].tolist()),
                                    tuple(map(tuple, rec["couplings"].tolist())),
                                    tuple(rec["degeneracy"].tolist()), 90)
                dec = _run_inference(model, in90, std, vocab, region_tokens=use_regions)
            st.session_state["mol_pred"] = {"dec": dec, "mol_id": rec["mol_id"]}

    with draw_col:
        if rec.get("smiles"):
            b64 = base64.b64encode(_mol2d_svg(rec["smiles"], 210, 210).encode()).decode()
            st.markdown(f'<div style="background:white;border-radius:8px;padding:4px;">'
                        f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;display:block;"/></div>',
                        unsafe_allow_html=True)
        else:
            st.info("No SMILES")
    with mol3d_col:
        if rec.get("smiles"):
            components.html(_mol3d_html(rec["smiles"], 210, 210), height=222, scrolling=False)
        else:
            st.info("No SMILES")

    # ── Matrices (field-independent) ──────────────────────────────────────────
    pred_state = st.session_state.get("mol_pred")
    pred_ready = bool(pred_state and pred_state["mol_id"] == rec["mol_id"])
    dec = pred_state["dec"] if pred_ready else None

    col_gt, col_pred = st.columns(2)
    with col_gt:
        st.markdown("#### Ground-truth matrix")
        st.plotly_chart(_fig_matrix(rec["shifts"], rec["couplings"], rec["degeneracy"],
                                    title="diag = δ ppm · off-diag = J Hz"),
                        use_container_width=True)
    with col_pred:
        st.markdown("#### Predicted matrix")
        if pred_ready:
            st.plotly_chart(_fig_matrix(dec["shifts"][0], dec["couplings"][0], dec["degeneracy"][0],
                                        title="predicted"), use_container_width=True)
        else:
            st.info("Press **▶ Run Inference** to see the model output.")

    # ── Spectra: experimental (ground truth) vs predicted, overlaid per field ──
    st.markdown("#### Spectra — experimental vs predicted")
    if not pred_ready:
        st.caption("Showing experimental only — press **▶ Run Inference** to overlay the prediction.")
    for field in (90, 600):
        gx, gy = _simulate(tuple(rec["shifts"].tolist()),
                           tuple(map(tuple, rec["couplings"].tolist())),
                           tuple(rec["degeneracy"].tolist()), field)
        traces = [(gx, gy, "experimental (ground truth)", "#2563EB")]
        if pred_ready:
            px, py = _simulate(tuple(dec["shifts"][0].tolist()),
                               tuple(map(tuple, dec["couplings"][0].tolist())),
                               tuple(dec["degeneracy"][0].tolist()), field)
            traces.append((px, py, "predicted", "#DC2626"))
        st.plotly_chart(_fig_overlay(traces, f"{field} MHz"), use_container_width=True)


# ─── AWS login gate ────────────────────────────────────────────────────────────

def _page_aws_login() -> None:
    st.title("AWS Login Required")
    st.warning(f"The `{AWS_PROFILE}` SSO session has expired. Click Login, then Refresh.")
    a, b, _ = st.columns([1, 1, 4])
    if a.button("🔑  Login with AWS SSO", type="primary"):
        r = subprocess.run(["aws", "sso", "login", "--profile", AWS_PROFILE],
                           capture_output=True, text=True, timeout=180)
        st.rerun() if r.returncode == 0 else st.code(r.stderr or r.stdout)
    if b.button("↺  Refresh"):
        st.rerun()


# ─── Router ─────────────────────────────────────────────────────────────────--

_ensure_aws_config()
if not _aws_ok():
    _page_aws_login()
    st.stop()
if st.session_state.get("page", "select") == "select":
    _page_select()
else:
    _page_analysis()
