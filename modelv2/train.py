"""
modelv2.train — losses, metrics, training loop, diagnostics, CLI
================================================================
Everything needed to take the model from random weights to a trained checkpoint,
plus the diagnostics that tell us *why* it plateaus.

Single-stage training: all four heads are trained jointly from epoch 0 and the
encoder regresses the spin-system matrix directly from the input spectrum. The
loss is deliberately the minimal four-term matrix loss; the candidate anti-
collapse terms (one-sided variance matching, beta-NLL) and the loss-weight
``Ramp`` schedule framework are built here but default to off / constant so the
fixed-weight arm and the scheduled arm cost nothing to A/B.

Numerical-stability contract (hard): bf16 autocast only (never fp16 / GradScaler
— bf16 keeps fp32's exponent range so there is no overflow path); every loss and
metric is computed in fp32 (predictions/logits are upcast first); every masked or
weighted reduction floors its denominator; linear-warmup -> cosine LR, global
grad-norm clip <= 1.0, and AdamW are always on; a non-finite loss skips+logs the
batch and aborts if the skip rate is too high.

This module imports torch **lazily** (inside functions) so that ``--dry-run``
exercises the entire data path in a torch-free environment.

CLI:
    PYTHONPATH=. python -m modelv2.train --dry-run --spin_systems=<f>
    PYTHONPATH=. python -m modelv2.train --smoke
    PYTHONPATH=. python -m modelv2.train --spin_systems=<f> --spectra=<f> [--out ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelv2 import data as D

# Cap on samples collected for the variance/Pearson diagnostics (metrics that
# need every prediction in memory at once); headline MAE/F1 are computed on the
# same bounded, shuffled sample and are stable well below this size.
DIAG_MAX_SAMPLES = 20000

# Default on-GPU augmentation strength (fractions of per-spectrum peak height).
AUG_KWARGS = dict(noise_sigma_frac=0.01, max_ref_shift_ppm=0.01,
                  baseline_amp_frac=0.02, broaden_sigma_pts=0.0)


# =============================================================================
# Configuration — one dataclass, sane full-dataset defaults
# =============================================================================

@dataclass
class TrainConfig:
    # Data
    points: int = 16384
    ppm_from: float = 0.0
    ppm_to: float = 12.0
    spectrum_field: str = "spec90"
    seed: int = 0
    # Model
    n_groups: int = 8
    # Training
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    epochs: int = 60
    warmup_frac: float = 0.03
    linewidth_hz: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {
        "shift": 1.0, "jmag": 1.0, "presence": 0.5, "deg": 0.5})
    patience: int = 10
    ema_decay: float = 0.999        # per-step EMA; 0 disables
    save_every: int = 10            # periodic last.pt for crash safety
    log_every: int = 50             # steps between event-log flushes (no per-step sync)
    # Infrastructure
    device: str = "cuda"
    amp_dtype: str = "bf16"
    ckpt_path: str = "checkpoint.pt"
    cache_spectra: bool = True      # stream 90MHz.tar.gz into RAM at startup
    gpu_augment: bool = True        # vectorized augmentation on-GPU per batch
    num_workers: int = 0            # RAM cache + GPU aug: no CPU loader needed
    compile: bool = False           # opt-in torch.compile; falls back to eager
    val_every: int = 1


# =============================================================================
# Loss-weight schedule framework (a Ramp per term) + LR factor
# =============================================================================

@dataclass
class Ramp:
    """A per-term weight schedule over training progress in [0, 1].

    A bare float is a constant weight (the fixed-weight arm); a Ramp with
    ``end`` set sweeps ``start -> end`` between ``start_frac`` and ``end_frac``
    with ``shape`` in {const, linear, cosine}. Building this regardless lets the
    scheduled arm of any A/B be expressed without new code.
    """
    start: float
    end: float | None = None
    start_frac: float = 0.0
    end_frac: float = 1.0
    shape: str = "const"

    def value(self, progress: float) -> float:
        if self.end is None or self.shape == "const":
            return float(self.start)
        if progress <= self.start_frac:
            return float(self.start)
        if progress >= self.end_frac:
            return float(self.end)
        f = (progress - self.start_frac) / max(self.end_frac - self.start_frac, 1e-9)
        if self.shape == "cosine":
            f = 0.5 * (1.0 - math.cos(math.pi * f))
        return float(self.start + (self.end - self.start) * f)


def resolve_weight(w, progress: float) -> float:
    return w.value(progress) if isinstance(w, Ramp) else float(w)


def lr_factor(step, warmup_steps, total_steps, min_factor=0.05):
    """Linear warmup then cosine decay to ``min_factor`` (multiplier for LambdaLR)."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    prog = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * prog))


# =============================================================================
# Storage — local filesystem or s3:// (used by fit; reused read-side by gui.py)
# =============================================================================

def is_s3(uri: str) -> bool:
    return str(uri).startswith("s3://")


def _s3_split(uri):
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _s3_client():
    import boto3
    region = os.environ.get("AWS_REGION", "us-west-2")
    profile = os.environ.get("AWS_PROFILE", "hack-scripps")
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("s3")


def s3_put_bytes(uri, data, content_type="application/octet-stream"):
    bucket, key = _s3_split(uri)
    _s3_client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def s3_get_bytes(uri):
    from botocore.exceptions import ClientError
    bucket, key = _s3_split(uri)
    try:
        return _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def s3_get_json(uri, default=None):
    b = s3_get_bytes(uri)
    if b is None:
        return default
    try:
        return json.loads(b)
    except Exception:
        return default


def s3_get_text(uri, default=None):
    b = s3_get_bytes(uri)
    return b.decode() if b is not None else default


def s3_list_prefixes(uri_prefix):
    bucket, prefix = _s3_split(uri_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    cl = _s3_client()
    names = []
    for page in cl.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            names.append(cp["Prefix"].rstrip("/").rsplit("/", 1)[-1])
    return names


def s3_list_keys(uri_prefix):
    bucket, prefix = _s3_split(uri_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    cl = _s3_client()
    keys = []
    for page in cl.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if rel:
                keys.append(rel)
    return keys


class RunStore:
    """Write training artifacts to a session root that is local or an s3:// URI.

    JSON is written atomically (local: tmp+rename); JSONL is append-only locally
    and buffered+re-put on S3 (flushed every ``log_every`` steps by the caller),
    so a mid-epoch crash leaves prior rows intact.
    """

    def __init__(self, root: str):
        self.root = str(root).rstrip("/")
        self.s3 = is_s3(self.root)
        self._jsonl: dict[str, list[str]] = {}
        if not self.s3:
            Path(self.root).mkdir(parents=True, exist_ok=True)

    def full(self, name: str) -> str:
        return f"{self.root}/{name}"

    def write_json(self, name, obj):
        if self.s3:
            s3_put_bytes(self.full(name),
                         json.dumps(obj, indent=2, sort_keys=True, default=_jsonable).encode(),
                         "application/json")
        else:
            p = Path(self.root) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_jsonable))
            os.replace(tmp, p)

    def append_jsonl(self, name, obj):
        line = json.dumps({"time": time.time(), **obj}, sort_keys=True, default=_jsonable)
        if self.s3:
            buf = self._jsonl.setdefault(name, [])
            buf.append(line)
            s3_put_bytes(self.full(name), ("\n".join(buf) + "\n").encode(),
                         "application/x-ndjson")
        else:
            p = Path(self.root) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(line + "\n")

    def write_bytes(self, name, data, content_type="application/octet-stream"):
        if self.s3:
            s3_put_bytes(self.full(name), data, content_type)
        else:
            p = Path(self.root) / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def write_npy(self, name, arr):
        buf = io.BytesIO()
        np.save(buf, arr)
        self.write_bytes(name, buf.getvalue(), "application/octet-stream")

    def save_torch(self, name, obj):
        import torch
        buf = io.BytesIO()
        torch.save(obj, buf)
        self.write_bytes(name, buf.getvalue())


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# =============================================================================
# Losses (torch; computed in fp32)
# =============================================================================

def _masked_mean(x, mask):
    """Mean of ``x`` over ``mask>0`` with a floored denominator (no 0/0)."""
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def compute_losses(pred, target, cfg: TrainConfig, balance, progress=0.0):
    """Four-term matrix loss (+ gated candidate terms). Returns (total, raw, weighted).

    ``raw`` are the unweighted per-term values; ``weighted`` are after applying the
    resolved per-term weights (so the GUI can show both). Everything is fp32.
    """
    import torch
    import torch.nn.functional as F

    w = dict(cfg.loss_weights)
    rw = {k: resolve_weight(v, progress) for k, v in w.items()}

    shift_mu = pred["shifts"].float()
    jmag_mu = pred["j_mag"].float()
    pres_logit = pred["j_presence"].float()
    deg_logit = pred["deg_logits"].float()

    tgt_shift = target["shifts"].float()
    tgt_jmag = target["j_mag"].float()
    mask = target["j_presence"].float()          # (B, n_pairs) in {0,1}
    tgt_deg = target["deg_class"].long()

    # Optional epsilon-band relabeling of near-degenerate slots (candidate; off
    # unless loss_weights["eps_band_ppm"] > 0). Removes canonical-ordering label
    # noise without discarding slot semantics the J head relies on.
    eps_ppm = float(rw.get("eps_band_ppm", 0.0))
    if eps_ppm > 0:
        tgt_shift, tgt_jmag, mask, tgt_deg = _eps_band_relabel(
            shift_mu.detach(), tgt_shift, tgt_jmag, mask, tgt_deg, eps_ppm,
            cfg.n_groups)

    shift = F.smooth_l1_loss(shift_mu, tgt_shift)
    jmag_el = F.smooth_l1_loss(jmag_mu, tgt_jmag, reduction="none")
    jmag = _masked_mean(jmag_el, mask)
    pos_w = balance.get("presence_pos_weight") if balance else None
    presence = F.binary_cross_entropy_with_logits(pres_logit, mask, pos_weight=pos_w)
    B, G, C = deg_logit.shape
    deg = F.cross_entropy(deg_logit.reshape(B * G, C), tgt_deg.reshape(B * G),
                          weight=(balance.get("deg_weights") if balance else None))

    raw = {"shift": shift, "jmag": jmag, "presence": presence, "deg": deg}

    # --- candidate anti-collapse terms (default weight 0) --------------------
    if rw.get("var_match", 0.0) > 0:
        raw["var_match"] = _var_match(shift_mu, tgt_shift) + \
            _var_match(jmag_mu, tgt_jmag, mask)
    if rw.get("beta_nll", 0.0) > 0:
        beta = float(rw.get("beta_nll_beta", 0.5))
        raw["beta_nll"] = (_beta_nll(shift_mu, pred["shift_logvar"].float(), tgt_shift, beta)
                           + _beta_nll(jmag_mu, pred["jmag_logvar"].float(), tgt_jmag, beta, mask))

    weighted = {k: rw.get(k, 0.0) * v for k, v in raw.items()}
    total = sum(weighted.values())
    return total, {k: v.detach() for k, v in raw.items()}, \
        {k: (v.detach() if hasattr(v, "detach") else v) for k, v in weighted.items()}


def _var_match(pred, target, mask=None):
    """One-sided per-cell variance matching: penalize predicted under-dispersion.

    ``mean(relu(std_target - std_pred)**2)`` over cells (columns). With a mask,
    per-column std is taken over present entries only (floored denominators)."""
    import torch
    if mask is None:
        sp = pred.std(dim=0)
        st = target.std(dim=0)
    else:
        n = mask.sum(dim=0).clamp_min(1.0)
        mp = (pred * mask).sum(0) / n
        mt = (target * mask).sum(0) / n
        sp = torch.sqrt(((pred - mp) ** 2 * mask).sum(0) / n + 1e-12)
        st = torch.sqrt(((target - mt) ** 2 * mask).sum(0) / n + 1e-12)
    return torch.relu(st - sp).pow(2).mean()


def _beta_nll(mu, logvar, target, beta=0.5, mask=None):
    """beta-NLL (Seitzer et al.): Gaussian NLL weighted by detached var**beta.

    Plain NLL lets the model inflate sigma^2 to dodge hard cells; the var**beta
    weight restores the gradient on those cells. Used only after an MSE warmup."""
    import torch
    var = torch.exp(logvar)
    nll = 0.5 * ((target - mu) ** 2 / var + logvar)
    loss = nll * var.detach().pow(beta)
    if mask is None:
        return loss.mean()
    return _masked_mean(loss, mask)


def _eps_band_relabel(pred_shift, tgt_shift, tgt_jmag, mask, tgt_deg, eps_ppm, G):
    """Permute only near-degenerate target slots to best match predictions.

    For each sample, groups of target slots whose shifts lie within ``eps_ppm``
    of each other are re-assigned to predicted slots by minimum-cost matching
    (scipy). Couplings/presence/degeneracy are permuted by the same permutation
    so slot semantics are preserved. Default off; this is a candidate term.
    """
    import torch
    from scipy.optimize import linear_sum_assignment

    Bn = pred_shift.shape[0]
    ps = pred_shift.detach().cpu().numpy()
    ts = tgt_shift.detach().cpu().numpy()
    tj = D.pairs_to_matrix(tgt_jmag.detach().cpu().numpy(), G)   # (B,G,G)
    tm = D.pairs_to_matrix(mask.detach().cpu().numpy(), G)
    td = tgt_deg.detach().cpu().numpy()

    iu = D.triu_index_map(G)
    new_s = ts.copy()
    new_j = tj.copy()
    new_m = tm.copy()
    new_d = td.copy()
    for b in range(Bn):
        order = np.argsort(-ts[b])
        # cluster consecutive (sorted) shifts within eps
        clusters, cur = [], [order[0]]
        for k in order[1:]:
            if abs(ts[b, k] - ts[b, cur[-1]]) < eps_ppm:
                cur.append(k)
            else:
                clusters.append(cur); cur = [k]
        clusters.append(cur)
        perm = np.arange(G)
        for cl in clusters:
            if len(cl) < 2:
                continue
            cl = np.array(cl)
            cost = np.abs(ts[b, cl][:, None] - ps[b, cl][None, :])
            ri, ci = linear_sum_assignment(cost)
            perm[cl[ri]] = cl[ci]
        new_s[b] = ts[b][perm]
        new_d[b] = td[b][perm]
        new_j[b] = tj[b][np.ix_(perm, perm)]
        new_m[b] = tm[b][np.ix_(perm, perm)]
    dev = tgt_shift.device
    return (torch.as_tensor(new_s, device=dev, dtype=tgt_shift.dtype),
            torch.as_tensor(new_j[:, iu[0], iu[1]], device=dev, dtype=tgt_jmag.dtype),
            torch.as_tensor(new_m[:, iu[0], iu[1]], device=dev, dtype=mask.dtype),
            torch.as_tensor(new_d, device=dev, dtype=tgt_deg.dtype))


# =============================================================================
# Metrics (numpy + scipy; eval-side, no grad)
# =============================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def decode(pred, std: "D.Standardizer", vocab: "D.DegeneracyVocab", presence_thresh=0.5):
    """Standardized/logit model outputs -> physical spin-system matrix."""
    G = pred["shifts"].shape[1]
    shifts = std.inverse_shifts(pred["shifts"])
    present = _sigmoid(pred["j_presence"]) > presence_thresh
    jmag = std.inverse_j(pred["j_mag"]) * present
    couplings = D.pairs_to_matrix(jmag, G)
    deg_idx = np.argmax(pred["deg_logits"], axis=-1)
    degeneracy = vocab.from_index(deg_idx)
    return dict(shifts=shifts, couplings=couplings, degeneracy=degeneracy,
                presence=present.astype(np.float32))


def _hungarian_perm(pred_shifts, tgt_shifts):
    """Per-sample assignment perms[b,i] = pred index matched to target slot i."""
    from scipy.optimize import linear_sum_assignment
    B, G = pred_shifts.shape
    perms = np.zeros((B, G), dtype=int)
    for b in range(B):
        cost = np.abs(tgt_shifts[b, :, None] - pred_shifts[b, None, :])
        _, ci = linear_sum_assignment(cost)
        perms[b] = ci
    return perms


def compute_metrics(pred, target, std, vocab, presence_thresh=0.5):
    """Physical-unit metrics from STANDARDIZED predictions/targets (canonical order)."""
    G = pred["shifts"].shape[1]
    dec = decode(pred, std, vocab, presence_thresh)
    iu = D.triu_index_map(G)

    tgt_shifts = std.inverse_shifts(target["shifts"])
    tgt_present = target["j_presence"] > 0.5
    tgt_jmag = std.inverse_j(target["j_mag"]) * tgt_present

    shift_mae = float(np.abs(dec["shifts"] - tgt_shifts).mean())
    pred_jmag_ut = dec["couplings"][:, iu[0], iu[1]]
    m = tgt_present
    j_mae = float(np.abs(pred_jmag_ut[m] - tgt_jmag[m]).mean()) if m.any() else 0.0

    pp = dec["presence"] > 0.5
    tp = int((pp & tgt_present).sum()); fp = int((pp & ~tgt_present).sum())
    fn = int((~pp & tgt_present).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    pres_acc = float((pp == tgt_present).mean())

    tgt_deg = vocab.from_index(target["deg_class"])
    pred_deg = dec["degeneracy"]
    deg_acc = float((pred_deg == tgt_deg).mean())
    recalls = [float((pred_deg[tgt_deg == v] == v).mean()) for v in np.unique(tgt_deg)]
    deg_acc_balanced = float(np.mean(recalls)) if recalls else 0.0

    out = dict(shift_mae_ppm=shift_mae, j_mae_hz=j_mae, presence_acc=pres_acc,
               presence_f1=float(f1), presence_recall=float(rec),
               presence_precision=float(prec), deg_acc=deg_acc,
               deg_acc_balanced=deg_acc_balanced)

    # Hungarian-matched shift/J/deg (G=8 -> negligible cost)
    B = dec["shifts"].shape[0]
    perms = _hungarian_perm(dec["shifts"], tgt_shifts)
    bi = np.arange(B)[:, None]
    out["h_shift_mae_ppm"] = float(np.abs(dec["shifts"][bi, perms] - tgt_shifts).mean())
    tgt_C = D.pairs_to_matrix(tgt_jmag, G)
    h_errs = []
    for b in range(B):
        p = perms[b]
        pC = dec["couplings"][b][np.ix_(p, p)]
        mm = tgt_present[b]
        if mm.any():
            h_errs.extend(np.abs(pC[iu[0], iu[1]][mm] - tgt_C[b][iu[0], iu[1]][mm]).tolist())
    out["h_j_mae_hz"] = float(np.mean(h_errs)) if h_errs else 0.0
    out["h_deg_acc"] = float((dec["degeneracy"][bi, perms] == tgt_deg).mean())
    return out


# =============================================================================
# EMA (exponential moving average of parameters)
# =============================================================================

class EMA:
    """Shadow parameter copy updated every optimizer step.

    ``shadow = decay*shadow + (1-decay)*live`` per step (decay default 0.999).
    GroupNorm has no running buffers, so only parameters are tracked; non-trainable
    parameters (if any) are copied straight through. Validation, metrics,
    diagnostics, and checkpoint selection all read the shadow.
    """

    def __init__(self, model, decay=0.999):
        import torch
        self.decay = float(decay)
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters()}
        self._backup = None

    @property
    def enabled(self):
        return self.decay > 0

    def update(self, model):
        import torch
        if not self.enabled:
            return
        with torch.no_grad():
            for k, p in model.named_parameters():
                s = self.shadow[k]
                if p.requires_grad:
                    s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
                else:
                    s.copy_(p.detach())

    def store(self, model):
        self._backup = {k: p.detach().clone() for k, p in model.named_parameters()}

    def copy_to(self, model):
        import torch
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in self.shadow:
                    p.copy_(self.shadow[k])

    def restore(self, model):
        import torch
        if self._backup is None:
            return
        with torch.no_grad():
            for k, p in model.named_parameters():
                p.copy_(self._backup[k])
        self._backup = None

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}


class _UseEMA:
    """Context manager: evaluate on the shadow weights, then restore the live ones."""
    def __init__(self, ema, model):
        self.ema, self.model = ema, model

    def __enter__(self):
        if self.ema is not None and self.ema.enabled:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
        return self.model

    def __exit__(self, *exc):
        if self.ema is not None and self.ema.enabled:
            self.ema.restore(self.model)
        return False


# =============================================================================
# Evaluation + diagnostics (run on the shadow model)
# =============================================================================

def _collect_predictions(model, loader, device, amp_ctx, max_samples=DIAG_MAX_SAMPLES):
    """Run the model over a loader; return stacked numpy pred/target arrays."""
    import torch
    model.eval()
    keep = {k: [] for k in ("shifts", "shift_logvar", "j_mag", "jmag_logvar",
                             "j_presence", "deg_logits")}
    tgt = {k: [] for k in ("shifts", "j_mag", "j_presence", "deg_class")}
    n = 0
    with torch.no_grad():
        for batch in loader:
            spec = batch["spectrum"].to(device, non_blocking=True)
            with amp_ctx():
                pred = model(spec)
            for k in keep:
                keep[k].append(pred[k].float().cpu().numpy())
            for k in tgt:
                tgt[k].append(batch[k].numpy())
            n += spec.shape[0]
            if n >= max_samples:
                break
    pred_np = {k: np.concatenate(v) for k, v in keep.items()}
    tgt_np = {k: np.concatenate(v) for k, v in tgt.items()}
    return pred_np, tgt_np


def _pearson_per_cell(pred, target):
    """Column-wise Pearson r between (N, K) prediction and target arrays."""
    p = pred - pred.mean(0, keepdims=True)
    t = target - target.mean(0, keepdims=True)
    num = (p * t).sum(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0)) + 1e-12
    return num / den


def evaluate(model, loader, std, vocab, device, amp_ctx, max_samples=DIAG_MAX_SAMPLES):
    """Return (metrics, diagnostics, arrays) computed on a bounded val sample."""
    pred_np, tgt_np = _collect_predictions(model, loader, device, amp_ctx, max_samples)
    metrics = compute_metrics(pred_np, tgt_np, std, vocab)

    # physical-unit pred/target for variance & correlation diagnostics
    G = pred_np["shifts"].shape[1]
    p_shift = std.inverse_shifts(pred_np["shifts"])
    t_shift = std.inverse_shifts(tgt_np["shifts"])
    present = tgt_np["j_presence"] > 0.5
    p_j = std.inverse_j(pred_np["j_mag"])
    t_j = std.inverse_j(tgt_np["j_mag"])

    var_ratio_shift = (p_shift.var(0) / np.maximum(t_shift.var(0), 1e-12))
    pearson_shift = _pearson_per_cell(p_shift, t_shift)
    # couplings: use cells that are present in at least a few samples
    col_has = present.sum(0) >= 8
    if col_has.any():
        vr_j, pr_j = [], []
        for k in np.where(col_has)[0]:
            m = present[:, k]
            vr_j.append(p_j[m, k].var() / max(t_j[m, k].var(), 1e-12))
            pr_j.append(_pearson_per_cell(p_j[m, k][:, None], t_j[m, k][:, None])[0])
        var_ratio_j = float(np.mean(vr_j)); pearson_j = float(np.mean(pr_j))
    else:
        var_ratio_j = pearson_j = float("nan")

    diagnostics = dict(
        var_ratio_shift_mean=float(np.mean(var_ratio_shift)),
        var_ratio_shift=var_ratio_shift.tolist(),
        var_ratio_j_mean=var_ratio_j,
        pearson_shift_mean=float(np.mean(pearson_shift)),
        pearson_shift=pearson_shift.tolist(),
        pearson_j_mean=pearson_j,
        pred_logvar_shift_mean=float(pred_np["shift_logvar"].mean()),
        pred_logvar_j_mean=float(pred_np["jmag_logvar"].mean()),
        n_eval=int(pred_np["shifts"].shape[0]),
    )
    arrays = dict(pred=pred_np, tgt=tgt_np)
    return metrics, diagnostics, arrays


def constant_mean_baseline(train_records, vocab, std):
    """MAE of always predicting the per-slot TRAIN mean (the trivial baseline).

    Returns physical-unit shift/J MAE of the constant predictor, evaluated on the
    same train means (a fixed reference the model must beat — what matters is the
    gap to the achievable floor, not to this number)."""
    S, J, Jpres = [], [], []
    for r in train_records:
        t = D.encode_target(r, vocab)
        S.append(t["shifts"])
        J.append(t["j_mag"]); Jpres.append(t["j_presence"])
    S = np.stack(S); J = np.stack(J); Jpres = np.stack(Jpres) > 0.5
    shift_mean = S.mean(0)                       # per-slot physical mean (G,)
    shift_mae = float(np.abs(S - shift_mean).mean())
    if Jpres.any():
        col_mean = np.array([J[Jpres[:, k], k].mean() if Jpres[:, k].any() else 0.0
                             for k in range(J.shape[1])])
        diffs = [np.abs(J[Jpres[:, k], k] - col_mean[k]) for k in range(J.shape[1])
                 if Jpres[:, k].any()]
        j_mae = float(np.concatenate(diffs).mean()) if diffs else 0.0
    else:
        j_mae = 0.0
    return dict(baseline_shift_mae_ppm=shift_mae, baseline_j_mae_hz=j_mae)


# =============================================================================
# Per-head gradient norms (logged every log_every steps)
# =============================================================================

def head_grad_norms(model):
    import torch
    groups = {"encoder": model.encoder, "shift": model.shift_head,
              "jmag": model.jmag_head, "presence": model.jpres_head,
              "deg": model.deg_head}
    out = {}
    for name, mod in groups.items():
        sq = 0.0
        for p in mod.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                sq += float(g.pow(2).sum())
        out[f"gradnorm_{name}"] = math.sqrt(sq)
    return out


# =============================================================================
# Probe set materialization + per-epoch probe artifacts
# =============================================================================

def materialize_diagnostic_set(test_records, cache, cfg, n=500, seed=0):
    """Sample up to ``n`` molecules from the held-out test fold, canonically ordered.

    Returns (diag_records_for_json, spectra_fp16).  The caller passes
    ``test_records`` already filtered to the test fold — no assignment dict
    needed here.  Materializing server-side is the guarantee the GUI never
    touches a trained-on molecule."""
    rng = np.random.default_rng(seed)
    recs = test_records
    if len(recs) > n:
        idx = rng.choice(len(recs), n, replace=False)
        recs = [recs[i] for i in sorted(idx)]

    out_records, specs = [], []
    for r in recs:
        s, c, d = D.reorder(r["shifts"], r["couplings"], r["degeneracy"],
                             D.canonical_order(r["shifts"], r["couplings"], r["degeneracy"]))
        out_records.append({
            "mol_id": r["mol_id"], "index": r.get("index"),
            "smiles": r.get("smiles"), "chembl_id": r.get("chembl_id"),
            "inchikey": r.get("inchikey"),
            "shifts": s.tolist(), "couplings": c.tolist(),
            "degeneracy": [int(x) for x in d],
        })
        if cache is not None and r["mol_id"] in cache:
            specs.append(cache[r["mol_id"]])
        elif r.get(cfg.spectrum_field) is not None:
            specs.append(np.asarray(r[cfg.spectrum_field], dtype=np.float32))
        else:
            specs.append(np.zeros(cfg.points, dtype=np.float32))

    spectra = np.stack(specs).astype(np.float16) if specs else np.zeros((0, cfg.points), np.float16)
    return out_records, spectra


def _matrix_png(true_mat, pred_mat, title=""):
    """Return PNG bytes of a true/pred/|error| matrix triptych (or None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    err = np.abs(pred_mat - true_mat)
    vmax = max(float(np.abs(true_mat).max()), float(np.abs(pred_mat).max()), 1e-6)
    for ax, mat, label, cmap, lo, hi in [
        (axes[0], true_mat, "True", "RdBu_r", -vmax, vmax),
        (axes[1], pred_mat, "Predicted", "RdBu_r", -vmax, vmax),
        (axes[2], err, "|Error|", "hot_r", 0.0, max(float(err.max()), 1e-6)),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(label, fontsize=9); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


_FAILURE_HINTS = {
    "large_shift_error": "Increase shift loss weight or enable epsilon-band matching",
    "false_negative_couplings": "Increase presence_pos_weight",
    "false_positive_couplings": "Raise presence threshold or up-weight the absent class",
    "bad_j_magnitude": "Increase j_mag loss weight",
    "wrong_degeneracy": "Check degeneracy vocab / class weights",
}


def _tag_failure(m):
    if m["shift_mae_ppm"] > 0.25:
        t = "large_shift_error"
    elif m["presence_f1"] < 0.4 and m.get("presence_recall", 1.0) < 0.5:
        t = "false_negative_couplings"
    elif m["presence_f1"] < 0.4:
        t = "false_positive_couplings"
    elif m["j_mae_hz"] > 3.5:
        t = "bad_j_magnitude"
    elif m["deg_acc"] < 0.75:
        t = "wrong_degeneracy"
    else:
        t = "ok"
    return t


def run_probes(model, diag_records, diag_spectra, std, vocab, cfg, store, epoch,
               device, amp_ctx, n_plots=16, save_plots=True):
    """Inference on the held-out diagnostic set; write per-epoch probe artifacts."""
    import torch
    from collections import Counter
    if not diag_records:
        return {}
    prefix = f"probes/epoch_{epoch:04d}"
    model.eval()
    x = torch.from_numpy(np.asarray(diag_spectra, dtype=np.float32)).to(device)
    preds = []
    with torch.no_grad():
        for s in range(0, x.shape[0], cfg.batch_size):
            with amp_ctx():
                p = model(x[s:s + cfg.batch_size])
            preds.append({k: p[k].float().cpu().numpy() for k in p})
    pred_np = {k: np.concatenate([pp[k] for pp in preds]) for k in preds[0]}

    G = cfg.n_groups
    iu = D.triu_index_map(G)
    dec = decode(pred_np, std, vocab)

    per_mol, tags = [], []
    for b, rec in enumerate(diag_records):
        t_shift = np.asarray(rec["shifts"], dtype=float)
        t_coup = np.asarray(rec["couplings"], dtype=float)
        t_deg = np.asarray(rec["degeneracy"])
        tgt_std = std.transform(D.encode_target(
            {"shifts": t_shift, "couplings": t_coup, "degeneracy": t_deg}, vocab))
        met = compute_metrics({k: pred_np[k][b:b + 1] for k in pred_np},
                              {k: tgt_std[k][None] for k in
                               ("shifts", "j_mag", "j_presence", "deg_class")},
                              std, vocab)
        tag = _tag_failure(met)
        tags.append(tag)
        pred_mat = dec["couplings"][b].copy(); np.fill_diagonal(pred_mat, dec["shifts"][b])
        true_mat = t_coup.copy(); np.fill_diagonal(true_mat, t_shift)
        per_mol.append({
            "mol_id": rec["mol_id"], "smiles": rec.get("smiles", ""),
            "chembl_id": rec.get("chembl_id", ""), "failure_type": tag, **met,
            "true_shifts": t_shift.tolist(), "pred_shifts": dec["shifts"][b].tolist(),
            "true_deg": [int(v) for v in t_deg], "pred_deg": [int(v) for v in dec["degeneracy"][b]],
            "true_couplings": true_mat.tolist(), "pred_couplings": pred_mat.tolist(),
        })
        if save_plots and b < n_plots:
            png = _matrix_png(true_mat, pred_mat,
                              f"{rec['mol_id']}  shift={met['shift_mae_ppm']:.3f}ppm "
                              f"J={met['j_mae_hz']:.2f}Hz")
            if png:
                store.write_bytes(f"{prefix}/matrix_{b:03d}_{rec['mol_id']}.png", png, "image/png")

    agg = {k: float(np.mean([m[k] for m in per_mol]))
           for k in ("shift_mae_ppm", "j_mae_hz", "presence_f1", "deg_acc",
                     "deg_acc_balanced", "h_shift_mae_ppm", "h_j_mae_hz")}
    counts = Counter(tags)
    dominant = counts.most_common(1)[0][0] if counts else "none"
    failure_summary = dict(dominant_failure=dominant, n_ok=counts.get("ok", 0),
                           failure_distribution=dict(counts.most_common()),
                           n_molecules=len(per_mol),
                           recommendation=_FAILURE_HINTS.get(dominant, ""))

    store.write_json(f"{prefix}/probe_metrics.json", agg)
    store.write_json(f"{prefix}/predictions.json", per_mol)
    store.write_json(f"{prefix}/failure_summary.json", failure_summary)
    store.write_json(f"{prefix}/worst_cases.json",
                     sorted(per_mol, key=lambda m: m["shift_mae_ppm"], reverse=True)[:32])
    for metric, fname, desc in [("shift_mae_ppm", "worst_shift_cases.json", True),
                                ("j_mae_hz", "worst_j_cases.json", True),
                                ("presence_f1", "worst_presence_cases.json", False),
                                ("deg_acc", "worst_deg_cases.json", False)]:
        store.write_json(f"{prefix}/{fname}",
                         sorted(per_mol, key=lambda m: m[metric], reverse=desc)[:32])
    return {**agg, "dominant_failure": dominant}


# =============================================================================
# Checkpoint
# =============================================================================

def _strip_compile(sd):
    """Drop the ``_orig_mod.`` prefix torch.compile adds, so an eager model (the
    GUI rebuilds one) can ``load_state_dict`` the saved weights cleanly."""
    pre = "_orig_mod."
    return {(k[len(pre):] if k.startswith(pre) else k): v for k, v in sd.items()}


def make_checkpoint(model, ema, optimizer, std, vocab, cfg, epoch, metrics):
    base = getattr(model, "_orig_mod", model)      # unwrap torch.compile, if any
    return {
        "model_state": _strip_compile({k: v.detach().cpu()
                                       for k, v in base.state_dict().items()}),
        "ema_state": (_strip_compile(ema.state_dict())
                      if (ema and ema.enabled) else None),
        "optimizer_state": optimizer.state_dict(),
        "standardizer": std.state_dict(),
        "vocab": list(vocab.vocab),
        "model_build": base.build_kwargs(),
        "cfg": dataclasses.asdict(cfg),
        "epoch": epoch,
        "metrics": metrics,
    }


# =============================================================================
# Training loop
# =============================================================================

def _resolve_out_dir(out_dir, cfg):
    if out_dir:
        return out_dir
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    env_root = os.environ.get("SPINHANCE_OUT", "").rstrip("/")
    if env_root:
        return f"{env_root}/session_{ts}"
    return str(Path("modelv2/runs") / f"session_{ts}")


def fit(records, assignment, cfg: TrainConfig, out_dir=None, cache=None,
        small=False):
    """Train and return (model, std, vocab). Writes the full session artifact tree.

    ``records`` already carry shifts/couplings/degeneracy (+ optionally spectra or
    spectrum paths); ``assignment`` maps mol_id -> fold; ``cache`` is an optional
    prebuilt :class:`data.SpectraCache` (serving spectra from RAM)."""
    import torch
    from modelv2.model import build_model

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch: "
            "micromamba install -n spinhance pytorch pytorch-cuda=12.4 -c pytorch -c nvidia -y"
        )
    device = cfg.device
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    store = RunStore(_resolve_out_dir(out_dir, cfg))
    run_id = store.root.rstrip("/").rsplit("/", 1)[-1]
    store.write_json("config.json", dataclasses.asdict(cfg))
    store.append_jsonl("events.jsonl", {"event": "run_start",
                                        "run_id": run_id, "device": device,
                                        "epochs": cfg.epochs})

    vocab = D.DegeneracyVocab()
    by_fold = {"train": [], "val": [], "test": []}
    for r in records:
        f = assignment.get(r["mol_id"])
        if f and len(r["shifts"]) == cfg.n_groups and vocab.contains(r["degeneracy"]):
            by_fold[f].append(r)
    train_recs, val_recs = by_fold["train"], by_fold["val"]
    assert train_recs and val_recs, "empty train/val fold after filtering"

    std = D.Standardizer().fit(train_recs, vocab)
    std.assert_valid()
    cb = D.class_balance(train_recs, vocab)
    balance = {
        "deg_weights": torch.tensor(cb["deg_weights"], device=device),
        "presence_pos_weight": torch.tensor(cb["presence_pos_weight"], device=device),
    }
    print(f"class balance: deg_counts={cb['deg_counts'].tolist()} "
          f"presence_pos_weight={cb['presence_pos_weight']:.2f}")

    ds_train = D.SpinDataset(train_recs, vocab, std, cache=cache,
                             spectrum_field=cfg.spectrum_field, points=cfg.points)
    ds_val = D.SpinDataset(val_recs, vocab, std, cache=cache,
                           spectrum_field=cfg.spectrum_field, points=cfg.points)
    pin = (device != "cpu")
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=pin, collate_fn=D.collate,
        persistent_workers=cfg.num_workers > 0)
    # shuffle=False so the bounded diagnostics sample (DIAG_MAX_SAMPLES) is the
    # same fixed subset every epoch — stable metrics for checkpoint selection.
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=pin, collate_fn=D.collate, persistent_workers=cfg.num_workers > 0)

    model = build_model(n_groups=cfg.n_groups, n_deg_classes=len(vocab),
                        small=small).to(device)
    if cfg.compile:
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[train] torch.compile failed ({e}); using eager")
    ema = EMA(model, cfg.ema_decay)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per = max(1, len(dl_train))
    total_steps = steps_per * cfg.epochs
    warmup = int(cfg.warmup_frac * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, warmup, total_steps))

    amp_ctx = _amp_context(cfg, device)

    # held-out diagnostic probe set (server-side guarantee of "never trained on")
    diag_records, diag_spectra = materialize_diagnostic_set(
        by_fold["test"], cache, cfg, n=500, seed=cfg.seed)
    store.write_json("diagnostic_set.json", diag_records)
    store.write_npy("diagnostic_spectra.npy", diag_spectra)
    baseline = constant_mean_baseline(train_recs, vocab, std)

    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed)
    aug_kwargs = dict(AUG_KWARGS)

    best, bad, best_epoch, last_metrics = float("inf"), 0, 0, {}
    global_step = 0
    skipped = 0

    for epoch in range(cfg.epochs):
        model.train()
        running = {}
        t_epoch = time.time()
        for batch in dl_train:
            spec = batch["spectrum"].to(device, non_blocking=True)
            target = {k: batch[k].to(device, non_blocking=True)
                      for k in ("shifts", "j_mag", "j_presence", "deg_class")}
            if cfg.gpu_augment:
                spec = D.augment_spectrum(spec, cfg.ppm_from, cfg.ppm_to,
                                          generator=gen, **aug_kwargs)
            opt.zero_grad(set_to_none=True)
            progress = global_step / max(1, total_steps)
            with amp_ctx():
                pred = model(spec)
            total, raw, weighted = compute_losses(pred, target, cfg, balance, progress)

            if not torch.isfinite(total):
                skipped += 1
                store.append_jsonl("events.jsonl", {"event": "nonfinite_loss",
                                                     "step": global_step, "epoch": epoch})
                if global_step > 50 and skipped / max(global_step, 1) > 0.1:
                    raise RuntimeError(f"non-finite loss skip rate too high ({skipped} skips)")
                global_step += 1
                continue

            total.backward()
            do_log = (global_step % cfg.log_every == 0)
            gnorms = head_grad_norms(model) if do_log else None
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ema.update(model)

            for k, v in raw.items():
                running[k] = running.get(k, 0.0) + float(v)
            running["total"] = running.get("total", 0.0) + float(total.detach())

            if do_log:
                ev = {"event": "train_step", "epoch": epoch, "step": global_step,
                      "lr": float(sched.get_last_lr()[0]),
                      "loss_total": float(total.detach()),
                      **{f"loss_{k}": float(v) for k, v in raw.items()},
                      **{f"wloss_{k}": float(v) for k, v in weighted.items()},
                      **(gnorms or {})}
                if device != "cpu":
                    ev["cuda_alloc_gb"] = torch.cuda.memory_allocated(device) / 1e9
                store.append_jsonl("events.jsonl", ev)
            global_step += 1

        n_steps = max(1, steps_per)
        train_loss = {f"train_{k}": v / n_steps for k, v in running.items()}

        # ---- validation / metrics / diagnostics on the EMA (shadow) model ----
        do_val = (epoch % cfg.val_every == 0) or (epoch == cfg.epochs - 1)
        va, diagnostics = {}, {}
        with _UseEMA(ema, model):
            if do_val:
                va, diagnostics, _ = evaluate(model, dl_val, std, vocab, device, amp_ctx)
                # train-vs-val gap on a bounded train sample
                tr_metrics, _, _ = evaluate(model, dl_train, std, vocab, device, amp_ctx,
                                            max_samples=min(DIAG_MAX_SAMPLES, 4000))
                gap = {f"gap_{k}": float(tr_metrics[k] - va[k])
                       for k in ("shift_mae_ppm", "j_mae_hz") if k in va and k in tr_metrics}
                probe_agg = run_probes(model, diag_records, diag_spectra, std, vocab,
                                       cfg, store, epoch, device, amp_ctx)
            else:
                gap, probe_agg = {}, {}

        if do_val:
            metrics_row = {**va, **baseline, **diagnostics, **gap,
                           **{f"train_{k}": v for k, v in tr_metrics.items()}}
            store.append_jsonl("metrics.jsonl", {"split": "val", "epoch": epoch,
                                                 "step": global_step, "metrics": metrics_row})
            store.append_jsonl("metrics.jsonl", {"split": "train", "epoch": epoch,
                                                 "step": global_step, "metrics": train_loss})
            last_metrics = va

        score = va.get("shift_mae_ppm", float("inf")) + va.get("j_mae_hz", float("inf")) / 10.0
        is_best = do_val and score < best
        if is_best:
            best, bad, best_epoch = score, 0, epoch
            store.append_jsonl("events.jsonl", {"event": "best", "epoch": epoch,
                                                "score": float(score), **va})
        elif do_val:
            bad += 1

        # ---- checkpoints (best on improvement; last periodically; per-epoch probe) ----
        ckpt = make_checkpoint(model, ema, opt, std, vocab, cfg, epoch, va)
        if is_best:
            store.save_torch("checkpoints/best.pt", ckpt)
        if (epoch % cfg.save_every == 0) or (epoch == cfg.epochs - 1) or is_best:
            store.save_torch("checkpoints/last.pt", ckpt)
        if do_val:
            store.save_torch(f"probes/epoch_{epoch:04d}/checkpoint.pt", ckpt)

        store.write_json("status.json", {
            "state": "running", "run_id": run_id, "epoch": epoch, "epochs": cfg.epochs,
            "global_step": global_step, "device": device,
            "best_score": (float(best) if best != float("inf") else None),
            "best_epoch": best_epoch, "last_update_time": time.time(),
            "checkpoint_best": store.full("checkpoints/best.pt"),
            "checkpoint_last": store.full("checkpoints/last.pt"),
        })

        print(f"epoch {epoch:3d} | train {train_loss.get('train_total', 0):.4f}"
              + (f" | val shift {va.get('shift_mae_ppm', 0):.3f}ppm "
                 f"J {va.get('j_mae_hz', 0):.2f}Hz f1 {va.get('presence_f1', 0):.3f} "
                 f"deg {va.get('deg_acc_balanced', 0):.3f} "
                 f"| var(s) {diagnostics.get('var_ratio_shift_mean', 0):.2f} "
                 f"r(s) {diagnostics.get('pearson_shift_mean', 0):.2f}"
                 if do_val else " | val skipped")
              + f" | {time.time() - t_epoch:.1f}s")

        if do_val and bad >= cfg.patience:
            store.append_jsonl("events.jsonl", {"event": "early_stop", "epoch": epoch})
            print(f"early stop at epoch {epoch}")
            break

    summary = {"run_id": run_id, "state": "finished", "best_epoch": best_epoch,
               "best_score": (float(best) if best != float("inf") else None),
               "best_metrics": last_metrics, "baseline": baseline,
               "score_formula": "shift_mae_ppm + j_mae_hz / 10"}
    store.write_json("summary.json", summary)
    store.write_json("status.json", {
        "state": "finished", "run_id": run_id, "epoch": best_epoch, "epochs": cfg.epochs,
        "global_step": global_step, "device": device,
        "best_score": (float(best) if best != float("inf") else None),
        "best_epoch": best_epoch, "last_update_time": time.time(),
        "checkpoint_best": store.full("checkpoints/best.pt"),
        "checkpoint_last": store.full("checkpoints/last.pt"),
    })
    store.append_jsonl("events.jsonl", {"event": "run_end", "best_epoch": best_epoch})
    # convenience copy of the best checkpoint to cfg.ckpt_path (local only)
    if cfg.ckpt_path and not is_s3(cfg.ckpt_path) and not store.s3:
        bp = Path(store.root) / "checkpoints" / "best.pt"
        if bp.exists():
            Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.ckpt_path).write_bytes(bp.read_bytes())
    print(f"Run complete -> {store.root}  (best epoch {best_epoch})")
    return model, std, vocab


def _amp_context(cfg, device):
    import contextlib
    import torch
    if cfg.amp_dtype == "none" or device == "cpu":
        return lambda: contextlib.nullcontext()
    dt = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    assert cfg.amp_dtype != "fp16", "fp16/GradScaler is forbidden; use bf16"
    return lambda: torch.autocast(device_type="cuda", dtype=dt)


# =============================================================================
# --dry-run : exercise the whole data path with no torch
# =============================================================================

def run_dry_run(args):
    """adapter -> splits -> standardizer -> target encoding -> renderable mask.

    Validates the hard correctness invariants and never imports torch."""
    assert args.spin_systems, "--dry-run needs --spin_systems"
    print(f"[dry-run] loading records from {args.spin_systems}")
    records = D.load_records(args.spin_systems, max_records=args.max_records)
    print(f"[dry-run] loaded {len(records)} records")
    vocab = D.DegeneracyVocab()

    mask = D.renderable_mask(records, vocab, n_groups=args.n_groups)
    usable = [r for r, ok in zip(records, mask) if ok]
    print(f"[dry-run] renderable (G={args.n_groups}, deg in vocab): "
          f"{len(usable)}/{len(records)}")
    assert usable, "no usable molecules"

    assignment, report = D.make_splits(usable, seed=args.seed,
                                       compute_scaffold=not args.no_scaffold)
    print(f"[dry-run] splits: {report['counts']} ratios="
          f"{{'train': {report['ratios']['train']:.3f}, "
          f"'val': {report['ratios']['val']:.3f}, 'test': {report['ratios']['test']:.3f}}} "
          f"groups={report['n_groups']} scaffold_leaks={report['scaffold_leaks']} "
          f"dup_leaks={report['dup_matrix_leaks']} used_scaffold={report['used_scaffold']}")
    assert report["scaffold_leaks"] == 0 and report["dup_matrix_leaks"] == 0

    train_recs = [r for r in usable if assignment.get(r["mol_id"]) == "train"]
    std = D.Standardizer().fit(train_recs, vocab)
    std.assert_valid()
    print(f"[dry-run] standardizer: shift mean={std.shift_mean:.3f} std={std.shift_std:.3f}; "
          f"J mean={std.j_mean:.3f} std={std.j_std:.3f}")

    # target encoding + index-map agreement
    G = args.n_groups
    iu = D.triu_index_map(G)
    n_pairs = G * (G - 1) // 2
    assert len(iu[0]) == n_pairs
    t = std.transform(D.encode_target(usable[0], vocab))
    assert t["shifts"].shape == (G,)
    assert t["j_mag"].shape == (n_pairs,) == t["j_presence"].shape
    assert t["deg_class"].shape == (G,)
    assert all(int(c) in range(len(vocab)) for c in t["deg_class"])
    # presence/J index maps agree (decode round-trip through the same map)
    M = D.pairs_to_matrix(t["j_mag"], G)
    assert np.allclose(D.matrix_to_pairs(M, G), t["j_mag"])

    cb = D.class_balance(train_recs, vocab)
    assert np.isfinite(cb["deg_weights"]).all() and cb["presence_pos_weight"] > 0
    print(f"[dry-run] class balance: deg_counts={cb['deg_counts'].tolist()} "
          f"presence_pos_weight={cb['presence_pos_weight']:.2f}")

    # optional spectra path
    if args.spectra and Path(args.spectra).exists():
        cache = D.SpectraCache(usable, args.spectra, points=args.points,
                               spectrum_field="spec90")
        hit = sum(1 for r in usable if r["mol_id"] in cache)
        print(f"[dry-run] spectra cache: {cache.n_loaded} loaded, {hit} join the records")
    else:
        print("[dry-run] spectra: skipped (no --spectra given or path missing)")

    test_recs = [r for r in usable if assignment.get(r["mol_id"]) == "test"]
    diag, _spec = materialize_diagnostic_set(test_recs, None,
                                             TrainConfig(points=args.points, seed=args.seed),
                                             n=min(500, len(test_recs)), seed=args.seed)
    print(f"[dry-run] diagnostic set: {len(diag)} held-out test molecules")
    print("[dry-run] PASSED — data path is sound, no torch touched.")


# =============================================================================
# --smoke : tiny synthetic end-to-end run (needs torch)
# =============================================================================

def run_smoke(args):
    rng = np.random.default_rng(0)
    G, P, N = 8, 1024, 160

    def mol(i):
        c = np.zeros((G, G))
        for a in range(G):
            for b in range(a + 1, G):
                if rng.random() < 0.4:
                    c[a, b] = c[b, a] = float(rng.uniform(1, 12))
        deg = rng.choice([1, 2, 3, 6], size=G).astype(int)
        shifts = np.sort(rng.uniform(0.5, 9, G))[::-1].copy()
        # a crude but input-dependent spectrum so the encoder has signal to learn
        x = np.linspace(0, 12, P)
        spec = np.zeros(P)
        for g in range(G):
            spec += deg[g] * np.exp(-0.5 * ((x - shifts[g]) / 0.05) ** 2)
        spec = np.clip(spec + rng.normal(0, 0.01, P), 0, None)
        spec /= spec.sum() * (12.0 / P)
        return dict(mol_id=f"mol_{i:06d}", smiles=None, shifts=shifts, couplings=c,
                    degeneracy=deg, spec90=spec.astype(np.float32), scaffold=f"s{i % 20}")

    recs = [mol(i) for i in range(N)]
    assignment, report = D.make_splits(recs, seed=0, compute_scaffold=False)
    print(f"[smoke] splits {report['counts']}")
    cfg = TrainConfig(points=P, batch_size=16, epochs=args.epochs or 3, lr=1e-3,
                      device=args.device, amp_dtype="none", cache_spectra=False,
                      gpu_augment=True, num_workers=0, warmup_frac=0.1,
                      save_every=1, log_every=5, patience=10,
                      ckpt_path=str(Path(args.out or "modelv2/runs/smoke") / "best.pt"))
    fit(recs, assignment, cfg, out_dir=(args.out or "modelv2/runs/smoke"),
        cache=None, small=True)
    print("[smoke] PASSED")


# =============================================================================
# CLI
# =============================================================================

def build_argparser():
    ap = argparse.ArgumentParser(description="modelv2 training / data-path CLI")
    ap.add_argument("--spin_systems", help="spin_systems json or .json.tar.gz (ground truth)")
    ap.add_argument("--spectra", help="90MHz.tar.gz (or a dir of mol_*.npy)")
    ap.add_argument("--out", default="", help="session root (local dir or s3:// URI)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ckpt", default=None, help="convenience copy path for best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--points", type=int, default=16384)
    ap.add_argument("--n_groups", type=int, default=8)
    ap.add_argument("--max_records", type=int, default=None,
                    help="cap records loaded (handy for quick runs)")
    ap.add_argument("--no_scaffold", action="store_true",
                    help="force matrix-dedup split (skip RDKit scaffold)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    if args.dry_run:
        run_dry_run(args)
        return
    if args.smoke:
        run_smoke(args)
        return

    assert args.spin_systems and args.spectra, \
        "training needs --spin_systems and --spectra (or use --dry-run / --smoke)"

    cfg = TrainConfig(points=args.points, n_groups=args.n_groups, seed=args.seed,
                      device=args.device)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.lr is not None:
        cfg.lr = args.lr
    if args.ckpt:
        cfg.ckpt_path = args.ckpt

    print(f"[train] loading ground truth: {args.spin_systems}")
    records = D.load_records(args.spin_systems, max_records=args.max_records)
    vocab = D.DegeneracyVocab()
    mask = D.renderable_mask(records, vocab, n_groups=cfg.n_groups)
    records = [r for r, ok in zip(records, mask) if ok]
    print(f"[train] usable records: {len(records)}")

    assignment, report = D.make_splits(records, seed=cfg.seed,
                                       compute_scaffold=not args.no_scaffold)
    print(f"[train] splits: {report['counts']} (scaffold_leaks={report['scaffold_leaks']}, "
          f"dup_leaks={report['dup_matrix_leaks']})")

    cache = None
    if cfg.cache_spectra:
        print(f"[train] building spectra RAM cache from {args.spectra}")
        cache = D.SpectraCache(records, args.spectra, points=cfg.points,
                               spectrum_field=cfg.spectrum_field)

    fit(records, assignment, cfg, out_dir=(args.out or None), cache=cache)


if __name__ == "__main__":
    import sys
    if "" not in sys.path:
        sys.path.insert(0, "")
    main()
