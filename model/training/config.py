"""
model.training.config
=====================
YAML run configuration with dotted ``--set key=value`` overrides. Typed sections
for run/data/training/diagnostics; model and loss stay as plain dicts (flexible
component configs consumed by the registries).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunCfg:
    name: str = "run"
    output_dir: str = "model/runs"
    dir: str = ""                # explicit run dir; if set, used verbatim (no ts/hash suffix)


@dataclass
class DataCfg:
    records: str = ""
    spectra: str = ""
    field: int = 90
    split: str = "scaffold"      # "scaffold" | "none"
    max_mol: int = 0
    region_tokens: bool = False  # extract support-region tokens (Family D/E/H); off = unchanged
    region_max: int = 48         # max regions per spectrum (padded/truncated)
    # NOTE: the attribute `field` (NMR field) above shadows dataclasses.field in
    # this class body, so qualify it for the mutable default.
    region_kwargs: dict = dataclasses.field(default_factory=dict)


@dataclass
class TrainingCfg:
    epochs: int = 80
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    amp: str = "bf16"            # bf16 | fp16 | none
    warmup_frac: float = 0.03
    lr_min_factor: float = 0.05  # cosine-decay floor (LR never below this * peak)
    lr_stable_frac: float = 0.0  # WSD: fraction of steps to hold peak LR before decaying
    patience: int = 10
    num_workers: int = 0
    val_every: int = 1
    save_every: int = 1          # also save checkpoints/epoch_NNNN.pt every N epochs (0 = best/last only)
    seed: int = 0
    device: str | None = None


@dataclass
class DiagnosticsCfg:
    enabled: bool = True
    log_every_steps: int = 25
    probe_every_epochs: int = 5
    probe_count: int = 16


@dataclass
class Config:
    run: RunCfg
    data: DataCfg
    training: TrainingCfg
    diagnostics: DiagnosticsCfg
    model: dict[str, Any] = field(default_factory=lambda: {"name": "resnet1d", "size": "small"})
    loss: dict[str, Any] = field(default_factory=lambda: {"name": "composite",
                                                          "terms": [{"name": "matrix", "weight": 1.0}]})
    raw: dict[str, Any] = field(default_factory=dict)


# ── overrides ──────────────────────────────────────────────────────────────────

def _parse_scalar(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":            # YAML null only; "none" stays a string (e.g. amp=none)
        return None
    return v


def _apply_override(d: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = _parse_scalar(value)


def _section(raw: dict, key: str, cls):
    sub = raw.get(key, {}) or {}
    allowed = {f.name for f in fields(cls)}
    unknown = set(sub) - allowed
    if unknown:
        raise ValueError(f"unknown {key} config keys: {sorted(unknown)}")
    return cls(**{k: v for k, v in sub.items() if k in allowed})


def config_from_dict(raw: dict) -> Config:
    return Config(
        run=_section(raw, "run", RunCfg),
        data=_section(raw, "data", DataCfg),
        training=_section(raw, "training", TrainingCfg),
        diagnostics=_section(raw, "diagnostics", DiagnosticsCfg),
        model=raw.get("model", {"name": "resnet1d", "size": "small"}),
        loss=raw.get("loss", {"name": "composite", "terms": [{"name": "matrix", "weight": 1.0}]}),
        raw=raw,
    )


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got '{ov}'")
        key, value = ov.split("=", 1)
        _apply_override(raw, key.strip(), value.strip())
    return config_from_dict(raw)
