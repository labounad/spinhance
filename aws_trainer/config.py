"""
aws_trainer.config
===================
Single source of truth for all hyperparameters.  JSON-serializable so every
checkpoint carries its own config and runs are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class VAWSConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    json_path: str = "mol_to_matrix/data/spin_systems.json"
    spectra_root: str = "simulation/data/spectra"
    spectrum_field: str = "spec90"           # key prefix used in records
    preload_spectra: bool = True             # load all spectra to RAM at startup

    # ── Splits ────────────────────────────────────────────────────────────────
    split_ratios: tuple = (0.70, 0.20, 0.10)
    split_seed: int = 42
    scaffold_split: bool = True

    # ── Model ─────────────────────────────────────────────────────────────────
    model_size: str = "medium"               # tiny|small|medium|large|*-attn variants
    n_groups: int = 8
    dropout: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size: int = 256
    accum_steps: int = 1                     # gradient accumulation (Stage 1 only)
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    epochs: int = 100
    warmup_frac: float = 0.03
    min_lr_frac: float = 0.05

    # ── Stage 1 / Stage 2 ─────────────────────────────────────────────────────
    stage1_epochs: int = 70
    ramp_epochs: int = 10
    spectral_max: float = 1.0
    matrix_anchor: float = 0.3
    stage2_enabled: bool = True
    render_subset_frac: float = 0.15
    linewidth_hz: float = 1.0
    eigh_eps: float = 1.0

    # ── Loss weights ──────────────────────────────────────────────────────────
    loss_shift: float = 1.0
    loss_jmag: float = 1.0
    loss_presence: float = 0.5
    loss_deg: float = 0.5

    # ── Spectral grid (must match the precomputed spectra) ─────────────────────
    points: int = 16384
    ppm_from: float = 0.0
    ppm_to: float = 12.0
    field_low: float = 90.0

    # ── AMP / compile ─────────────────────────────────────────────────────────
    amp_dtype: str = "bf16"                  # bf16 | fp16 | none
    compile_model: bool = False              # torch.compile (~20-30% faster, slower start)

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema_decay: float = 0.9999

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug_noise_frac: float = 0.01
    aug_ref_shift_ppm: float = 0.01
    aug_baseline_amp_frac: float = 0.02

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ckpt_dir: str = "aws_trainer/checkpoints"
    save_every: int = 10                     # save a checkpoint every N epochs
    patience: int = 15

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir: str = "aws_trainer/runs"
    wandb_project: str = ""                  # empty = disabled
    wandb_entity: str = ""
    run_name: str = ""

    # ── DataLoader ────────────────────────────────────────────────────────────
    num_workers: int = 4

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42

    # ── Derived (read-only, not serialized) ───────────────────────────────────
    @property
    def loss_weights(self) -> dict:
        return {"shift": self.loss_shift, "jmag": self.loss_jmag,
                "presence": self.loss_presence, "deg": self.loss_deg}

    @property
    def total_epochs(self) -> int:
        return self.epochs

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "VAWSConfig":
        with open(path) as f:
            d = json.load(f)
        # Convert lists back to tuples for tuple fields
        for k in ("split_ratios",):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_args(cls, args) -> "VAWSConfig":
        """Build from argparse namespace, applying CLI overrides over defaults."""
        if args.config:
            cfg = cls.from_json(args.config)
        else:
            cfg = cls()
        overrides = {
            "json_path":         args.json,
            "spectra_root":      args.spectra,
            "model_size":        args.model_size,
            "batch_size":        args.batch,
            "accum_steps":       args.accum,
            "lr":                args.lr,
            "epochs":            args.epochs,
            "stage1_epochs":     args.stage1_epochs,
            "ramp_epochs":       args.ramp_epochs,
            "stage2_enabled":    not args.no_stage2,
            "render_subset_frac": args.render_frac,
            "amp_dtype":         args.amp,
            "compile_model":     args.compile,
            "ckpt_dir":          args.ckpt_dir,
            "run_name":          args.run_name,
            "wandb_project":     args.wandb_project,
            "preload_spectra":   not args.no_preload,
            "num_workers":       args.workers,
            "seed":              args.seed,
            "scaffold_split":    not args.no_scaffold,
        }
        # Only apply overrides that were explicitly set (non-None/non-default)
        d = asdict(cfg)
        for k, v in overrides.items():
            if v is not None:
                d[k] = v
        for k in ("split_ratios",):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
