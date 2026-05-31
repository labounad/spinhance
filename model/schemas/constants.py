"""
model.schemas.constants
========================
Shared problem dimensions and canonical artifact names. Everything that needs
"how many groups" or "how many spectral points" reads it from here.
"""
from __future__ import annotations

# Problem dimensions ------------------------------------------------------------
N_GROUPS = 8           # spin groups per molecule (the S_G-symmetric set)
N_POINTS = 16384       # spectral grid points (2**14) over PPM_FROM..PPM_TO
PPM_FROM = 0.0
PPM_TO = 12.0

# Number of unordered group pairs (upper triangle), i.e. distinct couplings.
N_PAIRS = N_GROUPS * (N_GROUPS - 1) // 2   # 28 for G=8

# Default degeneracy vocabulary (protons per group). Index order is the class id
# used by the degeneracy classification head. Mirrors model_legacy DEFAULT_DEG_VOCAB
# (kept identical so real data never KeyErrors; pruning to observed classes is a
# future optimization).
DEFAULT_DEG_VOCAB = (1, 2, 3, 4, 6, 9, 12, 18)

# Canonical run-directory artifact names (the diagnostics contract). AutoAI and
# the dashboard read these names; do not rename without updating both.
RUN_CONFIG = "config.json"
RUN_STATUS = "status.json"
RUN_METRICS = "metrics.jsonl"
RUN_EVENTS = "events.jsonl"
RUN_SUMMARY = "summary.json"
CHECKPOINTS_DIR = "checkpoints"
PROBES_DIR = "probes"
