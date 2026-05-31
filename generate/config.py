"""generate/config.py — package-wide constants for the generate pipeline.

Every numeric threshold that governs molecule selection is defined here.
Downstream modules import from this file rather than embedding literals,
so changing the target spin-group count (or any other threshold) propagates
everywhere with a single edit.

To retarget the whole pipeline from 8 to, say, 6 spin groups::

    # config.py
    N_SPIN_GROUPS = 6          # everything else adjusts automatically
"""

from __future__ import annotations

# ── Primary target ─────────────────────────────────────────────────────────────

#: Legacy single-target spin-group count (the original ChEMBL screen selected
#: for exactly this many groups).  Still the default when no range is given, so
#: ``run`` with no ``--min/--max-groups`` reproduces the historical behaviour.
N_SPIN_GROUPS: int = 8

#: Inclusive spin-group range for a *categorising* scan.  Setting the pipeline
#: to ``[MIN_SPIN_GROUPS, MAX_SPIN_GROUPS]`` lets a single pass over the
#: database bucket every molecule by its spin-group count instead of running
#: one screen per count.  26 is the natural ceiling: spin groups are labelled
#: with single letters A–Z, so ≤26 keeps one-character labels.
MIN_SPIN_GROUPS: int = 1
MAX_SPIN_GROUPS: int = 26

# ── Pre-filter heuristics ──────────────────────────────────────────────────────
#
# The heuristic bounds are now derived per-run from the requested group range
# (carbons ≤ max_groups, protons ≥ min_groups); these module constants remain
# as the defaults for the legacy single-target path.

#: Maximum number of carbon atoms bearing ≥1 hydrogen a molecule may have to
#: pass the cheap first-pass filter.  A molecule with more CH carbons than the
#: target maximum would need extreme symmetry to collapse to it, which is rare.
MAX_PROTON_BEARING_C: int = N_SPIN_GROUPS

#: Minimum total C-H proton count required: a molecule with fewer protons than
#: the target minimum cannot possibly fill that many groups.
MIN_PROTONS: int = N_SPIN_GROUPS

# ── 3-D conformer generation ──────────────────────────────────────────────────

#: Random seed passed to ETKDG v3 for conformer generation.  Kept fixed
#: for reproducibility across pipeline runs.  Changing this may occasionally
#: rescue molecules whose embedding fails at the default seed.
EMBED_RANDOM_SEED: int = 0xC0FFEE

#: Maximum optimisation iterations for MMFF94 (and the UFF fallback).
#: 300 is sufficient for drug-like molecules; increase for large macrocycles.
EMBED_MAX_OPT_ITERS: int = 300
