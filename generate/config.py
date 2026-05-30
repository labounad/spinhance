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

#: Number of magnetically distinct ¹H spin groups the pipeline selects for.
#: This is *the* magic number that every filter, viewer, and test consults.
#: Change it once here; every downstream module picks up the new value.
N_SPIN_GROUPS: int = 8

# ── ChEMBL pre-filter heuristics ──────────────────────────────────────────────

#: Maximum number of carbon atoms bearing ≥1 hydrogen a molecule may have
#: to pass the cheap first-pass filter.  A molecule with more CH carbons
#: than N_SPIN_GROUPS would require enough symmetry to collapse them down to
#: N_SPIN_GROUPS, which is rare in ChEMBL drug-space.  This upper bound
#: removes the obvious over-counted molecules before the expensive 3-D test.
MAX_PROTON_BEARING_C: int = N_SPIN_GROUPS

#: Minimum total C-H proton count required for a molecule to have
#: N_SPIN_GROUPS non-empty spin groups.  A molecule with fewer protons
#: than N_SPIN_GROUPS cannot possibly fill all groups.
MIN_PROTONS: int = N_SPIN_GROUPS

# ── 3-D conformer generation ──────────────────────────────────────────────────

#: Random seed passed to ETKDG v3 for conformer generation.  Kept fixed
#: for reproducibility across pipeline runs.  Changing this may occasionally
#: rescue molecules whose embedding fails at the default seed.
EMBED_RANDOM_SEED: int = 0xC0FFEE

#: Maximum optimisation iterations for MMFF94 (and the UFF fallback).
#: 300 is sufficient for drug-like molecules; increase for large macrocycles.
EMBED_MAX_OPT_ITERS: int = 300
