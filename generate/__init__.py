"""
generate — SpinHance Task 1 (molecule screening pipeline).

Screens the ChEMBL compound database to identify small organic molecules
that have exactly :data:`N_SPIN_GROUPS` magnetically distinct ¹H spin
groups, making them suitable training candidates for the downstream NMR
simulation (Task 3) and machine-learning (Task 4) stages.

Module map
----------
- :mod:`generate.config`         — package-wide constants (N_SPIN_GROUPS etc.).
- :mod:`generate.chembl_filter`  — fast heuristic pre-filter over ChEMBL.
- :mod:`generate.spin_equivalence` — 3-D deuterium substitution test.
- :mod:`generate.screen`         — batch application of the equivalence test.
- :mod:`generate.viewer`         — interactive Tkinter triage GUI.
- :mod:`generate.cli`            — ``spinhance-gen`` command-line entry point.

Quick start
-----------
Run the two-stage pipeline from the command line::

    spinhance-gen run                       # ChEMBL → chembl_8spin.csv
    spinhance-gen view                      # opens chembl_8spin.csv in viewer

Or call the Python API directly::

    from generate import analyze_spin_systems, N_SPIN_GROUPS
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CC1CC(C)CC(C)C1")
    n, sizes = analyze_spin_systems(mol)
    print(f"{n} spin groups: {sizes}")
"""

from .config import (
    N_SPIN_GROUPS,
    MAX_PROTON_BEARING_C,
    MIN_PROTONS,
    EMBED_RANDOM_SEED,
    EMBED_MAX_OPT_ITERS,
)
from .spin_equivalence import (
    passes_heuristic,
    embed_3d,
    strip_exchangeable_protons,
    substitution_signature,
    analyze_spin_systems,
)
from .pipeline import run_pipeline

__all__ = [
    # config
    "N_SPIN_GROUPS",
    "MAX_PROTON_BEARING_C",
    "MIN_PROTONS",
    "EMBED_RANDOM_SEED",
    "EMBED_MAX_OPT_ITERS",
    # spin_equivalence
    "passes_heuristic",
    "embed_3d",
    "strip_exchangeable_protons",
    "substitution_signature",
    "analyze_spin_systems",
    # pipeline
    "run_pipeline",
]
