"""
simulation — SpinHance Task 3 (spin simulation pipeline).

Converts shift+J+degeneracy matrices to ``mnova-spinsim`` XML, drives MestReNova
to simulate ¹H spectra at low (90 MHz) and high (600.15 MHz) field, and exports
normalised intensity arrays as ``.npy``.

Module map
----------
- :mod:`simulation.xml_io`       — matrix ⇄ mnova-spinsim XML (pure).
- :mod:`simulation.mnova_runner` — MestReNova CLI invocation.
- :mod:`simulation.pipeline`     — patch → simulate → convert orchestration.
- :mod:`simulation.plotting`     — QC plots.
- :mod:`simulation.cli`          — command-line entry point.

Quick start
-----------
>>> from simulation import matrix_to_xml, save_xml
>>> tree = matrix_to_xml([3.0, 7.5], [[0, 8], [8, 0]], [1, 1], frequency_mhz=90.0)
>>> save_xml(tree, "/tmp/ax.xml")
"""

from .xml_io import (
    matrix_to_xml,
    save_xml,
    patch_frequency,
    generate_field_pair,
    LOW_FIELD_MHZ,
    HIGH_FIELD_MHZ,
)
from .pipeline import (
    prepare_xmls,
    txt_to_npy,
    run_pipeline,
    DEFAULT_FIELDS_MHZ,
    N_POINTS,
    PPM_FROM,
    PPM_TO,
)
from .mnova_runner import run_mnova_batch, MNOVA_DEFAULT

__all__ = [
    # xml_io
    "matrix_to_xml",
    "save_xml",
    "patch_frequency",
    "generate_field_pair",
    "LOW_FIELD_MHZ",
    "HIGH_FIELD_MHZ",
    # pipeline
    "prepare_xmls",
    "txt_to_npy",
    "run_pipeline",
    "DEFAULT_FIELDS_MHZ",
    "N_POINTS",
    "PPM_FROM",
    "PPM_TO",
    # mnova_runner
    "run_mnova_batch",
    "MNOVA_DEFAULT",
]
