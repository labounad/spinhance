"""
autoai/eval_harness.py — immutable eval harness interface.

*** STUB — implement when eval data is available. ***

CONTRACT (do not break):
  - run_eval() is called by the orchestrator harness after each worker run.
  - It is NOT exposed as a model tool — models cannot call or read this file.
  - Eval data lives under data/eval/ which is also blocked from model reads.
  - This file must never be modified by either agent (enforced by tool_read_file guard).

TO IMPLEMENT:
  1. Populate data/eval/ with held-out spectra + ground-truth matrices.
     These must never appear in training data.
  2. Replace the stub body of run_eval() with real scoring logic:
     - Load the checkpoint from checkpoint_path.
     - Run inference on every spectrum in data/eval/.
     - Compare predictions to ground truth using a permutation-invariant metric
       (e.g. Hungarian-matched MAE on shifts and couplings).
  3. Return the metrics dict — it will be merged into the experiment record
     and fed to the orchestrator as verified ground-truth performance.
"""

from __future__ import annotations

from pathlib import Path

EVAL_DATA_DIR = Path(__file__).parent.parent / "data" / "eval"


def run_eval(checkpoint_path: str | None) -> dict:
    """
    Evaluate a trained model checkpoint against the held-out eval set.

    Parameters
    ----------
    checkpoint_path : repo-relative path to the model checkpoint (.pt file),
                      or None if no checkpoint was produced.

    Returns
    -------
    dict  Eval metrics, e.g.:
          {"mae_shift_ppm": 0.12, "mae_j_hz": 1.4, "pct_j_within_1hz": 0.83}
          Empty dict if the stub is not yet implemented.
    """
    # *** STUB *** — no eval data exists yet.
    # Remove this block and implement scoring when data/eval/ is populated.
    if not EVAL_DATA_DIR.exists() or not any(EVAL_DATA_DIR.iterdir()):
        return {}

    raise NotImplementedError(
        "eval_harness.run_eval() is not implemented. "
        "Populate data/eval/ and replace this stub."
    )
