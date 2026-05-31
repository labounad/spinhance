"""Evaluation: metrics, Hungarian matching, probe evaluator, failure analysis."""
from model.evaluation.metrics import compute_metrics, decode, evaluate_output
from model.evaluation.hungarian import hungarian_perm
from model.evaluation.probes import ProbeEvaluator
from model.evaluation.failure_analysis import per_sample_evaluate, save_failure_cases

__all__ = ["compute_metrics", "decode", "evaluate_output", "hungarian_perm",
           "ProbeEvaluator", "per_sample_evaluate", "save_failure_cases"]
