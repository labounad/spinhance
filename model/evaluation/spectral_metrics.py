"""
model.evaluation.spectral_metrics
=================================
Spectral-distance metrics for comparing a rendered spectrum to a reference
(Stage-2A exact no-grad evaluation). Torch-based, batched, differentiable —
usable both as eval metrics and, later, inside a spectral-consistency loss.

  wasserstein1  1-D earth-mover (|CDF_a - CDF_b|), degrades gracefully under
                small peak misalignment (preferred over intensity MSE).
  smoothed_mse  plain intensity MSE.
  cosine        cosine similarity of the (non-negative) intensity vectors.
"""
from __future__ import annotations

import torch

__all__ = ["wasserstein1", "smoothed_mse", "cosine_similarity"]


def _normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.sum(dim=-1, keepdim=True) + eps)


def wasserstein1(spec_a: torch.Tensor, spec_b: torch.Tensor, dx: float = 1.0,
                 eps: float = 1e-12) -> torch.Tensor:
    """1-D Wasserstein-1 between two non-negative spectra over a common grid.
    Batched over the leading dim; differentiable in both inputs."""
    ca = torch.cumsum(_normalize(spec_a, eps), dim=-1)
    cb = torch.cumsum(_normalize(spec_b, eps), dim=-1)
    return (ca - cb).abs().sum(dim=-1) * dx


def smoothed_mse(spec_a: torch.Tensor, spec_b: torch.Tensor) -> torch.Tensor:
    return ((spec_a - spec_b) ** 2).mean(dim=-1)


def cosine_similarity(spec_a: torch.Tensor, spec_b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = (spec_a * spec_b).sum(dim=-1)
    den = spec_a.norm(dim=-1) * spec_b.norm(dim=-1) + eps
    return num / den
