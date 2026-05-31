"""
model.renderers.exact
=====================
Exact quantum spin renderer (composite-particle reduction + Mz block
diagonalisation), ported from the legacy package and wrapped in the typed
``RendererOutput`` contract.

Two registered modes (master plan §Renderer):
  exact_no_grad               default — render under torch.no_grad() for
                              evaluation / probes / surrogate-teacher targets.
  exact_autograd_experimental opt-in, differentiable, tiny systems only; NEVER
                              the default Stage-2 loss (this is the path that ran
                              out of memory). Strict cost guard + diagnostics.

Both guard on ``max_block_dim`` (the largest Mz block actually diagonalised) and
skip — reporting the reason — when a molecule exceeds ``max_block``.
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch

from model.renderers._composite import max_block_dim
from model.renderers import _torch_exact as _engine
from model.renderers.base import Renderer
from model.renderers.registry import RENDERERS
from model.schemas import RendererOutput
from model.schemas.constants import N_POINTS, PPM_FROM, PPM_TO


def _as_t(x, device, dtype):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


class ExactRenderer(Renderer):
    """Wrap the exact torch engine; render one molecule -> RendererOutput."""

    def __init__(self, *, no_grad: bool = True, max_block: int = 4096,
                 points: int = N_POINTS, ppm_from: float = PPM_FROM, ppm_to: float = PPM_TO,
                 linewidth_hz: float = 1.0, eigh_eps: float = 1.0,
                 device: str = "cpu", dtype: torch.dtype = torch.float64):
        self.no_grad = no_grad
        self.max_block = max_block
        self.points = points
        self.ppm_from, self.ppm_to = ppm_from, ppm_to
        self.linewidth_hz = linewidth_hz
        self.eigh_eps = eigh_eps
        self.device = device
        self.dtype = dtype

    def render(self, shifts, couplings, degeneracy, field_mhz, **kwargs) -> RendererOutput:
        deg = [int(d) for d in np.asarray(degeneracy).ravel().tolist()]
        cost = int(max_block_dim(deg))
        if cost > self.max_block:
            return RendererOutput(
                spectrum=None,
                diagnostics={"rendered": False, "skip_reason": "max_block_exceeded",
                             "max_block_dim": cost, "max_block_limit": self.max_block})
        s = _as_t(shifts, self.device, self.dtype).reshape(-1)
        c = _as_t(couplings, self.device, self.dtype)
        ctx = torch.no_grad() if self.no_grad else nullcontext()
        t0 = time.time()
        with ctx:
            _, spec = _engine.simulate(s, c, deg, float(field_mhz),
                                       points=self.points, ppm_from=self.ppm_from,
                                       ppm_to=self.ppm_to, linewidth_hz=self.linewidth_hz,
                                       eigh_eps=self.eigh_eps)
        return RendererOutput(
            spectrum=spec.reshape(1, -1),
            metrics={"render_seconds": time.time() - t0},
            diagnostics={"rendered": True, "max_block_dim": cost,
                         "grad": not self.no_grad})


@RENDERERS.register("exact_no_grad")
def _build_exact_no_grad(**kwargs):
    kwargs.setdefault("no_grad", True)
    return ExactRenderer(**kwargs)


@RENDERERS.register("exact_autograd_experimental")
def _build_exact_autograd(**kwargs):
    # Experimental, opt-in, tiny systems only. Never the default Stage-2 loss.
    kwargs.setdefault("no_grad", False)
    kwargs.setdefault("max_block", 256)
    return ExactRenderer(**kwargs)
