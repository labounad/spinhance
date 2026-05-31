"""
model.losses.composite
======================
Config-driven weighted sum of loss terms with optional curriculum ramps. Lets a
config blend, e.g., a matrix anchor with a (later, ramped-in) surrogate spectral
term without any architecture or trainer changes:

    loss:
      name: composite
      terms:
        - {name: matrix, weight: 1.0}
        - {name: surrogate_spectral, weight: 0.1, start_epoch: 40, ramp_epochs: 10}

The trainer calls ``set_epoch`` once per epoch; term weights ramp linearly from
``start_epoch`` over ``ramp_epochs``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from model.losses.base import Loss
from model.losses.registry import LOSSES, build_loss
from model.schemas import LossOutput, ModelOutput, SpinBatch


@dataclass
class Term:
    name: str
    loss: Loss
    weight: float = 1.0
    start_epoch: int = 0
    ramp_epochs: int = 0


@LOSSES.register("composite")
class CompositeLoss(Loss):
    name = "composite"

    def __init__(self, terms: list[Term]):
        self.terms = terms
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for t in self.terms:
            t.loss.set_epoch(epoch)

    def _weight(self, t: Term) -> float:
        if self.epoch < t.start_epoch:
            return 0.0
        if t.ramp_epochs > 0:
            frac = min(1.0, (self.epoch - t.start_epoch + 1) / t.ramp_epochs)
            return t.weight * frac
        return t.weight

    def __call__(self, output: ModelOutput, batch: SpinBatch) -> LossOutput:
        total = output.shifts.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        metrics: dict[str, float] = {}
        active = 0
        for t in self.terms:
            w = self._weight(t)
            metrics[f"weight/{t.name}"] = float(w)
            if w == 0.0:
                continue
            lo = t.loss(output, batch)
            total = total + w * lo.total
            components[t.name] = lo.total.detach()
            for k, v in lo.metrics.items():
                metrics[f"{t.name}/{k}"] = v
            active += 1
        return LossOutput(total=total, components=components, metrics=metrics,
                          diagnostics={"active_terms": active})


def build_composite(terms_cfg: list[dict], **shared) -> CompositeLoss:
    """Construct a CompositeLoss from config term dicts.

    ``shared`` kwargs (e.g. deg_class_weight, presence_pos_weight) are forwarded
    to each sub-loss constructor; unknown kwargs are ignored per sub-loss via the
    registry call (sub-losses accept what they need).
    """
    terms = []
    for tc in terms_cfg:
        tc = dict(tc)
        name = tc.pop("name")
        weight = tc.pop("weight", 1.0)
        start = tc.pop("start_epoch", 0)
        ramp = tc.pop("ramp_epochs", 0)
        kwargs = {**tc}
        for k, v in shared.items():
            kwargs.setdefault(k, v)
        sub = build_loss(name, **_filter_kwargs(name, kwargs))
        terms.append(Term(name=name, loss=sub, weight=weight,
                          start_epoch=start, ramp_epochs=ramp))
    return CompositeLoss(terms)


def _filter_kwargs(name: str, kwargs: dict) -> dict:
    """Drop kwargs a sub-loss constructor doesn't accept (keeps shared injection safe)."""
    import inspect
    cls = LOSSES.get(name)
    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}
