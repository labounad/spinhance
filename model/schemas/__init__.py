"""
model.schemas
=============
Typed data contracts that every layer communicates through. Importing from here
keeps the dependency direction clean: data -> SpinBatch; architectures ->
ModelOutput; losses -> LossOutput; renderers -> RendererOutput.
"""
from model.schemas import constants
from model.schemas.batch import RegionTokenBatch, SpinBatch
from model.schemas.outputs import ModelOutput
from model.schemas.losses import LossOutput
from model.schemas.renderers import RendererOutput
from model.schemas.diagnostics import MetricRecord, RunStatus, RunSummary

__all__ = [
    "constants",
    "SpinBatch",
    "RegionTokenBatch",
    "ModelOutput",
    "LossOutput",
    "RendererOutput",
    "RunStatus",
    "MetricRecord",
    "RunSummary",
]
