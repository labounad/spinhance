"""
Spectrum -> ModelOutput architectures. Importing this package registers the
built-in models (resnet1d, resnet1d_attention_pool) in ARCHITECTURES.
"""
from model.architectures.registry import ARCHITECTURES, build_architecture
from model.architectures.base import SpinArchitecture

# Import concrete models so their @ARCHITECTURES.register decorators run.
from model.architectures import resnet1d as _resnet1d       # noqa: F401
from model.architectures import attention_pool as _attn     # noqa: F401

__all__ = ["ARCHITECTURES", "build_architecture", "SpinArchitecture"]
