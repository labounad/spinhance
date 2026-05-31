"""
SpinHance model package (modular rebuild).

Layers communicate only through the typed contracts in ``model.schemas``:
    data -> SpinBatch -> architecture -> ModelOutput -> loss -> LossOutput

See model/README.md for the package map. The previous flat implementation is
the pre-rebuild flat layout (see git history). Kept intentionally import-light (no torch at
package import) so ``from model import s3io`` stays cheap for AutoAI.
"""
