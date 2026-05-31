"""
model.registry
==============
A tiny generic registry. Each layer (architectures, losses, renderers) owns one
instance, so a config can select a component by name and the trainer never hard-codes
class references.

    from model.registry import Registry
    ARCH = Registry("architecture")

    @ARCH.register("resnet1d")
    class ResNet1DModel(...): ...

    model = ARCH.build("resnet1d", **cfg)      # constructs by name
"""
from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def add(self, name: str, obj: T) -> None:
        """Imperative registration (for things that aren't decorated definitions)."""
        if name in self._items:
            raise KeyError(f"{self.kind} '{name}' already registered")
        self._items[name] = obj

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} '{name}'. available: {self.available()}")
        return self._items[name]

    def build(self, name: str, *args, **kwargs):
        """Look up by name and call it (constructor or factory)."""
        return self.get(name)(*args, **kwargs)

    def available(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)
