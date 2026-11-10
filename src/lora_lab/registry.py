"""Registries for the four independent extension axes.

New methods, targets, quantizers, and experiments register themselves here. The
runner and the analysis code never import concrete implementations directly;
they look components up by name.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

METHODS: dict[str, type[Any]] = {}
TARGETS: dict[str, type[Any]] = {}
QUANTIZERS: dict[str, type[Any]] = {}
EXPERIMENTS: dict[str, type[Any]] = {}


def register(registry: dict[str, T], name: str) -> Callable[[T], T]:
    """Return a decorator that registers ``item`` under ``name``.

    Raises:
        ValueError: if ``name`` has already been registered.
    """

    def deco(item: T) -> T:
        if name in registry:
            raise ValueError(f"duplicate registration in {registry!r}: {name}")
        registry[name] = item
        return item

    return deco
