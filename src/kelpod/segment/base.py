"""Segmenter protocol and backend registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from kelpod.contracts import RasterPage, Segmentation


@runtime_checkable
class Segmenter(Protocol):
    """Anything that turns one cleaned page into a :class:`Segmentation` (image space)."""

    name: str

    def segment(self, page: RasterPage) -> Segmentation: ...


_REGISTRY: dict[str, Callable[[], Segmenter]] = {}


def register(name: str, factory: Callable[[], Segmenter]) -> None:
    """Register a backend factory under ``name`` (replaces an existing one)."""
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_segmenter(name: str) -> Segmenter:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown segmentation backend {name!r}; available: {', '.join(available())}"
        ) from None
    return factory()
