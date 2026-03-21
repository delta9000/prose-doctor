"""Lens framework — self-contained analytical perspectives on prose."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


@dataclass
class LensResult:
    """Standard output from any lens."""
    lens_name: str

    per_sentence: dict[str, list[float]] | None = None
    per_paragraph: dict[str, list[float]] | None = None
    per_scene: dict[str, list[float]] | None = None
    per_chapter: dict[str, float] | None = None

    issues: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class Lens(ABC):
    """Base class for all analytical lenses."""

    name: str = ""
    requires_providers: list[str] = []
    consumes_lenses: list[str] = []

    @abstractmethod
    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        ...


class LensRegistry:
    """Registry of available lens instances."""

    def __init__(self) -> None:
        self._lenses: dict[str, Lens] = {}

    def register(self, lens: Lens) -> None:
        if lens.name in self._lenses:
            raise ValueError(f"Lens '{lens.name}' already registered")
        self._lenses[lens.name] = lens

    def get(self, name: str) -> Lens | None:
        return self._lenses.get(name)

    def all_names(self) -> list[str]:
        return list(self._lenses.keys())

    def all_lenses(self) -> list[Lens]:
        return list(self._lenses.values())
