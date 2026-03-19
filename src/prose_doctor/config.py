"""Configuration loading from .prose-doctor.toml files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_CONFIG_FILENAME = ".prose-doctor.toml"


@dataclass
class ProjectConfig:
    """Project-level configuration for prose-doctor."""

    character_names: list[str] = field(default_factory=list)
    exempt_words: set[str] = field(default_factory=set)
    pov_map: dict[str, list[str]] = field(default_factory=dict)
    density_budgets: dict[str, int] = field(default_factory=dict)
    slop_classifier_model: str = ""

    @classmethod
    def load(cls, directory: Path | None = None) -> ProjectConfig:
        """Load config from .prose-doctor.toml, walking up from directory.

        If no config file is found, returns defaults.
        """
        search = directory or Path.cwd()
        for parent in [search, *search.parents]:
            config_path = parent / _CONFIG_FILENAME
            if config_path.is_file():
                return cls._from_toml(config_path)
        return cls()

    @classmethod
    def _from_toml(cls, path: Path) -> ProjectConfig:
        """Parse a .prose-doctor.toml file."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        section = raw.get("prose-doctor", {})

        character_names = section.get("character_names", [])
        exempt_words = set(section.get("exempt_words", []))
        pov_map = section.get("pov", {})
        density_budgets = section.get("density_budgets", {})

        models = section.get("models", {})
        slop_classifier_model = models.get("slop_classifier", "")

        return cls(
            character_names=character_names,
            exempt_words=exempt_words,
            pov_map=pov_map,
            density_budgets=density_budgets,
            slop_classifier_model=slop_classifier_model,
        )

    @classmethod
    def default_template(cls) -> str:
        """Return a default .prose-doctor.toml template."""
        return """\
[prose-doctor]
character_names = []
exempt_words = []

[prose-doctor.pov]
# Character = ["chapter_01", "chapter_04"]

[prose-doctor.density_budgets]
# tricolon = 2

[prose-doctor.models]
# slop_classifier = "dt9k/prose-doctor-slop-classifier"
"""
