"""Human/LLM corpus management for lens validation."""
from __future__ import annotations
from pathlib import Path


def load_corpus(directory: Path) -> list[tuple[str, str]]:
    """Load all .md files from a directory. Returns [(filename, text), ...]."""
    files = sorted(directory.glob("**/*.md"))
    return [(f.name, f.read_text()) for f in files if f.stat().st_size > 100]
