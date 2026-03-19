"""Slop category taxonomy and class definitions."""

from __future__ import annotations

from enum import Enum


class SlopCategory(Enum):
    """Categories of LLM prose patterns."""

    CLEAN = "clean"
    THESIS = "thesis"
    EMOTION = "emotion"
    DEAD_FIGURE = "dead_figure"
    STANDALONE = "standalone"
    NARRATOR_GLOSS = "narrator_gloss"
    FORBIDDEN = "forbidden"
    PADDING = "padding"


CLASS_NAMES = [cat.value for cat in SlopCategory]
NUM_CLASSES = len(CLASS_NAMES)
