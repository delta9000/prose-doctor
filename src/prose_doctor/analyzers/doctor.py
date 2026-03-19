"""Unified diagnosis: runs all core analyzers and produces a ChapterHealth report."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from prose_doctor.analyzers.density import DensityAnalyzer
from prose_doctor.analyzers.proof_scanner import ProofScanner
from prose_doctor.analyzers.vocabulary import find_vocabulary_crutches
from prose_doctor.config import ProjectConfig
from prose_doctor.text import count_words, split_paragraphs


def _find_colon_lists(text: str) -> list[dict]:
    """Find 'X: noun, noun, noun, noun' list constructions."""
    hits = []
    pat = re.compile(r":\s+(?:the\s+)?[a-z]+(?:,\s+(?:the\s+)?[a-z]+){2,}")
    for i, line in enumerate(text.split("\n"), 1):
        if pat.search(line.lower()):
            hits.append({
                "pattern": "colon_list",
                "line": i,
                "text": line.strip()[:120],
                "severity": "style",
            })
    return hits


@dataclass
class ChapterHealth:
    """Complete health report for a chapter."""

    filename: str
    word_count: int
    vocabulary_crutches: list[dict] = field(default_factory=list)
    pattern_hits: list[dict] = field(default_factory=list)
    colon_lists: list[dict] = field(default_factory=list)
    density_over_budget: dict[str, int] = field(default_factory=dict)
    # ML fields (None = not run)
    perplexity: dict | None = None
    foregrounding: dict | None = None
    emotion: dict | None = None

    @property
    def total_issues(self) -> int:
        return (
            sum(c["excess"] for c in self.vocabulary_crutches)
            + len(self.pattern_hits)
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "filename": self.filename,
            "word_count": self.word_count,
            "total_issues": self.total_issues,
            "vocabulary_crutches": self.vocabulary_crutches,
            "pattern_hits": self.pattern_hits,
            "colon_lists": self.colon_lists,
            "density_over_budget": self.density_over_budget,
            "perplexity": self.perplexity,
            "foregrounding": self.foregrounding,
            "emotion": self.emotion,
        }


def diagnose(
    text: str,
    filename: str = "",
    config: ProjectConfig | None = None,
) -> ChapterHealth:
    """Run all core diagnostics on a chapter.

    Returns a ChapterHealth report. Does not run ML analysis — use
    the ML analyzers separately for that.
    """
    cfg = config or ProjectConfig()

    scanner = ProofScanner(character_names=cfg.character_names)
    findings = scanner.scan(text)
    pattern_hits = [
        {
            "pattern": f.category,
            "line": f.line,
            "match": f.match,
            "severity": f.severity,
        }
        for f in findings
    ]

    density = DensityAnalyzer(
        character_names=cfg.character_names,
        budget_overrides=cfg.density_budgets or None,
    )
    density_report = density.analyze(text, filename=filename)

    return ChapterHealth(
        filename=filename,
        word_count=count_words(text),
        vocabulary_crutches=find_vocabulary_crutches(
            text, exempt_words=cfg.exempt_words
        ),
        pattern_hits=pattern_hits,
        colon_lists=_find_colon_lists(text),
        density_over_budget=density_report.over_budget,
        perplexity=None,
        foregrounding=None,
        emotion=None,
    )
