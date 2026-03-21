"""Boyd narrative role lens -- staging/progression/tension classification.

Brian Boyd's narrative theory identifies three functional modes in prose:
- Staging: setting the scene (spatial/temporal words, articles, prepositions)
- Progression: advancing the plot (causal/sequential words, action verbs)
- Tension: conflict and stakes (negation, intensifiers, contrast)

Each paragraph is scored by counting function words from each set,
normalized by total word count. Pure word-set counting -- no ML needed.

Extracted from narrative_attention_proto.py.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs_with_breaks

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

# Boyd function-word sets
STAGING_WORDS = frozenset({
    "where", "when", "the", "a", "an", "in", "on", "at", "through",
    "before", "after", "during", "until", "between", "within", "around",
    "above", "below", "near", "across",
})

PROGRESSION_WORDS = frozenset({
    "then", "so", "because", "therefore", "consequently", "next",
    "finally", "suddenly", "immediately", "quickly", "began", "started",
    "continued", "turned", "moved", "went",
})

TENSION_WORDS = frozenset({
    "not", "never", "but", "however", "although", "yet", "still",
    "even", "only", "just", "almost", "barely", "despite", "against",
    "without",
})


def _score_paragraph(text: str) -> tuple[float, float, float]:
    """Score a paragraph on staging, progression, and tension.

    Returns (staging, progression, tension) as fractions of word count.
    """
    words = text.lower().split()
    n = len(words)
    if n == 0:
        return (0.0, 0.0, 0.0)

    staging = sum(1 for w in words if w in STAGING_WORDS) / n
    progression = sum(1 for w in words if w in PROGRESSION_WORDS) / n
    tension = sum(1 for w in words if w in TENSION_WORDS) / n
    return (staging, progression, tension)


class BoydNarrativeRoleLens(Lens):
    """Classify paragraphs by Boyd narrative role (staging/progression/tension)."""

    name = "boyd_narrative_role"
    requires_providers: list[str] = []
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        items = split_paragraphs_with_breaks(text)

        staging_scores: list[float] = []
        progression_scores: list[float] = []
        tension_scores: list[float] = []
        dominant_modes: list[str] = []

        for item in items:
            if item is None:
                continue
            s, p, t = _score_paragraph(item)
            staging_scores.append(round(s, 4))
            progression_scores.append(round(p, 4))
            tension_scores.append(round(t, 4))

            # Determine dominant mode
            scores = {"staging": s, "progression": p, "tension": t}
            dominant_modes.append(max(scores, key=scores.get))

        n = len(staging_scores)
        if n == 0:
            return LensResult(
                lens_name=self.name,
                per_chapter={"staging": 0.0, "progression": 0.0, "tension": 0.0},
                per_paragraph={"staging": [], "progression": [], "tension": []},
                raw={"dominant_modes": []},
            )

        per_chapter = {
            "staging": round(sum(staging_scores) / n, 4),
            "progression": round(sum(progression_scores) / n, 4),
            "tension": round(sum(tension_scores) / n, 4),
        }

        per_paragraph = {
            "staging": staging_scores,
            "progression": progression_scores,
            "tension": tension_scores,
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            per_paragraph=per_paragraph,
            raw={"dominant_modes": dominant_modes},
        )
