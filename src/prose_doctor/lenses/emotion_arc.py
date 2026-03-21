"""Emotion arc lens — track emotional intensity across a chapter.

Uses a distilbert sentiment pipeline to score paragraph intensity,
then identifies peaks, valleys, and flat arcs.

Ported from prose_doctor.ml.emotion into the Lens interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


class EmotionArcLens(Lens):
    """Analyze emotional arc of a chapter via sentiment scoring."""

    name = "emotion_arc"
    requires_providers: list[str] = []
    consumes_lenses: list[str] = []

    FLAT_ARC_THRESHOLD = 0.15

    def __init__(self):
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        from transformers import pipeline

        self._pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
            max_length=512,
        )

    def _score_intensity(self, text: str) -> float:
        self._load()
        result = self._pipeline(text[:512])[0]
        return result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        import numpy as np

        paras = split_paragraphs(text)

        # Not enough content for meaningful arc
        if len(paras) < 5:
            return LensResult(
                lens_name=self.name,
                per_chapter={
                    "flat": 0.0,
                    "dynamic_range": 0.0,
                    "mean_sentiment": 0.0,
                    "peaks": 0.0,
                    "valleys": 0.0,
                },
                raw={"arc": "", "stats": {}},
            )

        # Sample every 3rd paragraph (must have >= 10 words)
        scores = []
        indices = []
        for i in range(0, len(paras), 3):
            if len(paras[i].split()) < 10:
                continue
            score = self._score_intensity(paras[i])
            scores.append(score)
            indices.append(i)

        if len(scores) < 3:
            return LensResult(
                lens_name=self.name,
                per_chapter={
                    "flat": 0.0,
                    "dynamic_range": 0.0,
                    "mean_sentiment": 0.0,
                    "peaks": 0.0,
                    "valleys": 0.0,
                },
                raw={"arc": "", "stats": {}},
            )

        scores_arr = np.array(scores)
        std = float(np.std(scores_arr))
        mean = float(np.mean(scores_arr))

        peaks = []
        valleys = []
        for i in range(1, len(scores) - 1):
            if scores[i] > scores[i - 1] and scores[i] > scores[i + 1]:
                peaks.append((indices[i], scores[i]))
            if scores[i] < scores[i - 1] and scores[i] < scores[i + 1]:
                valleys.append((indices[i], scores[i]))

        if peaks and valleys:
            dynamic_range = max(s for _, s in peaks) - min(s for _, s in valleys)
        else:
            dynamic_range = max(scores) - min(scores)

        flat = std < self.FLAT_ARC_THRESHOLD

        # Build arc summary string
        third = len(scores) // 3
        arc_summary = {
            "opening": float(np.mean(scores_arr[:third])) if third > 0 else mean,
            "middle": float(np.mean(scores_arr[third:2 * third])) if third > 0 else mean,
            "closing": float(np.mean(scores_arr[2 * third:])) if third > 0 else mean,
        }

        arc_str = (
            f"{arc_summary['opening']:.2f}->"
            f"{arc_summary['middle']:.2f}->"
            f"{arc_summary['closing']:.2f}"
        )

        per_chapter = {
            "flat": 1.0 if flat else 0.0,
            "dynamic_range": dynamic_range,
            "mean_sentiment": mean,
            "peaks": float(len(peaks)),
            "valleys": float(len(valleys)),
        }

        raw = {
            "arc": arc_str,
            "stats": {
                "std": std,
                "arc_summary": arc_summary,
            },
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            raw=raw,
        )
