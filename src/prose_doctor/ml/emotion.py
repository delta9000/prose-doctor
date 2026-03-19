"""Sentiment arc analysis: track emotional intensity across a chapter."""

from __future__ import annotations

from prose_doctor.text import split_paragraphs


class EmotionArcAnalyzer:
    """Tracks emotional intensity using sentiment scoring.

    Flags chapters with flat arcs (no peaks/valleys).
    """

    FLAT_ARC_THRESHOLD = 0.15

    def __init__(self, model_manager=None):
        self._mm = model_manager
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

    def analyze_chapter(self, text: str, filename: str = "") -> dict:
        """Analyze emotional arc of a chapter."""
        import numpy as np

        paras = split_paragraphs(text)
        if len(paras) < 5:
            return {"filename": filename, "arc": [], "flat": False, "stats": {}}

        scores = []
        indices = []
        for i in range(0, len(paras), 3):
            if len(paras[i].split()) < 10:
                continue
            score = self._score_intensity(paras[i])
            scores.append(score)
            indices.append(i)

        if len(scores) < 3:
            return {"filename": filename, "arc": [], "flat": False, "stats": {}}

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

        return {
            "filename": filename,
            "flat": flat,
            "arc": arc_str,
            "stats": {
                "mean": mean,
                "std": std,
                "dynamic_range": dynamic_range,
                "peaks": len(peaks),
                "valleys": len(valleys),
            },
            "arc_summary": arc_summary,
        }
