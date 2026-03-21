"""Slop classifier lens — ModernBERT + rule hybrid slop detection.

Wraps SlopScorer as an internal class, scoring each paragraph for
LLM-generated prose patterns. Returns per-paragraph probabilities
and chapter-level summary stats.

Ported from prose_doctor.ml.slop_scorer into the Lens interface.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from prose_doctor.config import ProjectConfig
from prose_doctor.lenses import Lens, LensResult
from prose_doctor.patterns.rules import build_rule_patterns, check_rules
from prose_doctor.patterns.taxonomy import CLASS_NAMES, NUM_CLASSES
from prose_doctor.text import split_paragraphs_with_breaks

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

RULE_SLOP_PROB = 0.95
DEFAULT_HF_CHECKPOINT = "dt9k/prose-doctor-slop-classifier"


# ---------- Internal scorer (kept from ml/slop_scorer.py) ----------


class SlopScorer:
    """Lazy-loading wrapper around the trained slop classifier.

    Automatically detects multi-class (8-label) vs legacy binary (2-label).
    Uses context-aware triplet scoring with (prev, current, next) paragraphs.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: ProjectConfig | None = None,
    ):
        cfg = config or ProjectConfig()
        self._checkpoint = str(
            checkpoint or cfg.slop_classifier_model or DEFAULT_HF_CHECKPOINT
        )
        self._model = None
        self._tokenizer = None
        self._device = None
        self._is_multiclass = False
        self._rule_patterns = build_rule_patterns(cfg.character_names or None)

    def _load(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._checkpoint
        )
        self._model.eval()
        self._is_multiclass = self._model.config.num_labels == NUM_CLASSES

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._device == "cuda":
            try:
                self._model = self._model.cuda()
            except Exception:
                print("CUDA OOM -- falling back to CPU", file=sys.stderr)
                self._device = "cpu"

    def score_text_with_context(
        self, prev: str, current: str, next_: str
    ) -> dict:
        """Score a paragraph with surrounding context."""
        import torch

        self._load()

        if self._is_multiclass:
            text_a = prev or ""
            text_b = current + (" [SEP] " + next_ if next_ else "")
            inputs = self._tokenizer(
                text_a, text_b,
                return_tensors="pt", truncation=True, max_length=768,
            )
        else:
            inputs = self._tokenizer(
                current, return_tensors="pt", truncation=True, max_length=512,
            )

        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        if self._is_multiclass:
            class_id = probs.argmax().item()
            class_prob = probs[class_id].item()
            slop_prob = 1.0 - probs[0].item()
            class_name = (
                CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"
            )
        else:
            slop_prob = probs[1].item()
            class_id = 0 if slop_prob <= 0.5 else 1
            class_prob = probs[class_id].item()
            class_name = "slop" if class_id == 1 else "clean"

        # Rule-based override
        rule_matches = check_rules(current, self._rule_patterns)
        rules_fired = [m["pattern_name"] for m in rule_matches]
        if rule_matches and slop_prob < RULE_SLOP_PROB:
            match = rule_matches[0]
            slop_prob = RULE_SLOP_PROB
            class_id = match["class_id"]
            class_name = match["class_name"]
            class_prob = RULE_SLOP_PROB

        return {
            "slop_prob": slop_prob,
            "class_id": class_id,
            "class_name": class_name,
            "class_prob": class_prob,
            "rules": rules_fired,
        }

    def score_paragraphs(self, text: str) -> list[dict]:
        """Score all paragraphs with context triplets."""
        items = split_paragraphs_with_breaks(text)
        results = []
        para_index = 0

        for i, item in enumerate(items):
            if item is None:
                continue

            prev = ""
            if i > 0 and items[i - 1] is not None:
                prev = items[i - 1]

            next_ = ""
            if i < len(items) - 1 and items[i + 1] is not None:
                next_ = items[i + 1]

            scored = self.score_text_with_context(prev, item, next_)
            results.append({
                "index": para_index,
                "text": item,
                "slop_prob": scored["slop_prob"],
                "words": len(item.split()),
                "class_name": scored["class_name"],
                "class_prob": scored["class_prob"],
                "rules": scored.get("rules", []),
            })
            para_index += 1

        results.sort(key=lambda r: r["slop_prob"], reverse=True)
        return results


# ---------- Lens class ----------


class SlopClassifierLens(Lens):
    """Score prose paragraphs for LLM-generated patterns."""

    name = "slop_classifier"
    requires_providers: list[str] = []
    consumes_lenses: list[str] = []

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: ProjectConfig | None = None,
        threshold: float = 0.5,
    ):
        self._scorer = SlopScorer(checkpoint=checkpoint, config=config)
        self._threshold = threshold

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        scored = self._scorer.score_paragraphs(text)

        if not scored:
            return LensResult(
                lens_name=self.name,
                per_chapter={"flagged_pct": 0.0, "mean_slop": 0.0},
                per_paragraph={"slop_prob": []},
                raw={"scored": []},
            )

        flagged = [s for s in scored if s["slop_prob"] > self._threshold]
        mean_slop = sum(s["slop_prob"] for s in scored) / len(scored)

        # per_paragraph slop_prob in original paragraph order
        by_index = sorted(scored, key=lambda s: s["index"])
        slop_probs = [s["slop_prob"] for s in by_index]

        per_chapter = {
            "flagged_pct": len(flagged) / len(scored) * 100,
            "mean_slop": mean_slop,
        }

        raw = {
            "total_paragraphs": len(scored),
            "flagged_count": len(flagged),
            "max_slop": scored[0]["slop_prob"],
            "scored": scored,
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            per_paragraph={"slop_prob": slop_probs},
            raw=raw,
        )
