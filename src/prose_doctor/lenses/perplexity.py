"""Perplexity lens — GPT-2 perplexity scoring to detect AI-smooth prose.

Low perplexity = predictable/template prose.
High perplexity = surprising/human prose.

Ported from prose_doctor.ml.perplexity into the Lens interface.
"""
from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


SMOOTH_THRESHOLD = 45
WARNING_THRESHOLD = 55


def _paragraph_perplexity(text: str, model, tokenizer, device) -> float:
    """Compute perplexity for a single text passage."""
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if str(device) == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return math.exp(outputs.loss.item())


class PerplexityLens(Lens):
    """Detect AI-smooth prose via GPT-2 perplexity scoring."""

    name = "perplexity"
    requires_providers: list[str] = ["gpt2"]
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        import numpy as np

        model, tokenizer = providers.gpt2
        device = providers.device

        paras = split_paragraphs(text)
        if not paras:
            return LensResult(
                lens_name=self.name,
                per_chapter={"mean_ppl": 0.0, "pct_below_55": 0.0},
                per_paragraph={"perplexity": []},
                raw={"smoothest_paragraphs": [], "stats": {}},
            )

        scores: list[float] = []
        smooth: list[dict] = []

        print(
            f"  Scoring {len(paras)} paragraphs for perplexity...",
            file=sys.stderr,
            flush=True,
        )

        for pi, para in enumerate(paras):
            if len(para.split()) < 10:
                continue
            ppl = _paragraph_perplexity(para, model, tokenizer, device)
            scores.append(ppl)
            if ppl < WARNING_THRESHOLD:
                smooth.append({
                    "index": pi,
                    "perplexity": ppl,
                    "severity": "smooth" if ppl < SMOOTH_THRESHOLD else "warn",
                    "text": para[:120].replace("\n", " "),
                })

        if not scores:
            return LensResult(
                lens_name=self.name,
                per_chapter={"mean_ppl": 0.0, "pct_below_55": 0.0},
                per_paragraph={"perplexity": []},
                raw={"smoothest_paragraphs": [], "stats": {}},
            )

        stats = {
            "mean_ppl": float(np.mean(scores)),
            "median_ppl": float(np.median(scores)),
            "pct_below_55": (
                sum(1 for s in scores if s < 55) / len(scores) * 100
            ),
        }

        smoothest = sorted(smooth, key=lambda x: x["perplexity"])

        per_chapter = {
            "mean_ppl": round(stats["mean_ppl"], 2),
            "pct_below_55": round(stats["pct_below_55"], 1),
        }

        per_paragraph = {
            "perplexity": [round(s, 2) for s in scores],
        }

        raw = {
            "smoothest_paragraphs": smoothest[:8],
            "stats": stats,
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            per_paragraph=per_paragraph,
            raw=raw,
        )
