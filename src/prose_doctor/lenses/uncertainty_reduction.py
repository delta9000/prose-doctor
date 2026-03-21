"""Uncertainty reduction lens — entropy change at paragraph boundaries.

At each paragraph boundary, computes the entropy of GPT-2's next-token
distribution for the preceding context, then measures how much entropy
changes between consecutive paragraphs. Positive values mean the new
paragraph resolved uncertainty; negative values mean it introduced new
uncertainty.

Extracted from narrative_attention_proto.py.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs_with_breaks

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


# ---------- Internal helpers ----------


def _next_token_entropy(
    text: str, model, tokenizer, device, top_k: int = 100,
) -> float:
    """Compute top-k entropy of GPT-2's next-token distribution for *text*.

    Uses only the top-k most probable tokens (renormalized) to avoid
    the full-vocab entropy being dominated by noise.
    """
    import torch

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
    )
    if str(device) == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token logits
        probs = torch.softmax(logits, dim=-1)
        top_probs, _ = probs.topk(top_k)
        top_probs = top_probs / top_probs.sum()  # renormalize
        entropy = -float((top_probs * top_probs.log()).sum())

    return entropy


# ---------- Lens class ----------


class UncertaintyReductionLens(Lens):
    """Measure entropy change at paragraph boundaries via GPT-2."""

    name = "uncertainty_reduction"
    requires_providers: list[str] = ["gpt2"]
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        model, tokenizer = providers.gpt2
        device = providers.device

        items = split_paragraphs_with_breaks(text)
        paragraphs = [p for p in items if p is not None and p.strip()]

        n = len(paragraphs)

        if n < 2:
            return LensResult(
                lens_name=self.name,
                per_paragraph={"uncertainty_reduction": [0.0] * n},
                per_chapter={"mean_reduction": 0.0},
            )

        print(
            f"  Computing uncertainty reduction for {n} paragraphs...",
            file=sys.stderr,
            flush=True,
        )

        reductions: list[float] = []
        prev_entropy: float | None = None

        for i, para in enumerate(paragraphs):
            # Build context: last ~200 tokens worth of preceding text
            context = " ".join(paragraphs[max(0, i - 2) : i + 1])
            curr_entropy = _next_token_entropy(context, model, tokenizer, device)

            if prev_entropy is not None:
                reductions.append(prev_entropy - curr_entropy)
            else:
                reductions.append(0.0)

            prev_entropy = curr_entropy

        mean_reduction = sum(reductions) / len(reductions) if reductions else 0.0

        return LensResult(
            lens_name=self.name,
            per_paragraph={"uncertainty_reduction": [round(r, 4) for r in reductions]},
            per_chapter={"mean_reduction": round(mean_reduction, 4)},
        )
