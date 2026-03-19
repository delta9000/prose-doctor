"""GPT-2 perplexity scoring to detect AI-smooth prose."""

from __future__ import annotations

import math

from prose_doctor.text import split_paragraphs


class PerplexityScorer:
    """GPT-2 perplexity scoring.

    Low perplexity = predictable/template prose.
    High perplexity = surprising/human prose.
    """

    SMOOTH_THRESHOLD = 45
    WARNING_THRESHOLD = 55

    def __init__(self, model_manager=None):
        self._mm = model_manager
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        if self._mm:
            self._model, self._tokenizer = self._mm.gpt2
            self._device = self._mm.device
        else:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self._model = GPT2LMHeadModel.from_pretrained("gpt2")
            self._model.eval()
            if torch.cuda.is_available():
                try:
                    self._model = self._model.cuda()
                    self._device = "cuda"
                except Exception:
                    self._device = "cpu"
            else:
                self._device = "cpu"

    def score(self, text: str) -> float:
        """Return perplexity for a text passage."""
        import torch

        self._load()
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs["input_ids"])
        return math.exp(outputs.loss.item())

    def score_chapter(self, text: str, filename: str = "") -> dict:
        """Score all paragraphs in a chapter."""
        paras = split_paragraphs(text)
        if not paras:
            return {"filename": filename, "paragraphs": 0, "smooth": [], "stats": {}}

        scores = []
        smooth = []
        for pi, para in enumerate(paras):
            if len(para.split()) < 10:
                continue
            ppl = self.score(para)
            scores.append(ppl)
            if ppl < self.WARNING_THRESHOLD:
                smooth.append({
                    "index": pi,
                    "perplexity": ppl,
                    "severity": "smooth" if ppl < self.SMOOTH_THRESHOLD else "warn",
                    "text": para[:120].replace("\n", " "),
                })

        import numpy as np

        stats = {
            "mean_ppl": float(np.mean(scores)) if scores else 0,
            "median_ppl": float(np.median(scores)) if scores else 0,
            "pct_below_55": (
                sum(1 for s in scores if s < 55) / len(scores) * 100
                if scores
                else 0
            ),
        }

        smoothest = sorted(smooth, key=lambda x: x["perplexity"])
        return {
            "filename": filename,
            "paragraphs": len(scores),
            "mean_ppl": stats["mean_ppl"],
            "pct_below_55": stats["pct_below_55"],
            "smoothest_paragraphs": smoothest[:8],
            "smooth": smoothest,
            "stats": stats,
        }
