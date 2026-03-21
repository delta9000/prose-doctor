"""Sensory lens -- 6-axis perceptual analysis of prose.

Scores text on six sensory dimensions (visual, auditory, haptic,
olfactory, gustatory, interoceptive) using a distilled probe trained
on data-free pseudo-labels from Qwen3-Embedding-4B direction vectors
and GPT-2 cloze probabilities.

The probe runs on all-mpnet-base-v2 embeddings (768d) and ships as
a ~100KB .pt file with no external data dependencies.

Ported from prose_doctor.ml.sensory into the Lens interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs_with_breaks

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

PROBE_PATH = Path(__file__).resolve().parent.parent / "data" / "sensory_probe.pt"
MODALITIES = ["visual", "auditory", "haptic", "olfactory", "gustatory", "interoceptive"]


class _SensoryProbe:
    """Lazy-loading sensory modality probe.

    Self-contained: loads its own all-mpnet-base-v2 (768d) model,
    NOT the shared 384d sentence_transformer provider.
    """

    def __init__(self):
        self._probe = None
        self._st = None

    def _load(self):
        if self._probe is not None:
            return

        import torch
        import torch.nn as nn

        class _ProbeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(768, 96),
                    nn.ReLU(),
                    nn.Linear(96, 6),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        self._probe = _ProbeNet()
        self._probe.load_state_dict(torch.load(PROBE_PATH, weights_only=True))
        self._probe.eval()

        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer("all-mpnet-base-v2")

    def score_words(self, words: list[str]) -> np.ndarray:
        """Score a list of words on 6 sensory dimensions.

        Returns array of shape (len(words), 6) with values in [0, 1].
        """
        import torch

        self._load()
        embs = self._st.encode(words, show_progress_bar=False, batch_size=128)
        with torch.no_grad():
            preds = self._probe(torch.from_numpy(embs.astype(np.float32)))
        return preds.numpy()

    def score_text(self, text: str) -> np.ndarray:
        """Score content words in a text passage. Returns (6,) mean scores."""
        words = [w.lower() for w in text.split() if len(w) > 3 and w.isalpha()]
        if not words:
            return np.zeros(6)
        scores = self.score_words(words)
        return scores.mean(axis=0)


class SensoryLens(Lens):
    """Analyze sensory modality distribution across prose."""

    name = "sensory"
    requires_providers: list[str] = []
    consumes_lenses: list[str] = []

    def __init__(self, desert_threshold: int = 5) -> None:
        self.desert_threshold = desert_threshold
        self._probe = _SensoryProbe()

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        items = split_paragraphs_with_breaks(text)
        probe = self._probe

        para_profiles: list[np.ndarray] = []
        for item in items:
            if item is None:
                continue
            scores = probe.score_text(item)
            para_profiles.append(scores)

        if not para_profiles:
            return LensResult(
                lens_name=self.name,
                per_chapter={
                    "dominant_modality": "",
                    "weakest_modality": "",
                    "balance_ratio": 0.0,
                    **{m: 0.0 for m in MODALITIES},
                },
                per_paragraph={m: [] for m in MODALITIES},
                raw={"word_count": 0, "deserts": [], "paragraph_count": 0},
            )

        all_scores = np.array(para_profiles)
        means = all_scores.mean(axis=0)

        dominant_idx = int(means.argmax())
        weakest_idx = int(means.argmin())

        # Balance ratio: normalized entropy of modality distribution
        vals = means.copy()
        if vals.sum() == 0:
            balance_ratio = 0.0
        else:
            p = vals / vals.sum()
            p = p[p > 0]
            entropy = -float(np.sum(p * np.log2(p)))
            balance_ratio = entropy / np.log2(6)

        per_chapter: dict[str, float] = {
            "dominant_modality": MODALITIES[dominant_idx],
            "weakest_modality": MODALITIES[weakest_idx],
            "balance_ratio": round(float(balance_ratio), 4),
        }
        for i, m in enumerate(MODALITIES):
            per_chapter[m] = round(float(means[i]), 4)

        # Per-paragraph: list of scores per modality
        per_paragraph: dict[str, list[float]] = {
            m: [round(float(row[i]), 4) for row in para_profiles]
            for i, m in enumerate(MODALITIES)
        }

        # Detect sensory deserts: stretches where max sensory score is very low
        desert_thresh_val = float(np.percentile(all_scores.max(axis=1), 15))
        deserts: list[dict] = []
        run_start = None
        for i, scores in enumerate(all_scores):
            if scores.max() < desert_thresh_val:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and (i - run_start) >= self.desert_threshold:
                    deserts.append({"start": run_start, "end": i - 1, "length": i - run_start})
                run_start = None
        if run_start is not None and (len(all_scores) - run_start) >= self.desert_threshold:
            deserts.append({
                "start": run_start,
                "end": len(all_scores) - 1,
                "length": len(all_scores) - run_start,
            })

        # Prescription for weakest modality
        prescriptions = {
            "visual": "Add non-visual sensory detail -- what does the scene sound, smell, or feel like?",
            "auditory": "Ground a moment in sound -- ambient noise, voice texture, silence.",
            "haptic": "Add touch -- temperature, texture, weight, pressure.",
            "olfactory": "Add smell -- the most memory-linked sense. One specific scent anchors a scene.",
            "gustatory": "Add taste -- metallic fear, dry mouth, the flavor of a meal.",
            "interoceptive": "Add body sensation -- heartbeat, breath, gut feeling, muscle tension.",
        }

        raw = {
            "word_count": len(text.split()),
            "paragraph_count": len(para_profiles),
            "deserts": deserts,
            "prescription": prescriptions.get(MODALITIES[weakest_idx], ""),
            "filename": filename,
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            per_paragraph=per_paragraph,
            raw=raw,
        )
