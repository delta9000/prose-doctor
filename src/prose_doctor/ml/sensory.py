"""Sensory modality profiler: 6-axis perceptual analysis of prose.

Scores text on six sensory dimensions (visual, auditory, haptic,
olfactory, gustatory, interoceptive) using a distilled probe trained
on data-free pseudo-labels from Qwen3-Embedding-4B direction vectors
and GPT-2 cloze probabilities.

The probe runs on all-mpnet-base-v2 embeddings (768d) and ships as
a ~100KB .pt file with no external data dependencies.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prose_doctor.text import split_paragraphs_with_breaks

PROBE_PATH = Path(__file__).resolve().parent.parent / "data" / "sensory_probe.pt"
MODALITIES = ["Visual", "Auditory", "Haptic", "Olfactory", "Gustatory", "Interoceptive"]


class SensoryProbe:
    """Lazy-loading sensory modality probe."""

    def __init__(self, model_manager=None):
        self._mm = model_manager
        self._probe = None
        self._st = None

    def _load(self):
        if self._probe is not None:
            return

        import torch
        import torch.nn as nn

        class _Probe(nn.Module):
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

        self._probe = _Probe()
        self._probe.load_state_dict(torch.load(PROBE_PATH, weights_only=True))
        self._probe.eval()

        # Always load mpnet — the probe was trained on 768d mpnet embeddings,
        # not the 384d MiniLM that ModelManager uses for other tasks.
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
        # Extract content words (skip stopwords and short words)
        words = [w.lower() for w in text.split() if len(w) > 3 and w.isalpha()]
        if not words:
            return np.zeros(6)
        scores = self.score_words(words)
        return scores.mean(axis=0)


@dataclass
class SensoryProfile:
    """Sensory profile for a chapter."""

    filename: str
    word_count: int
    # Per-modality mean scores (0-1)
    visual: float
    auditory: float
    haptic: float
    olfactory: float
    gustatory: float
    interoceptive: float
    # Per-paragraph profiles
    paragraph_profiles: list[np.ndarray] = field(default_factory=list, repr=False)
    # Diagnostics
    dominant_modality: str = ""
    weakest_modality: str = ""
    deserts: list[dict] = field(default_factory=list)  # stretches with no sensory language

    @property
    def scores(self) -> dict[str, float]:
        return {
            "visual": self.visual,
            "auditory": self.auditory,
            "haptic": self.haptic,
            "olfactory": self.olfactory,
            "gustatory": self.gustatory,
            "interoceptive": self.interoceptive,
        }

    @property
    def balance_ratio(self) -> float:
        """How evenly distributed are the modalities (0=monopoly, 1=uniform)."""
        vals = np.array(list(self.scores.values()))
        if vals.sum() == 0:
            return 0.0
        # Normalized entropy
        p = vals / vals.sum()
        p = p[p > 0]
        entropy = -float(np.sum(p * np.log2(p)))
        max_entropy = np.log2(6)
        return entropy / max_entropy

    @property
    def prescription(self) -> str:
        prescriptions = {
            "visual": "Add non-visual sensory detail — what does the scene sound, smell, or feel like?",
            "auditory": "Ground a moment in sound — ambient noise, voice texture, silence.",
            "haptic": "Add touch — temperature, texture, weight, pressure.",
            "olfactory": "Add smell — the most memory-linked sense. One specific scent anchors a scene.",
            "gustatory": "Add taste — metallic fear, dry mouth, the flavor of a meal.",
            "interoceptive": "Add body sensation — heartbeat, breath, gut feeling, muscle tension.",
        }
        return prescriptions.get(self.weakest_modality, "")


def profile_chapter(
    text: str,
    filename: str,
    model_manager,
    desert_threshold: int = 5,
) -> SensoryProfile:
    """Profile a chapter's sensory modality distribution.

    Args:
        text: chapter text
        filename: for reporting
        model_manager: provides sentence transformer
        desert_threshold: paragraphs with no dominant sense for this many
            consecutive paragraphs are flagged as sensory deserts
    """
    probe = SensoryProbe(model_manager)
    items = split_paragraphs_with_breaks(text)

    para_profiles = []
    for item in items:
        if item is None:
            continue
        scores = probe.score_text(item)
        para_profiles.append(scores)

    if not para_profiles:
        return SensoryProfile(
            filename=filename, word_count=0,
            visual=0, auditory=0, haptic=0,
            olfactory=0, gustatory=0, interoceptive=0,
        )

    # Chapter-level means
    all_scores = np.array(para_profiles)
    means = all_scores.mean(axis=0)

    mod_names = [m.lower() for m in MODALITIES]
    dominant_idx = int(means.argmax())
    weakest_idx = int(means.argmin())

    # Detect sensory deserts: stretches where max sensory score is very low
    desert_thresh_val = np.percentile(all_scores.max(axis=1), 15)
    deserts = []
    run_start = None
    for i, scores in enumerate(all_scores):
        if scores.max() < desert_thresh_val:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= desert_threshold:
                deserts.append({"start": run_start, "end": i - 1, "length": i - run_start})
            run_start = None
    if run_start is not None and (len(all_scores) - run_start) >= desert_threshold:
        deserts.append({"start": run_start, "end": len(all_scores) - 1,
                        "length": len(all_scores) - run_start})

    return SensoryProfile(
        filename=filename,
        word_count=len(text.split()),
        visual=round(float(means[0]), 4),
        auditory=round(float(means[1]), 4),
        haptic=round(float(means[2]), 4),
        olfactory=round(float(means[3]), 4),
        gustatory=round(float(means[4]), 4),
        interoceptive=round(float(means[5]), 4),
        paragraph_profiles=para_profiles,
        dominant_modality=mod_names[dominant_idx],
        weakest_modality=mod_names[weakest_idx],
        deserts=deserts,
    )
