"""Narrative attention meta-lens -- cosine attention from consumed lens outputs.

This is a meta-lens: it doesn't analyze text directly. Instead it assembles
a per-paragraph feature matrix from other lenses' outputs, computes a cosine
similarity attention matrix, and derives structural metrics.

Algorithm (from narrative_attention_proto.py):
1. Gather per_paragraph features from consumed lenses
2. Broadcast per_chapter scalars to all paragraphs when per_paragraph is absent
3. Z-score normalize each feature column
4. Cosine similarity attention: A[i,j] = cos(features[i], features[j])
5. Per-paragraph attention entropy (softmax row → Shannon entropy)
6. Chapter-level coherence: mean of adjacent diagonal entries A[i, i+1]
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

# Features to extract from each consumed lens.
# (lens_name, key, source) where source is "per_paragraph" or "per_chapter".
_FEATURE_SPEC: list[tuple[str, str, str]] = [
    ("psychic_distance", "pd_mean", "per_paragraph"),
    ("info_contour", "mean_surprisal", "per_paragraph"),
    ("boyd_narrative_role", "staging", "per_paragraph"),
    ("boyd_narrative_role", "progression", "per_paragraph"),
    ("boyd_narrative_role", "tension", "per_paragraph"),
    ("uncertainty_reduction", "uncertainty_reduction", "per_paragraph"),
    # Chapter-level features broadcast to all paragraphs
    ("foregrounding", "sl_cv", "per_chapter"),
    ("foregrounding", "inversion_pct", "per_chapter"),
    ("foregrounding", "fragment_pct", "per_chapter"),
    ("emotion_arc", "mean_sentiment", "per_chapter"),
    ("emotion_arc", "dynamic_range", "per_chapter"),
]


def _gather_features(
    consumed: dict[str, LensResult],
    n_paragraphs: int,
) -> tuple[np.ndarray, list[str]]:
    """Build (N, D) feature matrix from consumed lens results.

    Returns (features, feature_names).  Skips features whose source lens
    is missing from *consumed*.
    """
    columns: list[np.ndarray] = []
    names: list[str] = []

    for lens_name, key, source in _FEATURE_SPEC:
        result = consumed.get(lens_name)
        if result is None:
            continue

        if source == "per_paragraph":
            data = result.per_paragraph
            if data is None or key not in data:
                continue
            values = list(data[key])
            # Pad or truncate to match n_paragraphs
            while len(values) < n_paragraphs:
                values.append(values[-1] if values else 0.0)
            values = values[:n_paragraphs]
            columns.append(np.array(values, dtype=np.float64))
            names.append(key)

        elif source == "per_chapter":
            data = result.per_chapter
            if data is None or key not in data:
                continue
            scalar = float(data[key])
            columns.append(np.full(n_paragraphs, scalar, dtype=np.float64))
            names.append(key)

    if not columns:
        # Fallback: uniform features so the matrix is still valid
        columns.append(np.ones(n_paragraphs, dtype=np.float64))
        names.append("_constant")

    features = np.column_stack(columns)
    return features, names


def _zscore_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize each column. Constant columns become zero."""
    means = arr.mean(axis=0, keepdims=True)
    stds = arr.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    return (arr - means) / stds


def _cosine_attention(features: np.ndarray) -> np.ndarray:
    """Compute NxN cosine similarity matrix."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = features / norms
    return normed @ normed.T


def _attention_entropy(attn: np.ndarray) -> list[float]:
    """Per-row entropy of softmax(attention) -- measures focus vs diffusion."""
    # Softmax each row (temperature=1)
    # Shift for numerical stability
    shifted = attn - attn.max(axis=1, keepdims=True)
    exp_a = np.exp(shifted)
    softmax = exp_a / exp_a.sum(axis=1, keepdims=True)
    # Shannon entropy per row
    # Clip to avoid log(0)
    softmax = np.clip(softmax, 1e-12, None)
    entropy = -np.sum(softmax * np.log(softmax), axis=1)
    return [round(float(e), 6) for e in entropy]


def _coherence(attn: np.ndarray) -> float:
    """Mean of adjacent-diagonal entries A[i, i+1]."""
    n = attn.shape[0]
    if n < 2:
        return 0.0
    diag_neighbors = [attn[i, i + 1] for i in range(n - 1)]
    return round(float(np.mean(diag_neighbors)), 6)


class NarrativeAttentionLens(Lens):
    """Meta-lens: assembles features from other lenses, computes attention."""

    name = "narrative_attention"
    requires_providers: list[str] = []
    consumes_lenses = [
        "psychic_distance",
        "info_contour",
        "foregrounding",
        "emotion_arc",
        "boyd_narrative_role",
        "uncertainty_reduction",
    ]

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        consumed = consumed or {}
        paragraphs = split_paragraphs(text)
        n = len(paragraphs)

        # 1. Assemble feature matrix
        features, feature_names = _gather_features(consumed, n)

        # 2. Normalize
        features_norm = _zscore_normalize(features)

        # 3. Cosine attention
        attn = _cosine_attention(features_norm)

        # 4. Per-paragraph entropy
        entropy = _attention_entropy(attn)

        # 5. Chapter-level coherence
        coherence = _coherence(attn)

        # Convert attention matrix to list-of-lists for JSON serialization
        attn_list = [[round(float(v), 6) for v in row] for row in attn]

        return LensResult(
            lens_name=self.name,
            per_paragraph={
                "attention_entropy": entropy,
            },
            per_chapter={
                "coherence": coherence,
                "n_features": len(feature_names),
            },
            raw={
                "attention_matrix": attn_list,
                "feature_names": feature_names,
            },
        )
