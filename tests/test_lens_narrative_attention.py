"""Tests for the narrative_attention meta-lens."""
import numpy as np
import pytest

from prose_doctor.lenses import LensResult
from prose_doctor.lenses.narrative_attention import NarrativeAttentionLens
from prose_doctor.providers import ProviderPool


def _make_consumed(n_paragraphs=8):
    """Build fake consumed lens results for testing."""
    return {
        "psychic_distance": LensResult(
            lens_name="psychic_distance",
            per_paragraph={"pd_mean": list(np.random.uniform(0.2, 0.5, n_paragraphs))},
        ),
        "info_contour": LensResult(
            lens_name="info_contour",
            per_sentence={"surprisal": list(np.random.uniform(2, 6, n_paragraphs * 3))},
            per_paragraph={"mean_surprisal": list(np.random.uniform(3, 5, n_paragraphs))},
        ),
        "foregrounding": LensResult(
            lens_name="foregrounding",
            per_chapter={"sl_cv": 0.7, "inversion_pct": 40.0, "fragment_pct": 5.0},
        ),
        "emotion_arc": LensResult(
            lens_name="emotion_arc",
            per_chapter={"mean_sentiment": 0.6, "dynamic_range": 0.3},
        ),
        "boyd_narrative_role": LensResult(
            lens_name="boyd_narrative_role",
            per_paragraph={
                "staging": list(np.random.uniform(0, 0.1, n_paragraphs)),
                "progression": list(np.random.uniform(0, 0.1, n_paragraphs)),
                "tension": list(np.random.uniform(0, 0.1, n_paragraphs)),
            },
        ),
        "uncertainty_reduction": LensResult(
            lens_name="uncertainty_reduction",
            per_paragraph={
                "uncertainty_reduction": list(np.random.uniform(-0.5, 0.5, n_paragraphs)),
            },
        ),
    }


def test_narrative_attention_returns_result():
    lens = NarrativeAttentionLens()
    pool = ProviderPool()
    consumed = _make_consumed(8)
    text = "\n\n".join([f"Paragraph {i} with some text." for i in range(8)])
    result = lens.analyze(text, "test.md", pool, consumed=consumed)
    assert isinstance(result, LensResult)
    assert result.lens_name == "narrative_attention"
    assert result.per_chapter is not None


def test_narrative_attention_has_attention_matrix():
    lens = NarrativeAttentionLens()
    pool = ProviderPool()
    consumed = _make_consumed(8)
    text = "\n\n".join([f"Paragraph {i} with some text." for i in range(8)])
    result = lens.analyze(text, "test.md", pool, consumed=consumed)
    assert "attention_matrix" in result.raw
    matrix = result.raw["attention_matrix"]
    assert len(matrix) == 8  # NxN matrix
    assert len(matrix[0]) == 8


def test_narrative_attention_per_paragraph():
    lens = NarrativeAttentionLens()
    pool = ProviderPool()
    consumed = _make_consumed(8)
    text = "\n\n".join([f"Paragraph {i} with some text." for i in range(8)])
    result = lens.analyze(text, "test.md", pool, consumed=consumed)
    assert result.per_paragraph is not None
    assert "attention_entropy" in result.per_paragraph
    assert len(result.per_paragraph["attention_entropy"]) == 8


def test_narrative_attention_metadata():
    lens = NarrativeAttentionLens()
    assert lens.name == "narrative_attention"
    assert lens.requires_providers == []
    assert "psychic_distance" in lens.consumes_lenses
    assert "info_contour" in lens.consumes_lenses


def test_narrative_attention_coherence():
    """Chapter-level coherence should be a float between -1 and 1."""
    lens = NarrativeAttentionLens()
    pool = ProviderPool()
    consumed = _make_consumed(8)
    text = "\n\n".join([f"Paragraph {i} with some text." for i in range(8)])
    result = lens.analyze(text, "test.md", pool, consumed=consumed)
    assert "coherence" in result.per_chapter
    assert -1.0 <= result.per_chapter["coherence"] <= 1.0


def test_narrative_attention_diagonal_is_one():
    """Self-attention (diagonal) should be 1.0 (cosine of vector with itself)."""
    lens = NarrativeAttentionLens()
    pool = ProviderPool()
    consumed = _make_consumed(8)
    text = "\n\n".join([f"Paragraph {i} with some text." for i in range(8)])
    result = lens.analyze(text, "test.md", pool, consumed=consumed)
    matrix = result.raw["attention_matrix"]
    for i in range(8):
        assert abs(matrix[i][i] - 1.0) < 1e-6


def test_narrative_attention_broadcasts_chapter_features():
    """Foregrounding only has per_chapter data -- should be broadcast to all paragraphs."""
    lens = NarrativeAttentionLens()
    pool = ProviderPool()
    consumed = _make_consumed(5)
    text = "\n\n".join([f"Paragraph {i} with some text." for i in range(5)])
    result = lens.analyze(text, "test.md", pool, consumed=consumed)
    assert len(result.raw["attention_matrix"]) == 5
    assert "feature_names" in result.raw


def test_narrative_attention_missing_consumed_lens():
    """Should handle missing consumed lenses gracefully."""
    lens = NarrativeAttentionLens()
    pool = ProviderPool()
    # Only provide some consumed lenses
    consumed = {
        "psychic_distance": LensResult(
            lens_name="psychic_distance",
            per_paragraph={"pd_mean": [0.3, 0.4, 0.5]},
        ),
        "boyd_narrative_role": LensResult(
            lens_name="boyd_narrative_role",
            per_paragraph={
                "staging": [0.05, 0.06, 0.07],
                "progression": [0.03, 0.04, 0.05],
                "tension": [0.02, 0.03, 0.04],
            },
        ),
    }
    text = "\n\n".join([f"Paragraph {i} with some text." for i in range(3)])
    result = lens.analyze(text, "test.md", pool, consumed=consumed)
    assert isinstance(result, LensResult)
    assert len(result.raw["attention_matrix"]) == 3
