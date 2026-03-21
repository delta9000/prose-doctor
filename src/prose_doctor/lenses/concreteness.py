"""Concreteness lens -- semantic vagueness detection via norm lookup + embedding fallback.

Scores prose on the concrete-abstract spectrum using:
1. Brysbaert et al. (2014) concreteness norms for known words (40K, CC-BY)
2. Direction-vector projection in mpnet-base-v2 embedding space for OOV words

Citation: Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness
ratings for 40,000 generally known English word lemmas. Behavior Research Methods,
46, 904-911. DOI: 10.3758/s13428-013-0403-5
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

NORMS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "refs" / "brysbaert_concreteness.csv"
DIRECTION_PATH = Path(__file__).resolve().parent.parent / "data" / "concreteness_direction.npy"

VAGUE_NOUNS = {
    "thing", "things", "something", "everything", "nothing", "anything",
    "way", "ways", "kind", "sort", "type", "stuff", "matter",
    "aspect", "factor", "element", "area", "issue",
}

_norms_cache: dict[str, float] | None = None
_direction_cache: np.ndarray | None = None
_st_cache = None


def _load_norms() -> dict[str, float]:
    global _norms_cache
    if _norms_cache is not None:
        return _norms_cache

    _norms_cache = {}
    if not NORMS_PATH.exists():
        return _norms_cache

    with open(NORMS_PATH) as f:
        reader = csv.DictReader(f)
        conc_col = None
        word_col = None
        for col in reader.fieldnames or []:
            if "conc" in col.lower() and ("m" in col.lower() or "mean" in col.lower()):
                conc_col = col
            if col.lower() == "word":
                word_col = col
        if conc_col is None or word_col is None:
            word_col = reader.fieldnames[0] if reader.fieldnames else "Word"
            conc_col = reader.fieldnames[2] if reader.fieldnames and len(reader.fieldnames) > 2 else "Conc.M"

        for row in reader:
            try:
                word = row[word_col].strip().lower()
                score = float(row[conc_col])
                _norms_cache[word] = score
            except (KeyError, ValueError):
                continue

    return _norms_cache


def _load_direction() -> np.ndarray | None:
    global _direction_cache
    if _direction_cache is not None:
        return _direction_cache
    if not DIRECTION_PATH.exists():
        return None
    _direction_cache = np.load(DIRECTION_PATH)
    return _direction_cache


def _get_st():
    """Lazy-load sentence-transformer for OOV fallback.

    Uses the same all-mpnet-base-v2 (768d) model as sensory.py's probe.
    """
    global _st_cache
    if _st_cache is None:
        from sentence_transformers import SentenceTransformer
        _st_cache = SentenceTransformer("all-mpnet-base-v2")
    return _st_cache


def _score_oov_batch(words: list[str], direction: np.ndarray) -> list[float]:
    st = _get_st()
    embs = st.encode(words, show_progress_bar=False, batch_size=128)
    projections = embs @ direction
    scores = 3.0 + projections * 6.67
    return list(np.clip(scores, 1.0, 5.0))


class ConcretenessLens(Lens):
    """Measure semantic concreteness vs abstraction in prose."""

    name = "concreteness"
    requires_providers = ["spacy"]
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict | None = None,
    ) -> LensResult:
        nlp = providers.spacy
        norms = _load_norms()
        direction = _load_direction()

        paragraphs = split_paragraphs(text)
        doc = nlp(text)

        sentence_scores: list[float] = []
        sentence_vague_counts: list[int] = []
        oov_words: list[str] = []
        oov_indices: list[tuple[int, int]] = []

        sentences = list(doc.sents)
        for sent_idx, sent in enumerate(sentences):
            scores = []
            vague_count = 0
            for token in sent:
                if not token.is_alpha or token.is_stop or len(token.text) <= 2:
                    continue
                word = token.lemma_.lower()
                if word in VAGUE_NOUNS:
                    vague_count += 1
                if word in norms:
                    scores.append(norms[word])
                elif direction is not None:
                    oov_words.append(word)
                    oov_indices.append((sent_idx, len(scores)))
                    scores.append(0.0)  # placeholder

            sentence_scores.append(float(np.mean(scores)) if scores else 3.0)
            sentence_vague_counts.append(vague_count)

        # Batch-score OOV words
        if oov_words and direction is not None:
            oov_scores = _score_oov_batch(oov_words, direction)
            oov_mean = float(np.mean(oov_scores))
        else:
            oov_mean = 3.0

        # Per-paragraph scores
        para_scores: list[float] = []
        para_boundaries = []
        offset = 0
        for para in paragraphs:
            start = text.find(para, offset)
            if start == -1:
                start = offset
            para_boundaries.append((start, start + len(para)))
            offset = start + len(para)

        for p_start, p_end in para_boundaries:
            para_sent_indices = [
                i for i, s in enumerate(sentences)
                if s.start_char >= p_start and s.start_char < p_end
            ]
            if para_sent_indices:
                para_scores.append(float(np.mean([sentence_scores[i] for i in para_sent_indices])))
            else:
                para_scores.append(3.0)

        # Chapter-level
        all_scores = [s for s in sentence_scores if s != 3.0] or [3.0]
        concreteness_mean = float(np.mean(all_scores))
        total_words = sum(1 for t in doc if t.is_alpha and not t.is_stop and len(t.text) > 2)
        total_vague = sum(sentence_vague_counts)
        abstractness_ratio = float(sum(1 for s in all_scores if s < 2.5) / max(len(all_scores), 1))

        return LensResult(
            lens_name="concreteness",
            per_sentence={"concreteness": sentence_scores},
            per_paragraph={"concreteness": para_scores},
            per_chapter={
                "concreteness_mean": round(concreteness_mean, 3),
                "abstractness_ratio": round(abstractness_ratio, 3),
                "vague_noun_density": round(total_vague / max(total_words, 1) * 100, 2),
                "oov_concreteness_mean": round(oov_mean, 3),
                "norms_coverage": round(
                    (total_words - len(oov_words)) / max(total_words, 1), 3
                ),
            },
            raw={
                "vague_nouns_found": total_vague,
                "oov_count": len(oov_words),
                "total_scored_words": total_words,
            },
        )
