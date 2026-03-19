"""Foregrounding index: 5-axis literary texture measurement."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from prose_doctor.text import split_paragraphs


def _measure_alliteration(doc) -> float:
    """Alliterative pairs per 1000 words."""
    content = [
        t for t in doc
        if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and len(t.text) > 2
    ]
    hits = 0
    for i in range(len(content) - 1):
        a = content[i].text.lower()
        b = content[i + 1].text.lower()
        if a[0] == b[0] and a[0].isalpha() and a[0] not in "aeiou":
            hits += 1
    return hits / max(len(doc), 1) * 1000


def _measure_inversion(doc) -> float:
    """Percentage of sentences with subject after verb."""
    inversions = 0
    total = 0
    for sent in doc.sents:
        total += 1
        root = None
        for t in sent:
            if t.dep_ == "ROOT":
                root = t
            if t.dep_ in ("nsubj", "nsubjpass") and root is not None and t.i > root.i:
                inversions += 1
                break
    return inversions / max(total, 1) * 100


def _measure_sentence_length_cv(doc) -> float:
    """Coefficient of variation of sentence lengths."""
    import numpy as np

    lengths = [len(s) for s in doc.sents if len(s) > 2]
    if len(lengths) < 3:
        return 0.0
    return float(np.std(lengths) / np.mean(lengths))


def _measure_fragment_ratio(doc) -> float:
    """Percentage of sentences that are fragments (< 5 tokens)."""
    sents = list(doc.sents)
    if not sents:
        return 0.0
    return sum(1 for s in sents if len(s) < 5) / len(sents) * 100


def _measure_unexpected_collocations(text: str, st_model) -> tuple[float, list[tuple[str, str, float]]]:
    """Semantically distant word pairs in close proximity per 1000 words."""
    import numpy as np
    from numpy.linalg import norm

    # Need spacy for POS tagging
    from prose_doctor.ml.models import ModelManager

    mm = ModelManager()
    nlp = mm.spacy
    doc = nlp(text[:5000])

    content = [
        (t.text.lower(), t.i)
        for t in doc
        if t.pos_ in ("NOUN", "VERB", "ADJ") and len(t.text) > 3 and not t.is_stop
    ]
    if len(content) < 10:
        return 0.0, []

    pairs = []
    for i in range(len(content)):
        for j in range(i + 1, min(i + 4, len(content))):
            w1, p1 = content[i]
            w2, p2 = content[j]
            if w1 != w2 and abs(p1 - p2) <= 8:
                pairs.append((w1, w2))

    if not pairs:
        return 0.0, []

    all_words = list(set(w for pair in pairs for w in pair))
    embs = st_model.encode(all_words, show_progress_bar=False)
    word_emb = {w: embs[i] for i, w in enumerate(all_words)}

    unexpected = []
    for w1, w2 in pairs:
        e1, e2 = word_emb[w1], word_emb[w2]
        sim = np.dot(e1, e2) / (norm(e1) * norm(e2))
        distance = 1 - sim
        if distance > 0.75:
            unexpected.append((w1, w2, float(distance)))

    seen = set()
    deduped = []
    for w1, w2, d in unexpected:
        key = tuple(sorted([w1, w2]))
        if key not in seen:
            seen.add(key)
            deduped.append((w1, w2, d))

    total_words = len(text.split())
    density = len(deduped) / max(total_words, 1) * 1000
    return density, sorted(deduped, key=lambda x: -x[2])[:5]


@dataclass
class ForegroundingScore:
    """Foregrounding score for a chapter."""

    filename: str
    word_count: int
    alliteration: float
    inversion_pct: float
    sl_cv: float
    fragment_pct: float
    unexpected_collocations: float = 0.0
    top_collocations: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def index(self) -> float:
        """Composite foregrounding index (0-10 scale)."""
        return (
            min(self.alliteration / 5, 10) * 0.15
            + min(self.inversion_pct / 2, 10) * 0.15
            + min(self.sl_cv / 0.1, 10) * 0.25
            + min(self.fragment_pct / 3, 10) * 0.15
            + min(self.unexpected_collocations / 2, 10) * 0.30
        )

    @property
    def weakest_axis(self) -> str:
        axes = {
            "alliteration": self.alliteration / 30,
            "inversion": self.inversion_pct / 25,
            "rhythm_variety": self.sl_cv / 0.85,
            "fragments": self.fragment_pct / 25,
            "unexpected_collocations": self.unexpected_collocations / 600,
        }
        return min(axes, key=axes.get)

    @property
    def prescription(self) -> str:
        prescriptions = {
            "alliteration": "Add 3-4 consonance moments in sensory passages.",
            "inversion": "Restructure 4-5 sentences: verb before subject, or open with prepositional phrase.",
            "rhythm_variety": "Break 3 long sentences into fragments; merge 3 short ones into complex sentences.",
            "fragments": "Add 2-3 standalone staccato sentences at high-tension moments.",
            "unexpected_collocations": "Find 3-4 descriptions and replace one word with something semantically distant.",
        }
        return prescriptions.get(self.weakest_axis, "")


def score_chapter(text: str, filename: str, model_manager) -> ForegroundingScore:
    """Score a chapter on all 5 foregrounding axes."""
    full_text = " ".join(split_paragraphs(text))
    word_count = len(full_text.split())

    nlp = model_manager.spacy
    st_model = model_manager.sentence_transformer
    doc = nlp(full_text)

    allit = _measure_alliteration(doc)
    inver = _measure_inversion(doc)
    sl_cv = _measure_sentence_length_cv(doc)
    frag = _measure_fragment_ratio(doc)
    unexp, top_colls = _measure_unexpected_collocations(full_text, st_model)

    return ForegroundingScore(
        filename=filename,
        word_count=word_count,
        alliteration=allit,
        inversion_pct=inver,
        sl_cv=sl_cv,
        fragment_pct=frag,
        unexpected_collocations=unexp,
        top_collocations=top_colls,
    )
