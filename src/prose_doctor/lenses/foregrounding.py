"""Foregrounding lens — 5-axis literary texture measurement.

Measures alliteration density, syntactic inversion, sentence-length
variation, fragment ratio, and unexpected collocations. Returns a
composite foregrounding index and a prescription for the weakest axis.

Ported from prose_doctor.ml.foregrounding into the Lens interface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


# ---------- Internal measurement helpers ----------


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


def _measure_unexpected_collocations(
    text: str, nlp, st_model
) -> tuple[float, list[tuple[str, str, float]]]:
    """Semantically distant word pairs in close proximity per 1000 words."""
    import numpy as np
    from numpy.linalg import norm

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

    seen: set[tuple[str, str]] = set()
    deduped = []
    for w1, w2, d in unexpected:
        key = tuple(sorted([w1, w2]))
        if key not in seen:
            seen.add(key)
            deduped.append((w1, w2, d))

    total_words = len(text.split())
    density = len(deduped) / max(total_words, 1) * 1000
    return density, sorted(deduped, key=lambda x: -x[2])[:5]


# ---------- Composite scoring helpers ----------


def _composite_index(
    alliteration: float,
    inversion_pct: float,
    sl_cv: float,
    fragment_pct: float,
    unexpected_collocations: float,
) -> float:
    """Composite foregrounding index (0-10 scale)."""
    return (
        min(alliteration / 5, 10) * 0.15
        + min(inversion_pct / 2, 10) * 0.15
        + min(sl_cv / 0.1, 10) * 0.25
        + min(fragment_pct / 3, 10) * 0.15
        + min(unexpected_collocations / 2, 10) * 0.30
    )


def _weakest_axis(
    alliteration: float,
    inversion_pct: float,
    sl_cv: float,
    fragment_pct: float,
    unexpected_collocations: float,
) -> str:
    axes = {
        "alliteration": alliteration / 30,
        "inversion": inversion_pct / 25,
        "rhythm_variety": sl_cv / 0.85,
        "fragments": fragment_pct / 25,
        "unexpected_collocations": unexpected_collocations / 600,
    }
    return min(axes, key=axes.get)  # type: ignore[arg-type]


_PRESCRIPTIONS = {
    "alliteration": "Add 3-4 consonance moments in sensory passages.",
    "inversion": "Restructure 4-5 sentences: verb before subject, or open with prepositional phrase.",
    "rhythm_variety": "Break 3 long sentences into fragments; merge 3 short ones into complex sentences.",
    "fragments": "Add 2-3 standalone staccato sentences at high-tension moments.",
    "unexpected_collocations": "Find 3-4 descriptions and replace one word with something semantically distant.",
}


# ---------- Lens class ----------


class ForegroundingLens(Lens):
    """Analyze foregrounding texture across 5 axes."""

    name = "foregrounding"
    requires_providers: list[str] = ["spacy", "sentence_transformer"]
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        full_text = " ".join(split_paragraphs(text))
        word_count = len(full_text.split())

        nlp = providers.spacy
        st_model = providers.sentence_transformer
        doc = nlp(full_text)

        allit = _measure_alliteration(doc)
        inver = _measure_inversion(doc)
        sl_cv = _measure_sentence_length_cv(doc)
        frag = _measure_fragment_ratio(doc)
        unexp, top_colls = _measure_unexpected_collocations(full_text, nlp, st_model)

        idx = _composite_index(allit, inver, sl_cv, frag, unexp)
        weak = _weakest_axis(allit, inver, sl_cv, frag, unexp)
        prescription = _PRESCRIPTIONS.get(weak, "")

        per_chapter = {
            "index": idx,
            "inversion_pct": inver,
            "sl_cv": sl_cv,
            "fragment_pct": frag,
            "alliteration": allit,
            "unexpected_collocations": unexp,
            "word_count": float(word_count),
        }

        raw = {
            "weakest_axis": weak,
            "prescription": prescription,
            "top_collocations": [
                {"w1": w1, "w2": w2, "distance": d} for w1, w2, d in top_colls
            ],
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            raw=raw,
        )
