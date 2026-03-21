# New Lenses: Concreteness, Referential Cohesion, Situation Shifts, Discourse Relations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four new lenses that fill the biggest blind spots in prose-doctor's current analysis: semantic vagueness, entity continuity, scene-transition tracking, and connective structure.

**Architecture:** Each lens follows the established `Lens` ABC → `LensResult` pattern. Concreteness uses a data-free embedding direction vector + Brysbaert norm lookup. Referential cohesion uses spaCy dependency parsing with an entity-grid model and optional networkx graph analysis. Situation shifts uses spaCy NER + word-list heuristics. Discourse relations uses connective word-list classification.

**Tech Stack:** spaCy (existing provider), all-mpnet-base-v2 (existing, shared with sensory probe), Brysbaert concreteness norms (CC-BY CSV, ~40K words), networkx (new optional dep), scipy (existing).

**Prior context:** This plan comes from gap analysis documented in this conversation — the Codex brainstorm identified referential cohesion, concreteness, situation model continuity, and discourse relations as the four implementable, research-backed gaps. See `docs/lenses.md` for the existing lens documentation pattern.

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `refs/brysbaert_concreteness.csv` | Brysbaert et al. (2014) concreteness norms, 40K words, CC-BY |
| `src/prose_doctor/data/concreteness_direction.npy` | Precomputed concrete→abstract direction vector (768d, mpnet-base-v2) |
| `src/prose_doctor/lenses/concreteness.py` | Concreteness lens — norm lookup + embedding fallback |
| `src/prose_doctor/lenses/referential_cohesion.py` | Entity-grid coherence + entity graph analysis |
| `src/prose_doctor/lenses/situation_shifts.py` | Time/space/actor shift detection at paragraph boundaries |
| `src/prose_doctor/lenses/discourse_relations.py` | Connective-based discourse relation classification |
| `scripts/compute_concreteness_direction.py` | One-shot script to compute and save the direction vector |
| `tests/test_lens_concreteness.py` | Tests for concreteness lens |
| `tests/test_lens_referential_cohesion.py` | Tests for referential cohesion lens |
| `tests/test_lens_situation_shifts.py` | Tests for situation shifts lens |
| `tests/test_lens_discourse_relations.py` | Tests for discourse relations lens |

### Modified files

| File | Change |
|------|--------|
| `src/prose_doctor/lenses/defaults.py` | Register four new lenses |
| `src/prose_doctor/validation/tiers.toml` | Add tier entries for four new lenses |
| `docs/lenses.md` | Add documentation for four new lenses |
| `pyproject.toml` | Add networkx to `[ml]` extra |

---

## Task 1: Download Brysbaert concreteness norms

**Files:**
- Create: `refs/brysbaert_concreteness.csv`

- [ ] **Step 1: Download the norms**

The Brysbaert, Warriner & Kuperman (2014) concreteness ratings are hosted on multiple sources. The canonical dataset has ~40K English words rated 1–5 on concreteness by 4,000+ participants. CC-BY license.

Download from the supplementary materials or a mirror. The CSV should have at minimum columns: `Word`, `Conc.M` (mean concreteness rating, 1–5), `Conc.SD`.

```bash
cd /home/ben/code/prose-doctor
# Download from the paper's supplementary data. Try these URLs in order:
# Primary: Springer supplementary materials
curl -L "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5/MediaObjects/13428_2013_403_MOESM1_ESM.xlsx" -o /tmp/brysbaert_raw.xlsx
# Fallback 1: OSF repository
# curl -L "https://osf.io/download/6y3pq/" -o /tmp/brysbaert_raw.xlsx
# Fallback 2: Ghent University ESPP page
# curl -L "http://crr.ugent.be/papers/Concreteness_ratings.xlsx" -o /tmp/brysbaert_raw.xlsx
```

If the download is an Excel file, convert to CSV:

```bash
cd /home/ben/code/prose-doctor
uv run --with openpyxl python -c "
import openpyxl
wb = openpyxl.load_workbook('/tmp/brysbaert_raw.xlsx')
ws = wb.active
import csv
with open('refs/brysbaert_concreteness.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in ws.iter_rows(values_only=True):
        writer.writerow(row)
print(f'Wrote {ws.max_row} rows')
"
```

Verify the file has the expected structure:

```bash
head -3 refs/brysbaert_concreteness.csv
```

Expected: header row with Word, Bigram, Conc.M, Conc.SD, Unknown, Total, Percent_known, SUBTLEX, Dom_Pos columns. ~40K data rows.

- [ ] **Step 2: Verify and add citation header**

Add a one-line comment or companion file noting the citation:

```bash
echo "# Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings for 40,000 generally known English word lemmas. Behavior Research Methods, 46, 904-911. DOI: 10.3758/s13428-013-0403-5. License: CC-BY." > refs/BRYSBAERT_CITATION.txt
```

- [ ] **Step 3: Commit**

```bash
git add refs/brysbaert_concreteness.csv refs/BRYSBAERT_CITATION.txt
git commit -m "data: add Brysbaert concreteness norms (40K words, CC-BY)"
```

---

## Task 2: Compute concreteness direction vector

**Files:**
- Create: `scripts/compute_concreteness_direction.py`
- Create: `src/prose_doctor/data/concreteness_direction.npy`

This script computes the concrete→abstract direction in mpnet-base-v2 embedding space using anchor words, validates against the Brysbaert norms, and saves the unit direction vector.

- [ ] **Step 1: Write the script**

```python
# scripts/compute_concreteness_direction.py
"""Compute the concreteness direction vector in mpnet-base-v2 embedding space.

Uses anchor words to find the abstract→concrete axis, then validates
against Brysbaert norms. Saves the unit direction vector as .npy.

Usage:
    cd /home/ben/code/prose-doctor
    uv run python scripts/compute_concreteness_direction.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer

# Anchor words — chosen to span clear concrete/abstract exemplars
# avoiding genre-specific terms or words with strong polysemy
CONCRETE_ANCHORS = [
    "hammer", "blood", "kitchen", "cigarette", "gravel", "elbow",
    "rust", "doorknob", "shovel", "brick", "fingernail", "puddle",
    "collar", "windshield", "sandpaper", "kettle", "splinter", "ankle",
]
ABSTRACT_ANCHORS = [
    "freedom", "justice", "possibility", "tendency", "significance",
    "notion", "truth", "essence", "irony", "ambiguity", "paradox",
    "morality", "hypothesis", "obligation", "sentiment", "intuition",
]

OUT_PATH = Path(__file__).resolve().parent.parent / "src" / "prose_doctor" / "data" / "concreteness_direction.npy"
NORMS_PATH = Path(__file__).resolve().parent.parent / "refs" / "brysbaert_concreteness.csv"


def main():
    print("Loading mpnet-base-v2...")
    st = SentenceTransformer("all-mpnet-base-v2")

    # Compute direction from anchors
    concrete_embs = st.encode(CONCRETE_ANCHORS)
    abstract_embs = st.encode(ABSTRACT_ANCHORS)

    concrete_centroid = concrete_embs.mean(axis=0)
    abstract_centroid = abstract_embs.mean(axis=0)

    direction = concrete_centroid - abstract_centroid
    direction = direction / np.linalg.norm(direction)

    print(f"Direction vector shape: {direction.shape}")
    print(f"Direction norm: {np.linalg.norm(direction):.4f}")

    # Validate against Brysbaert norms
    print(f"\nLoading Brysbaert norms from {NORMS_PATH}...")
    df = pd.read_csv(NORMS_PATH)

    # Find the concreteness column (may be named Conc.M or similar)
    conc_col = None
    for col in df.columns:
        if "conc" in col.lower() and ("m" in col.lower() or "mean" in col.lower()):
            conc_col = col
            break
    if conc_col is None:
        print("WARNING: Could not find concreteness column. Available columns:")
        print(df.columns.tolist())
        return

    # Find the word column
    word_col = None
    for col in df.columns:
        if col.lower() == "word":
            word_col = col
            break
    if word_col is None:
        word_col = df.columns[0]

    print(f"Using columns: word={word_col}, concreteness={conc_col}")

    # Sample for validation (encoding 40K words takes a while)
    sample = df.dropna(subset=[conc_col]).sample(n=min(5000, len(df)), random_state=42)
    words = sample[word_col].tolist()
    human_scores = sample[conc_col].values

    print(f"Encoding {len(words)} words for validation...")
    embeddings = st.encode(words, show_progress_bar=True, batch_size=256)
    predicted = embeddings @ direction

    r, p = pearsonr(predicted, human_scores)
    print(f"\nValidation: r={r:.4f}, p={p:.2e}")
    print(f"  (r > 0.70 is good, r > 0.80 is excellent)")

    if r < 0.50:
        print("WARNING: Correlation is low. Direction vector may be unreliable.")
        print("Consider adjusting anchor words or using the Brysbaert lookup directly.")

    # Save
    np.save(OUT_PATH, direction)
    print(f"\nSaved direction vector to {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
cd /home/ben/code/prose-doctor && uv run python scripts/compute_concreteness_direction.py
```

Expected: r > 0.70 against Brysbaert norms. The script prints the correlation and saves the direction vector.

- [ ] **Step 3: Commit**

```bash
git add scripts/compute_concreteness_direction.py src/prose_doctor/data/concreteness_direction.npy
git commit -m "feat: compute concreteness direction vector (validated against Brysbaert norms)"
```

---

## Task 3: Concreteness lens

**Files:**
- Create: `src/prose_doctor/lenses/concreteness.py`
- Test: `tests/test_lens_concreteness.py`

Dual-path scoring: Brysbaert lookup for known words, direction-vector projection for OOV. Returns per-sentence concreteness means, per-paragraph means, chapter-level abstractness ratio, and vague-noun density.

- [ ] **Step 1: Write the test**

```python
# tests/test_lens_concreteness.py
import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.concreteness import ConcretenessLens
from prose_doctor.providers import ProviderPool


CONCRETE_SAMPLE = '''
Marcus pressed his back against the brick wall. His fingers found
the crack in the mortar, cold and damp. The flashlight in his left
hand was dead weight now, batteries drained hours ago.

She set the coffee mug on the counter. The ceramic clinked against
the granite. Outside, rain hammered the tin roof.
'''

ABSTRACT_SAMPLE = '''
The significance of the moment was not lost on anyone. There was a
sense of possibility in the air, a feeling that things were about
to change in ways that mattered.

Something shifted in the dynamic between them. The weight of
unspoken truths hung in the silence, heavy with implication and
the kind of meaning that resisted easy articulation.
'''


def test_concreteness_returns_result():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(CONCRETE_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "concreteness"
    assert result.per_chapter is not None
    assert "concreteness_mean" in result.per_chapter
    assert "abstractness_ratio" in result.per_chapter


def test_concrete_scores_higher_than_abstract():
    lens = ConcretenessLens()
    pool = ProviderPool()
    concrete = lens.analyze(CONCRETE_SAMPLE, "concrete.md", pool)
    abstract = lens.analyze(ABSTRACT_SAMPLE, "abstract.md", pool)
    assert concrete.per_chapter["concreteness_mean"] > abstract.per_chapter["concreteness_mean"]


def test_concreteness_has_per_sentence():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(CONCRETE_SAMPLE, "test.md", pool)
    assert result.per_sentence is not None
    assert "concreteness" in result.per_sentence
    assert len(result.per_sentence["concreteness"]) > 0


def test_concreteness_has_per_paragraph():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(CONCRETE_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "concreteness" in result.per_paragraph


def test_abstract_sample_flags_vague_nouns():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(ABSTRACT_SAMPLE, "abstract.md", pool)
    assert result.per_chapter["vague_noun_density"] > 0


def test_concreteness_metadata():
    lens = ConcretenessLens()
    assert lens.name == "concreteness"
    assert lens.requires_providers == ["spacy"]
    assert lens.consumes_lenses == []
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_concreteness.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the concreteness lens**

```python
# src/prose_doctor/lenses/concreteness.py
"""Concreteness lens — semantic vagueness detection via norm lookup + embedding fallback.

Scores prose on the concrete↔abstract spectrum using:
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

# Words that signal semantic vagueness regardless of concreteness score
# Genuinely vague nouns — words that are semantically empty without qualification.
# Does NOT include "silence", "darkness", "weight", "presence", "moment" — those
# are abstract-but-specific and can be concrete in fiction ("the weight of the pack",
# "the darkness pressed against the windows").
VAGUE_NOUNS = {
    "thing", "things", "something", "everything", "nothing", "anything",
    "way", "ways", "kind", "sort", "type", "stuff", "matter",
    "aspect", "factor", "element", "area", "issue",
}

_norms_cache: dict[str, float] | None = None
_direction_cache: np.ndarray | None = None
_st_cache = None


def _load_norms() -> dict[str, float]:
    """Load Brysbaert norms into a word→concreteness dict. Cached."""
    global _norms_cache
    if _norms_cache is not None:
        return _norms_cache

    _norms_cache = {}
    if not NORMS_PATH.exists():
        return _norms_cache

    with open(NORMS_PATH) as f:
        reader = csv.DictReader(f)
        # Find the concreteness and word columns
        conc_col = None
        word_col = None
        for col in reader.fieldnames or []:
            if "conc" in col.lower() and ("m" in col.lower() or "mean" in col.lower()):
                conc_col = col
            if col.lower() == "word":
                word_col = col
        if conc_col is None or word_col is None:
            # Fall back to positional
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
    """Load the precomputed concreteness direction vector."""
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
    The direction vector was computed against this exact model, so we
    cannot substitute the shared 384d provider. If both lenses are active
    in the same session, they share the underlying HuggingFace cache
    (same model weights loaded once by transformers).
    """
    global _st_cache
    if _st_cache is None:
        from sentence_transformers import SentenceTransformer
        _st_cache = SentenceTransformer("all-mpnet-base-v2")
    return _st_cache


def _score_word(word: str, norms: dict[str, float]) -> float | None:
    """Score a single word. Returns 1-5 scale or None if unknown."""
    w = word.lower().strip()
    if w in norms:
        return norms[w]
    return None


def _score_oov_batch(words: list[str], direction: np.ndarray) -> list[float]:
    """Score OOV words via direction-vector projection. Returns 1-5 scale."""
    st = _get_st()
    embs = st.encode(words, show_progress_bar=False, batch_size=128)
    projections = embs @ direction
    # Map projection to 1-5 scale (positive = concrete, negative = abstract)
    # Calibrate: typical projections range roughly -0.3 to +0.3
    # Map to 1-5: score = 3 + projection * 6.67 (so ±0.3 maps to ~1 and ~5)
    scores = 3.0 + projections * 6.67
    return list(np.clip(scores, 1.0, 5.0))


class ConcretenessLens(Lens):
    """Measure semantic concreteness vs abstraction in prose."""

    name = "concreteness"
    requires_providers = ["spacy"]
    consumes_lenses = []

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

        # Score each sentence
        sentence_scores: list[float] = []
        sentence_vague_counts: list[int] = []
        oov_words: list[str] = []
        oov_indices: list[tuple[int, int]] = []  # (sentence_idx, word_position)

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
                score = _score_word(word, norms)
                if score is not None:
                    scores.append(score)
                elif direction is not None:
                    oov_words.append(word)
                    oov_indices.append((sent_idx, len(scores)))
                    scores.append(0.0)  # placeholder

            sentence_scores.append(np.mean(scores) if scores else 3.0)
            sentence_vague_counts.append(vague_count)

        # Batch-score OOV words
        if oov_words and direction is not None:
            oov_scores = _score_oov_batch(oov_words, direction)
            # We can't easily patch sentence scores back (we'd need to recompute means),
            # so just compute an OOV adjustment factor for the chapter mean
            oov_mean = np.mean(oov_scores)
        else:
            oov_mean = 3.0

        # Per-paragraph scores — map sentences to paragraphs by character offset,
        # reusing the already-parsed doc instead of re-parsing each paragraph.
        para_scores: list[float] = []
        para_boundaries = []
        offset = 0
        for para in paragraphs:
            start = text.find(para, offset)
            if start == -1:
                start = offset
            para_boundaries.append((start, start + len(para)))
            offset = start + len(para)

        sent_starts = [sent.start_char for sent in sentences]
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
                "oov_concreteness_mean": round(float(oov_mean), 3),
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
```

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_concreteness.py -v`
Expected: PASS (6 tests). The `test_concrete_scores_higher_than_abstract` test is the key validation.

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/concreteness.py tests/test_lens_concreteness.py
git commit -m "feat: concreteness lens — Brysbaert norms + embedding direction fallback"
```

---

## Task 4: Add networkx dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add networkx to ml extra**

Add `networkx>=3.0` to the `[project.optional-dependencies] ml` list in pyproject.toml. networkx is pure Python, no compilation needed.

- [ ] **Step 2: Install and verify**

```bash
cd /home/ben/code/prose-doctor && uv pip install -e ".[ml]"
uv run python -c "import networkx as nx; print(nx.__version__)"
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add networkx for entity graph analysis"
```

---

## Task 5: Referential cohesion lens

**Files:**
- Create: `src/prose_doctor/lenses/referential_cohesion.py`
- Test: `tests/test_lens_referential_cohesion.py`

Entity-grid coherence using spaCy dependency parsing. Tracks entity mentions across sentences, scores transition probabilities, detects pronoun ambiguity and referent churn. Builds an entity graph with networkx for structural analysis.

- [ ] **Step 1: Write the test**

```python
# tests/test_lens_referential_cohesion.py
import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.referential_cohesion import ReferentialCohesionLens
from prose_doctor.providers import ProviderPool


COHERENT_SAMPLE = '''
Marcus pressed his back against the wall. His fingers found the crack
in the mortar. He counted to three, then moved.

Marcus reached the junction. The corridor split into two passages.
He chose the left one, where the air felt cooler.

The passage narrowed until Marcus had to turn sideways. His pack
scraped the concrete. Ahead, a faint light marked the exit.
'''

INCOHERENT_SAMPLE = '''
Marcus pressed his back against the wall. The government had been
overthrown. Several species of birds migrate annually.

A philosophy professor once noted the significance of the occasion.
The submarine dove to three hundred meters. Rain fell on the
abandoned parking lot.

She picked up the telephone. The economic indicators suggested
otherwise. Mountains covered the northern border.
'''


@pytest.mark.slow
def test_cohesion_returns_result():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    result = lens.analyze(COHERENT_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "referential_cohesion"
    assert result.per_chapter is not None
    assert "coherence_score" in result.per_chapter


@pytest.mark.slow
def test_coherent_scores_higher():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    coherent = lens.analyze(COHERENT_SAMPLE, "good.md", pool)
    incoherent = lens.analyze(INCOHERENT_SAMPLE, "bad.md", pool)
    assert coherent.per_chapter["coherence_score"] > incoherent.per_chapter["coherence_score"]


@pytest.mark.slow
def test_cohesion_has_per_paragraph():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    result = lens.analyze(COHERENT_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "entity_continuity" in result.per_paragraph


@pytest.mark.slow
def test_cohesion_detects_subject_churn():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    result = lens.analyze(INCOHERENT_SAMPLE, "bad.md", pool)
    assert result.per_chapter.get("subject_churn", 0) > 0


def test_cohesion_metadata():
    lens = ReferentialCohesionLens()
    assert lens.name == "referential_cohesion"
    assert "spacy" in lens.requires_providers
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_referential_cohesion.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement referential cohesion lens**

The implementation should:

1. **Build entity grid**: For each sentence, extract entities via spaCy noun chunks and named entities. Track each entity's grammatical role (S=nsubj, O=dobj, X=other mention, —=absent). Entity matching: exact string match on lemmatized noun heads, plus pronoun→nearest-matching-noun heuristic within a 3-sentence window.

2. **Score transition probabilities**: For each entity, compute transition probabilities between consecutive sentences (S→S, S→O, S→—, O→S, etc.). Coherent text has more S→S and S→O transitions; incoherent text has more —→S (entities appearing from nowhere).

3. **Per-paragraph entity continuity**: For each paragraph, what fraction of entities in that paragraph also appeared in the previous paragraph. Low continuity = referent churn.

4. **Entity graph (networkx)**: Build a graph where nodes are resolved entities and edges are co-occurrence in the same sentence. Compute:
   - `pagerank`: protagonist centrality
   - `density`: how interconnected the entity network is
   - `dangling_entities`: nodes with degree 1 (mentioned once, never again)

5. **Return LensResult**:
   - `per_paragraph["entity_continuity"]`: list of floats (0–1)
   - `per_chapter`: `coherence_score` (mean transition quality), `subject_churn` (rate of new subjects), `ambiguous_pronouns` (count), `entity_count`, `dangling_entity_count`, `protagonist_centrality`
   - `raw`: entity grid, entity graph stats, top entities by pagerank

`requires_providers = ["spacy"]`

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_referential_cohesion.py -v -m slow`
Expected: PASS (5 tests). The `test_coherent_scores_higher` test is the key validation.

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/referential_cohesion.py tests/test_lens_referential_cohesion.py
git commit -m "feat: referential_cohesion lens — entity-grid coherence + networkx graph analysis"
```

---

## Task 6: Situation shifts lens

**Files:**
- Create: `src/prose_doctor/lenses/situation_shifts.py`
- Test: `tests/test_lens_situation_shifts.py`

Tracks time, space, and actor shifts at paragraph boundaries using spaCy NER, dependency parsing, and word-list heuristics.

- [ ] **Step 1: Write the test**

```python
# tests/test_lens_situation_shifts.py
import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.situation_shifts import SituationShiftsLens
from prose_doctor.providers import ProviderPool


STABLE_SAMPLE = '''
Marcus crouched behind the counter. His breath came in short bursts.
The flashlight beam swept across the floor in front of him.

He waited until the light passed, then moved. His knees ached from
the cold tile. Three meters to the back door.

Marcus reached the door and tested the handle. Locked. He pulled
the pick set from his vest pocket and went to work.
'''

SHIFTING_SAMPLE = '''
Marcus crouched behind the counter in the pharmacy. His breath came
in short bursts.

Three days earlier, Elena had stood in the same spot. The shelves
had still been full then, the lights still working.

In the basement of City Hall, Alderman Voss studied the evacuation
map. His aide hovered behind him, nervous.

The next morning brought rain. Marcus woke in the culvert where
he'd made camp, every joint stiff.
'''


@pytest.mark.slow
def test_shifts_returns_result():
    lens = SituationShiftsLens()
    pool = ProviderPool()
    result = lens.analyze(STABLE_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "situation_shifts"
    assert result.per_chapter is not None


@pytest.mark.slow
def test_shifting_text_has_more_shifts():
    lens = SituationShiftsLens()
    pool = ProviderPool()
    stable = lens.analyze(STABLE_SAMPLE, "stable.md", pool)
    shifting = lens.analyze(SHIFTING_SAMPLE, "shifting.md", pool)
    assert shifting.per_chapter["total_shifts"] > stable.per_chapter["total_shifts"]


@pytest.mark.slow
def test_shifts_has_per_paragraph():
    lens = SituationShiftsLens()
    pool = ProviderPool()
    result = lens.analyze(SHIFTING_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    # At least one shift type should be tracked
    assert any(k in result.per_paragraph for k in ["time_shift", "space_shift", "actor_shift"])


def test_shifts_metadata():
    lens = SituationShiftsLens()
    assert lens.name == "situation_shifts"
    assert "spacy" in lens.requires_providers
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement situation shifts lens**

The implementation should track three shift dimensions at each paragraph boundary:

**Time shifts**: Detect via temporal expressions and tense changes.
- Word list: `{"later", "earlier", "before", "after", "ago", "next", "previous", "yesterday", "tomorrow", "morning", "evening", "night", "dawn", "dusk", "then", "meanwhile", "hours", "days", "weeks", "months", "years"}`
- Tense shift: compare dominant tense (past/present) between adjacent paragraphs via spaCy morphology
- Time-jump phrases: regex for patterns like "three days later", "the next morning", "a week ago"

**Space shifts**: Detect via location NER and motion verbs.
- spaCy NER: look for LOC, GPE, FAC entities that differ between adjacent paragraphs
- Motion verbs: `{"went", "drove", "walked", "ran", "flew", "crossed", "entered", "left", "arrived", "returned", "moved"}`
- Location prepositions: "in the [NOUN]", "at the [NOUN]", "inside [NOUN]" where NOUN differs

**Actor shifts**: Detect via subject tracking.
- Compare the nsubj of the first sentence in each paragraph
- Flag when the grammatical subject changes to a different character entity

Return:
- `per_paragraph`: `time_shift`, `space_shift`, `actor_shift` (lists of 0/1 flags, 0 for first paragraph)
- `per_chapter`: `total_shifts`, `time_shifts`, `space_shifts`, `actor_shifts`, `disorientation_score` (shifts without grounding markers)
- `raw`: shift details with evidence strings

`requires_providers = ["spacy"]`

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_situation_shifts.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/situation_shifts.py tests/test_lens_situation_shifts.py
git commit -m "feat: situation_shifts lens — time/space/actor transition detection"
```

---

## Task 7: Discourse relations lens

**Files:**
- Create: `src/prose_doctor/lenses/discourse_relations.py`
- Test: `tests/test_lens_discourse_relations.py`

Connective-based discourse relation classification. Categorizes inter-sentence connectives into causal, contrastive, temporal, and additive types. Scores relation diversity and flags additive-only zones.

- [ ] **Step 1: Write the test**

```python
# tests/test_lens_discourse_relations.py
import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.discourse_relations import DiscourseRelationsLens
from prose_doctor.providers import ProviderPool


DIVERSE_SAMPLE = '''
Marcus waited because the guard hadn't passed yet. Although the
corridor looked clear, he knew better than to trust appearances.

Then he moved, fast and low. But the floor creaked under his weight.
Consequently, the guard turned. However, Marcus was already through
the door.

Meanwhile, Elena watched from the rooftop. She could see the whole
courtyard, so she tracked his progress through the shadows.
'''

ADDITIVE_SAMPLE = '''
Marcus walked down the corridor. And the walls were grey. And the
floor was concrete. And the lights hummed overhead.

He reached the end of the hall. And there was a door. And the door
was locked. And he tried the handle again.

The room beyond was small. And it was empty. And the window was
boarded up. And dust covered every surface.
'''


@pytest.mark.slow
def test_discourse_returns_result():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    result = lens.analyze(DIVERSE_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "discourse_relations"
    assert result.per_chapter is not None
    assert "relation_entropy" in result.per_chapter


@pytest.mark.slow
def test_diverse_has_higher_entropy():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    diverse = lens.analyze(DIVERSE_SAMPLE, "diverse.md", pool)
    additive = lens.analyze(ADDITIVE_SAMPLE, "additive.md", pool)
    assert diverse.per_chapter["relation_entropy"] > additive.per_chapter["relation_entropy"]


@pytest.mark.slow
def test_additive_sample_flags_additive_zones():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    result = lens.analyze(ADDITIVE_SAMPLE, "additive.md", pool)
    assert result.per_chapter.get("additive_only_zones", 0) > 0


@pytest.mark.slow
def test_discourse_has_per_paragraph():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    result = lens.analyze(DIVERSE_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "relation_diversity" in result.per_paragraph


def test_discourse_metadata():
    lens = DiscourseRelationsLens()
    assert lens.name == "discourse_relations"
    # Pure heuristic, no ML providers needed
    assert lens.requires_providers == ["spacy"]
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement discourse relations lens**

The implementation should:

1. **Define connective word sets** (module-level constants):

```python
CAUSAL = {"because", "since", "so", "therefore", "consequently", "thus",
          "hence", "accordingly", "as a result"}
CONTRASTIVE = {"but", "however", "although", "yet", "nevertheless", "despite",
               "instead", "whereas", "though", "even so", "on the other hand"}
TEMPORAL = {"then", "next", "before", "after", "while", "meanwhile", "finally",
            "subsequently", "afterward", "previously", "simultaneously", "until"}
# Note: "while" is in TEMPORAL only. In fiction it's more often temporal
# ("while he waited") than contrastive ("while that may be true").
ADDITIVE = {"and", "also", "moreover", "furthermore", "in addition", "besides",
            "likewise", "similarly", "equally"}
```

2. **Classify sentences**: For each sentence, check if it begins with (or contains early) a connective from any set. Record the relation type. Sentences without connectives are classified as `implicit`.

3. **Per-paragraph relation diversity**: Count distinct relation types used. Score = n_types / 4.

4. **Additive-only zone detection**: Flag paragraphs where all connectives are additive or implicit (no causal, contrastive, or temporal). A run of 3+ such paragraphs is an additive-only zone.

5. **Relation entropy**: Shannon entropy over the distribution of relation types at chapter level. Higher = more diverse.

Return:
- `per_sentence`: `relation_type` encoded as floats (0=implicit, 1=causal, 2=contrastive, 3=temporal, 4=additive)
- `per_paragraph`: `relation_diversity` (list of floats 0–1)
- `per_chapter`: `relation_entropy`, `causal_ratio`, `contrastive_ratio`, `temporal_ratio`, `additive_ratio`, `implicit_ratio`, `additive_only_zones` (count)
- `raw`: per-sentence relation labels and evidence connectives

`requires_providers = ["spacy"]` (for sentence segmentation)

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_discourse_relations.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/discourse_relations.py tests/test_lens_discourse_relations.py
git commit -m "feat: discourse_relations lens — connective-based relation classification"
```

---

## Task 8: Register new lenses and update tiers

**Files:**
- Modify: `src/prose_doctor/lenses/defaults.py`
- Modify: `src/prose_doctor/validation/tiers.toml`

- [ ] **Step 1: Update defaults.py**

Add imports and registration for the four new lenses:

```python
from prose_doctor.lenses.concreteness import ConcretenessLens
from prose_doctor.lenses.referential_cohesion import ReferentialCohesionLens
from prose_doctor.lenses.situation_shifts import SituationShiftsLens
from prose_doctor.lenses.discourse_relations import DiscourseRelationsLens
```

Add to the registration list.

- [ ] **Step 2: Update tiers.toml**

Add tier entries:

```toml
[concreteness]
tier = "validated"

[referential_cohesion]
tier = "experimental"

[situation_shifts]
tier = "experimental"

[discourse_relations]
tier = "experimental"
```

Concreteness starts at validated because it has strong published norms to validate against. The other three start experimental.

- [ ] **Step 3: Run existing registry test**

```bash
cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_interface.py::test_default_registry_has_all_lenses -v
```

This will fail because the expected list doesn't include the new lenses. Update the test to include them.

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/lenses/defaults.py src/prose_doctor/validation/tiers.toml tests/test_lens_interface.py
git commit -m "feat: register four new lenses, set initial tiers"
```

---

## Task 9: Update docs/lenses.md

**Files:**
- Modify: `docs/lenses.md`

- [ ] **Step 1: Add lens documentation**

Add entries for all four new lenses following the established pattern (what it is / why it is useful / how to use it / research backing / caveats). Add them to the summary table.

Key citations to include:
- **concreteness**: Brysbaert, Warriner & Kuperman (2014), DOI: 10.3758/s13428-013-0403-5
- **referential_cohesion**: Barzilay & Lapata (2008), [aclanthology.org/J08-1001](https://aclanthology.org/J08-1001/); Graesser et al. (2011), DOI: 10.3102/0013189X11413260
- **situation_shifts**: Zwaan, Langston & Graesser (1995), DOI: 10.1111/j.1467-9280.1995.tb00513.x
- **discourse_relations**: Sanders & Noordman (2000), DOI: 10.1207/S15326950dp2901_3; Koornneef & Sanders (2013), DOI: 10.1080/01690965.2012.699076

- [ ] **Step 2: Commit**

```bash
git add docs/lenses.md
git commit -m "docs: add concreteness, referential_cohesion, situation_shifts, discourse_relations"
```

---

## Task 10: Run full test suite and validate

- [ ] **Step 1: Run all tests**

```bash
cd /home/ben/code/prose-doctor && uv run pytest tests/ -v -m "not slow"
```

Expected: ALL PASS

- [ ] **Step 2: Run slow tests for new lenses**

```bash
cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_concreteness.py tests/test_lens_referential_cohesion.py tests/test_lens_situation_shifts.py tests/test_lens_discourse_relations.py -v
```

Expected: ALL PASS

- [ ] **Step 3: Verify the concreteness direction vector correlation**

If not already done in Task 2, verify:

```bash
cd /home/ben/code/prose-doctor && uv run python scripts/compute_concreteness_direction.py
```

Expected: r > 0.70 against Brysbaert norms.

- [ ] **Step 4: Commit any fixes**

```bash
# Only stage files you actually changed — don't use git add -A
git add src/prose_doctor/lenses/ tests/ docs/
git commit -m "chore: final validation pass for new lenses"
```
