"""Story sampler — chapter extraction from the Novelist dataset.

Provides utilities for pulling representative chapter-length excerpts from
refs/novelist/data.jsonl, stratified by genre, for use in arena comparisons.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Chapter heading patterns
# ---------------------------------------------------------------------------

# Matches:
#   **Chapter 3: Title**
#   **Chapter 3**
#   ## Chapter 3: Title
#   ## Chapter 3
#   ### Chapter 3: Title
_HEADING_RE = re.compile(
    r"(?m)^(?:"
    r"\*\*Chapter\s+(\d+)(?:[:\s][^\n]*)?\*\*"
    r"|"
    r"#{2,3}\s+Chapter\s+(\d+)(?:[:\s][^\n]*)?"
    r")$"
)


def extract_chapter(text: str, chapter_num: int = 1) -> str | None:
    """Return the body text of *chapter_num* from *text*, or None if not found.

    A chapter boundary is any line matching one of the supported heading
    patterns (``**Chapter N…**`` or ``## / ### Chapter N…``).  The returned
    string spans from the line *after* the matching heading to the line
    *before* the next heading (or end of text), stripped of leading/trailing
    whitespace.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return None

    target_match = None
    target_index = None
    for i, m in enumerate(matches):
        num_str = m.group(1) or m.group(2)
        if int(num_str) == chapter_num:
            target_match = m
            target_index = i
            break

    if target_match is None:
        return None

    # Body starts immediately after the heading line ends.
    body_start = target_match.end()

    # Body ends at the start of the next heading, or at end of text.
    if target_index + 1 < len(matches):
        next_match = matches[target_index + 1]
        # Walk back to include the newline before the heading.
        body_end = next_match.start()
    else:
        body_end = len(text)

    return text[body_start:body_end].strip()


# ---------------------------------------------------------------------------
# Dataset iteration
# ---------------------------------------------------------------------------

def _iter_full_books(dataset_path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects where record_type == 'full_book'."""
    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("record_type") == "full_book":
                yield obj


def _story_id(record: dict) -> str:
    """Return the best available ID for a full_book record."""
    return record.get("book_id") or record.get("record_id") or record.get("row_id", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_stories(
    dataset_path: str | Path,
    n: int = 20,
    chapter_range: tuple[int, int] = (3, 8),
    min_words: int = 1500,
    max_words: int = 4000,
    exclude_ids: set[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sample *n* stories from the Novelist dataset, stratified by genre.

    For each book the sampler tries chapters *chapter_range[0]* through
    *chapter_range[1]* (inclusive) in order and picks the first chapter whose
    word count falls within [*min_words*, *max_words*].

    Candidates are grouped by genre and sampled proportionally so that rare
    genres are not excluded.  Any remainder after proportional allocation is
    drawn randomly from the full remaining pool.

    Parameters
    ----------
    dataset_path:
        Path to ``refs/novelist/data.jsonl``.
    n:
        Desired number of stories to return.
    chapter_range:
        ``(first, last)`` chapter numbers to try (inclusive).
    min_words:
        Minimum word count for an extracted chapter to be eligible.
    max_words:
        Maximum word count for an extracted chapter to be eligible.
    exclude_ids:
        Set of story IDs to skip (e.g. a held-out set).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list of dicts with keys: ``story_id``, ``genre``, ``text``,
    ``word_count``, ``chapter_num``.
    """
    dataset_path = Path(dataset_path)
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Pass 1: collect all candidates that satisfy word-count constraints.
    # ------------------------------------------------------------------
    # genre -> list of candidate dicts
    by_genre: dict[str, list[dict]] = defaultdict(list)

    for record in _iter_full_books(dataset_path):
        sid = _story_id(record)
        if sid in exclude_ids:
            continue

        text = record.get("text", "")
        genre = record.get("genre", "unknown") or "unknown"

        for ch_num in range(chapter_range[0], chapter_range[1] + 1):
            chapter_text = extract_chapter(text, chapter_num=ch_num)
            if chapter_text is None:
                continue
            wc = len(chapter_text.split())
            if min_words <= wc <= max_words:
                by_genre[genre].append(
                    {
                        "story_id": sid,
                        "genre": genre,
                        "text": chapter_text,
                        "word_count": wc,
                        "chapter_num": ch_num,
                    }
                )
                break  # first eligible chapter wins

    if not by_genre:
        return []

    # ------------------------------------------------------------------
    # Pass 2: stratified sampling.
    # ------------------------------------------------------------------
    genres = list(by_genre.keys())
    per_genre = n // len(genres)
    selected: list[dict] = []
    leftover: list[dict] = []

    for genre, candidates in by_genre.items():
        rng.shuffle(candidates)
        selected.extend(candidates[:per_genre])
        leftover.extend(candidates[per_genre:])

    # Fill remainder randomly from the leftover pool.
    remainder = n - len(selected)
    if remainder > 0 and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[:remainder])

    # Trim to exactly n in case we over-selected.
    rng.shuffle(selected)
    return selected[:n]


def create_holdout(
    dataset_path: str | Path,
    holdout_path: str | Path,
    n: int = 50,
    seed: int = 99,
) -> set[str]:
    """Sample *n* story IDs and write them to *holdout_path* (one per line).

    Parent directories are created automatically.  Returns the set of IDs
    written so callers can pass them directly to :func:`sample_stories` as
    ``exclude_ids``.
    """
    stories = sample_stories(dataset_path, n=n, seed=seed)
    ids = {s["story_id"] for s in stories}

    holdout_path = Path(holdout_path)
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")

    return ids
