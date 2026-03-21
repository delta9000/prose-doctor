"""Tests for prose_doctor.arena.sampler."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from prose_doctor.arena.sampler import extract_chapter, sample_stories, create_holdout


# ---------------------------------------------------------------------------
# extract_chapter
# ---------------------------------------------------------------------------

def test_extract_chapter_by_heading():
    text = (
        "**Chapter 1: The Start**\n\nFirst chapter text here. "
        "Enough words to count.\n\n"
        "**Chapter 2: The Middle**\n\nSecond chapter text with "
        "more words to satisfy the minimum.\n\n"
        "**Chapter 3: The End**\n\nThird chapter wraps it up."
    )
    ch2 = extract_chapter(text, chapter_num=2)
    assert ch2 is not None
    assert "Second chapter" in ch2
    assert "First chapter" not in ch2
    assert "Third chapter" not in ch2


def test_extract_chapter_hash_heading():
    text = "## Chapter 1: Intro\n\nIntro text.\n\n## Chapter 2: Middle\n\nMiddle text."
    ch1 = extract_chapter(text, chapter_num=1)
    assert ch1 is not None
    assert "Intro text" in ch1


def test_extract_chapter_missing():
    text = "**Chapter 1: Only**\n\nSome text."
    assert extract_chapter(text, chapter_num=5) is None


def test_extract_chapter_triple_hash():
    text = "### Chapter 1: Intro\n\nIntro text.\n\n### Chapter 2: Middle\n\nMiddle text."
    ch2 = extract_chapter(text, chapter_num=2)
    assert ch2 is not None
    assert "Middle text" in ch2
    assert "Intro text" not in ch2


def test_extract_chapter_no_title():
    """Headings without a title suffix should still match."""
    text = "**Chapter 1**\n\nFirst.\n\n**Chapter 2**\n\nSecond."
    ch1 = extract_chapter(text, chapter_num=1)
    assert ch1 is not None
    assert "First" in ch1


def test_extract_chapter_last_chapter_runs_to_end():
    text = "**Chapter 1: Start**\n\nIntro.\n\n**Chapter 2: End**\n\nFinal content here."
    ch2 = extract_chapter(text, chapter_num=2)
    assert ch2 is not None
    assert "Final content here" in ch2


def test_extract_chapter_no_headings_returns_none():
    text = "Just some prose without any chapter headings at all."
    assert extract_chapter(text, chapter_num=1) is None


# ---------------------------------------------------------------------------
# sample_stories / create_holdout — unit tests against a synthetic dataset
# ---------------------------------------------------------------------------

def _make_chapter_text(sentences: int = 300) -> str:
    """Return a predictable block of prose ~300 words."""
    sentence = "The quick brown fox jumps over the lazy dog. "
    return sentence * sentences


def _make_novel(genre: str, book_id: str, num_chapters: int = 10) -> dict:
    parts = []
    for i in range(1, num_chapters + 1):
        parts.append(f"**Chapter {i}: Part {i}**\n\n{_make_chapter_text(300)}")
    text = "\n\n".join(parts)
    return {
        "row_id": f"row_{book_id}",
        "record_type": "full_book",
        "record_id": book_id,
        "story_id": "",
        "book_id": book_id,
        "genre": genre,
        "text": text,
        "text_word_count": len(text.split()),
    }


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Write a small synthetic JSONL dataset with 3 genres, 5 books each."""
    genres = ["fantasy", "thriller", "romance"]
    data_file = tmp_path / "data.jsonl"
    with data_file.open("w", encoding="utf-8") as fh:
        for genre in genres:
            for idx in range(5):
                book_id = f"{genre}_{idx:03d}"
                record = _make_novel(genre, book_id)
                fh.write(json.dumps(record) + "\n")
        # Add a non-full_book record that should be ignored.
        fh.write(
            json.dumps({"record_type": "story_quality", "genre": "fantasy", "text": "ignored"})
            + "\n"
        )
    return data_file


def test_sample_stories_returns_n(synthetic_dataset: Path):
    results = sample_stories(synthetic_dataset, n=6, chapter_range=(3, 8), min_words=100, max_words=99999)
    assert len(results) == 6


def test_sample_stories_result_schema(synthetic_dataset: Path):
    results = sample_stories(synthetic_dataset, n=3, chapter_range=(3, 5), min_words=100, max_words=99999)
    for item in results:
        assert set(item.keys()) == {"story_id", "genre", "text", "word_count", "chapter_num"}
        assert isinstance(item["text"], str)
        assert isinstance(item["word_count"], int)
        assert 3 <= item["chapter_num"] <= 5


def test_sample_stories_genre_stratification(synthetic_dataset: Path):
    """All three genres should appear when requesting a broad sample."""
    results = sample_stories(synthetic_dataset, n=9, chapter_range=(3, 8), min_words=100, max_words=99999)
    genres_present = {r["genre"] for r in results}
    assert genres_present == {"fantasy", "thriller", "romance"}


def test_sample_stories_exclude_ids(synthetic_dataset: Path):
    first_run = sample_stories(synthetic_dataset, n=6, chapter_range=(3, 8), min_words=100, max_words=99999)
    excluded = {r["story_id"] for r in first_run}
    second_run = sample_stories(
        synthetic_dataset, n=6, chapter_range=(3, 8), min_words=100, max_words=99999,
        exclude_ids=excluded,
    )
    second_ids = {r["story_id"] for r in second_run}
    assert second_ids.isdisjoint(excluded)


def test_sample_stories_word_count_filter(synthetic_dataset: Path):
    results = sample_stories(
        synthetic_dataset, n=6, chapter_range=(3, 8), min_words=100, max_words=500
    )
    for item in results:
        assert 100 <= item["word_count"] <= 500


def test_sample_stories_reproducible(synthetic_dataset: Path):
    r1 = sample_stories(synthetic_dataset, n=6, chapter_range=(3, 8), min_words=100, max_words=99999, seed=7)
    r2 = sample_stories(synthetic_dataset, n=6, chapter_range=(3, 8), min_words=100, max_words=99999, seed=7)
    assert [r["story_id"] for r in r1] == [r["story_id"] for r in r2]


def test_create_holdout_writes_file(synthetic_dataset: Path, tmp_path: Path):
    holdout_file = tmp_path / "holdout" / "ids.txt"
    ids = create_holdout(synthetic_dataset, holdout_file, n=5, seed=99)
    assert holdout_file.exists()
    written = set(holdout_file.read_text(encoding="utf-8").splitlines())
    # Filter out empty lines.
    written = {line for line in written if line}
    assert written == ids


def test_create_holdout_creates_parents(synthetic_dataset: Path, tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c" / "ids.txt"
    create_holdout(synthetic_dataset, deep, n=3, seed=1)
    assert deep.exists()
