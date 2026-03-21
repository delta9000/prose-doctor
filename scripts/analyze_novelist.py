#!/usr/bin/env python3
"""Run the four new experimental lenses across all full_book records in the Novelist dataset."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from prose_doctor.providers import ProviderPool, require_ml
from prose_doctor.lenses import LensRegistry
from prose_doctor.lenses.runner import LensRunner
from prose_doctor.lenses.concreteness import ConcretenessLens
from prose_doctor.lenses.referential_cohesion import ReferentialCohesionLens
from prose_doctor.lenses.situation_shifts import SituationShiftsLens
from prose_doctor.lenses.discourse_relations import DiscourseRelationsLens
from prose_doctor.text import split_paragraphs

DATA_PATH = Path(__file__).resolve().parent.parent / "refs" / "novelist" / "data.jsonl"
OUT_PATH = Path(__file__).resolve().parent.parent / "refs" / "novelist" / "lens_results.jsonl"

TARGET_LENSES = ["concreteness", "referential_cohesion", "situation_shifts", "discourse_relations"]

# Regex to split on chapter headings: newline followed by "Chapter" or "**Chapter" + digit
CH_SPLIT_RE = re.compile(r"\n(?=\*{0,2}Chapter\s+\d)", re.IGNORECASE)


def extract_chapter1(text: str) -> str | None:
    """Extract chapter 1 text, truncated to ~2000 words preserving paragraph breaks."""
    parts = CH_SPLIT_RE.split(text)
    if len(parts) < 2:
        return None
    ch1_raw = parts[1]  # first chapter after splitting

    # Truncate to ~2000 words preserving paragraph breaks
    lines = ch1_raw.split("\n")
    result_lines: list[str] = []
    word_count = 0
    for line in lines:
        words_in_line = len(line.split())
        if word_count + words_in_line > 2000 and word_count > 0:
            break
        result_lines.append(line)
        word_count += words_in_line

    return "\n".join(result_lines)


def parse_quality_json(raw) -> dict:
    """Parse quality_json which may be a dict, a JSON string, or empty."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def build_runner() -> LensRunner:
    """Build a LensRunner with only the four target lenses."""
    registry = LensRegistry()
    for cls in [ConcretenessLens, ReferentialCohesionLens, SituationShiftsLens, DiscourseRelationsLens]:
        registry.register(cls())
    providers = ProviderPool()
    return LensRunner(registry, providers)


def main() -> None:
    require_ml()

    # Resume support: count existing output lines
    skip = 0
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            skip = sum(1 for _ in f)
        if skip > 0:
            print(f"Resuming: skipping first {skip} books (already written)")

    print("Building runner...")
    runner = build_runner()

    processed = 0
    skipped_for_resume = 0
    t0 = time.time()

    with open(DATA_PATH) as fin, open(OUT_PATH, "a") as fout:
        for line_no, line in enumerate(fin, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rec.get("record_type") != "full_book":
                continue

            # Resume: skip already-processed books
            if skipped_for_resume < skip:
                skipped_for_resume += 1
                continue

            book_id = rec.get("book_id", rec.get("record_id", f"line_{line_no}"))

            try:
                text = rec.get("text", "")
                ch1 = extract_chapter1(text)
                if ch1 is None:
                    print(f"  WARN: no chapter split found for {book_id}, skipping")
                    continue

                paragraphs = split_paragraphs(ch1)
                if len(paragraphs) < 3:
                    print(f"  WARN: <3 paragraphs for {book_id}, skipping")
                    continue

                ch1_words = len(ch1.split())

                # Run lenses
                results = runner.run_all(ch1, str(book_id))

                # Build output row
                qj = parse_quality_json(rec.get("quality_json"))

                row: dict = {
                    "book_id": book_id,
                    "story_type": rec.get("story_type"),
                    "genre": rec.get("genre"),
                    "author_style": rec.get("author_style"),
                    "pov": rec.get("pov"),
                    "tone": rec.get("tone"),
                    "total_word_count": rec.get("text_word_count"),
                    "ch1_words": ch1_words,
                    "ch1_paragraphs": len(paragraphs),
                    "writer_average_overall": qj.get("writer_average_overall"),
                    "writer_min_overall": qj.get("writer_min_overall"),
                }

                # Flatten lens metrics
                for lens_name in TARGET_LENSES:
                    lr = results.get(lens_name)
                    if lr is None:
                        continue
                    if lr.per_chapter:
                        for k, v in lr.per_chapter.items():
                            row[f"{lens_name}__{k}"] = v

                fout.write(json.dumps(row) + "\n")
                fout.flush()
                processed += 1

                if processed % 100 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  [{processed} books] {elapsed:.1f}s elapsed, {rate:.1f} books/s")

            except Exception as e:
                print(f"  WARN: error on {book_id}: {e}")
                continue

    elapsed = time.time() - t0
    print(f"\nDone. {processed} books processed in {elapsed:.1f}s")
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
