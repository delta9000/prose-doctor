#!/usr/bin/env python3
"""Build a clean human-written prose dataset for classifier training.

Sources:
  1. Despina/project_gutenberg (HuggingFace) — 6,322 fiction books, public domain
  2. CC-licensed contemporary fiction (Doctorow, Watts, BCS)
  3. Gutenberg plain texts already downloaded

Outputs paragraphs labeled as 'clean' with source metadata.

Usage:
    uv run --with datasets --with prose-doctor python build_human_dataset.py
    uv run --with datasets --with prose-doctor python build_human_dataset.py --max-paragraphs 50000
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

OUTPATH = Path("corpus/human_clean.jsonl")

# Minimum paragraph quality filters
MIN_WORDS = 15
MAX_WORDS = 500
MIN_ALPHA_RATIO = 0.7  # reject paragraphs that are mostly non-alpha (tables, headers)


def is_good_paragraph(text: str) -> bool:
    """Filter out non-prose paragraphs."""
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    # Must be mostly alphabetic (reject tables, code, metadata)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars / max(len(text), 1) < MIN_ALPHA_RATIO:
        return False
    # Reject obvious Gutenberg boilerplate
    lower = text.lower()
    if any(marker in lower for marker in [
        "project gutenberg", "public domain", "copyright", "disclaimer",
        "electronic text", "e-text", "etext", "distributed proofreading",
        "transcriber", "illustration", "table of contents", "chapter ",
        "www.", "http", ".org", ".com", "gutenberg.org",
    ]):
        return False
    # Reject lines that look like headers/metadata
    if text.startswith(("CHAPTER", "PART ", "BOOK ", "VOLUME", "ACT ")):
        return False
    if text.isupper():
        return False
    return True


def clean_text(text: str) -> str:
    """Clean up common Gutenberg artifacts."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading/trailing underscores (italic markers)
    text = text.strip('_').strip()
    return text


def extract_paragraphs_from_text(text: str) -> list[str]:
    """Split text into paragraphs and filter."""
    # Split on double newlines
    raw_paras = re.split(r'\n\s*\n', text)
    result = []
    for p in raw_paras:
        cleaned = clean_text(p)
        if is_good_paragraph(cleaned):
            result.append(cleaned)
    return result


def harvest_gutenberg_hf(max_books: int = 2000, seed: int = 42) -> list[dict]:
    """Sample paragraphs from the HuggingFace Gutenberg fiction dataset."""
    from datasets import load_dataset

    print("Loading Gutenberg fiction from HuggingFace (streaming)...")
    ds = load_dataset(
        "Despina/project_gutenberg", "fiction_books_in_chunks",
        split="train", streaming=True,
    )

    # Collect chunks by book, sample books for diversity
    books: dict[int, dict] = {}
    chunk_count = 0

    for ex in ds:
        bid = ex["book_id"]
        if bid not in books:
            if len(books) >= max_books:
                break
            books[bid] = {
                "title": ex["title"],
                "author": ex["author"],
                "birth": ex.get("author_birth_year"),
                "gender": ex.get("author_gender", "unknown"),
                "chunks": [],
            }
        books[bid]["chunks"].append(ex["chunk"])
        chunk_count += 1
        if chunk_count % 10000 == 0:
            print(f"  {chunk_count} chunks from {len(books)} books...")

    print(f"  Loaded {chunk_count} chunks from {len(books)} books")

    # Extract paragraphs from each book
    rng = random.Random(seed)
    records = []
    for bid, book in books.items():
        full_text = "\n\n".join(book["chunks"])
        paras = extract_paragraphs_from_text(full_text)
        if not paras:
            continue

        # Sample up to 30 paragraphs per book for diversity
        sampled = rng.sample(paras, min(30, len(paras)))
        for idx, para in enumerate(sampled):
            # Get context
            all_paras = paras
            pi = paras.index(para) if para in paras else idx
            ctx_before = all_paras[pi - 1] if pi > 0 else ""
            ctx_after = all_paras[pi + 1] if pi < len(all_paras) - 1 else ""

            records.append({
                "text": para,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "label": 0,
                "class_name": "clean",
                "pattern": None,
                "source": f"gutenberg_{bid}",
                "source_meta": {
                    "title": book["title"],
                    "author": book["author"],
                    "birth_year": book["birth"],
                    "gender": book["gender"],
                },
            })

    print(f"  Extracted {len(records)} paragraphs from {len(books)} books")
    return records


def harvest_local_gutenberg() -> list[dict]:
    """Extract paragraphs from locally downloaded Gutenberg texts."""
    gutenberg_dir = Path("/tmp/gutenberg")
    if not gutenberg_dir.exists():
        return []

    records = []
    sources = {
        "frankenstein.txt": ("Mary Shelley", "Frankenstein", 1818),
        "heart_of_darkness.txt": ("Joseph Conrad", "Heart of Darkness", 1899),
        "yellow_wallpaper.txt": ("Charlotte Perkins Gilman", "The Yellow Wallpaper", 1892),
        "tale_two_cities.txt": ("Charles Dickens", "A Tale of Two Cities", 1859),
        "pride_full.txt": ("Jane Austen", "Pride and Prejudice", 1813),
    }

    for fname, (author, title, year) in sources.items():
        fpath = gutenberg_dir / fname
        if not fpath.exists():
            continue
        text = fpath.read_text()
        # Strip Gutenberg header/footer
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        if start > 0 and end > start:
            text = text[start:end]
            # Skip past the START line
            text = text[text.index("\n") + 1:]

        paras = extract_paragraphs_from_text(text)
        rng = random.Random(42)
        sampled = rng.sample(paras, min(50, len(paras)))

        for idx, para in enumerate(sampled):
            pi = paras.index(para) if para in paras else idx
            records.append({
                "text": para,
                "context_before": paras[pi - 1] if pi > 0 else "",
                "context_after": paras[pi + 1] if pi < len(paras) - 1 else "",
                "label": 0,
                "class_name": "clean",
                "pattern": None,
                "source": f"gutenberg_{title.lower().replace(' ', '_')}",
                "source_meta": {"title": title, "author": author, "year": year},
            })

    print(f"  Local Gutenberg: {len(records)} paragraphs")
    return records


def harvest_cc_fiction() -> list[dict]:
    """Download and extract CC-licensed contemporary fiction."""
    import urllib.request

    cc_sources = [
        {
            "url": "https://www.rifters.com/real/Blindsight.htm",
            "author": "Peter Watts",
            "title": "Blindsight",
            "year": 2006,
            "is_html": True,
        },
        {
            "url": "https://craphound.com/littlebrother/Cory_Doctorow_-_Little_Brother.txt",
            "author": "Cory Doctorow",
            "title": "Little Brother",
            "year": 2008,
            "is_html": False,
        },
    ]

    records = []
    rng = random.Random(42)

    for src in cc_sources:
        print(f"  Downloading {src['title']}...")
        try:
            req = urllib.request.Request(src["url"], headers={"User-Agent": "prose-doctor/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"    Failed: {e}", file=sys.stderr)
            continue

        if src["is_html"]:
            # Strip HTML tags
            text = re.sub(r'<[^>]+>', '', raw)
            text = re.sub(r'&[a-z]+;', ' ', text)
        else:
            text = raw

        paras = extract_paragraphs_from_text(text)
        sampled = rng.sample(paras, min(100, len(paras)))

        for idx, para in enumerate(sampled):
            pi = paras.index(para) if para in paras else idx
            records.append({
                "text": para,
                "context_before": paras[pi - 1] if pi > 0 else "",
                "context_after": paras[pi + 1] if pi < len(paras) - 1 else "",
                "label": 0,
                "class_name": "clean",
                "pattern": None,
                "source": f"cc_{src['title'].lower().replace(' ', '_')}",
                "source_meta": {
                    "title": src["title"],
                    "author": src["author"],
                    "year": src["year"],
                    "license": "CC BY-NC-SA",
                },
            })

    print(f"  CC fiction: {len(records)} paragraphs")
    return records


def dedupe(records: list[dict]) -> list[dict]:
    """Deduplicate on text hash."""
    seen = set()
    result = []
    for r in records:
        h = hashlib.sha256(r["text"].strip().encode()).hexdigest()[:16]
        if h not in seen:
            seen.add(h)
            result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description="Build human prose dataset")
    parser.add_argument("--max-paragraphs", type=int, default=50000)
    parser.add_argument("--max-books", type=int, default=2000,
                        help="Max books from HuggingFace Gutenberg")
    parser.add_argument("--out", type=Path, default=OUTPATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_records = []

    # Source 1: HuggingFace Gutenberg
    gutenberg_hf = harvest_gutenberg_hf(max_books=args.max_books, seed=args.seed)
    all_records.extend(gutenberg_hf)

    # Source 2: Local Gutenberg texts
    local_gut = harvest_local_gutenberg()
    all_records.extend(local_gut)

    # Source 3: CC-licensed contemporary fiction
    cc = harvest_cc_fiction()
    all_records.extend(cc)

    # Deduplicate
    all_records = dedupe(all_records)
    print(f"\nTotal after dedup: {len(all_records)}")

    # Cap to max
    rng = random.Random(args.seed)
    if len(all_records) > args.max_paragraphs:
        all_records = rng.sample(all_records, args.max_paragraphs)
        print(f"Sampled down to {args.max_paragraphs}")

    # Strip source_meta for JSONL (keep it lightweight)
    for r in all_records:
        r.pop("source_meta", None)

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # Stats
    from collections import Counter
    sources = Counter(r["source"].split("_")[0] for r in all_records)
    print(f"\nWrote {len(all_records)} paragraphs to {args.out}")
    print("Source breakdown:")
    for src, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
