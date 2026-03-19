#!/usr/bin/env python3
"""Merge bootstrap-labeled corpus with existing training data, balance, and emit.

Usage:
    uv run python merge_and_balance.py
    uv run python merge_and_balance.py --labeled corpus/labeled.jsonl --existing ../nladfg/story_tracker/data/training/classifier_dataset.jsonl
    uv run python merge_and_balance.py --clean-cap 0.5  # cap clean at 50%
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

CLASS_NAMES = [
    "clean", "thesis", "emotion", "dead_figure",
    "standalone", "narrator_gloss", "forbidden", "padding",
]

ROOT = Path(__file__).parent
DEFAULT_LABELED = ROOT / "corpus" / "labeled.jsonl"
DEFAULT_EXISTING = ROOT.parent / "nladfg" / "story_tracker" / "data" / "training" / "classifier_dataset.jsonl"
DEFAULT_OUTPUT = ROOT.parent / "nladfg" / "story_tracker" / "data" / "training" / "classifier_dataset.jsonl"


def text_hash(text: str) -> str:
    """Stable hash for deduplication."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_record(rec: dict, source: str) -> dict:
    """Normalize a record to the training dataset schema."""
    return {
        "text": rec["text"],
        "label": rec["label"],
        "pattern": rec.get("pattern"),
        "class_name": rec.get("class_name", CLASS_NAMES[rec["label"]]),
        "context_before": rec.get("context_before", ""),
        "context_after": rec.get("context_after", ""),
        "source": source,
    }


def merge_and_dedupe(
    existing: list[dict],
    labeled: list[dict],
    skip_review: bool = True,
) -> list[dict]:
    """Merge datasets, deduplicating on text hash."""
    seen = set()
    merged = []

    # Existing data takes priority
    for rec in existing:
        h = text_hash(rec["text"])
        if h not in seen:
            seen.add(h)
            merged.append(normalize_record(rec, rec.get("source", "existing")))

    # Add labeled corpus data
    skipped_review = 0
    skipped_dupe = 0
    added = 0
    for rec in labeled:
        if skip_review and rec.get("method") == "needs_review":
            skipped_review += 1
            continue
        h = text_hash(rec["text"])
        if h in seen:
            skipped_dupe += 1
            continue
        seen.add(h)
        source = f"corpus_{rec.get('source_model', 'unknown')}"
        merged.append(normalize_record(rec, source))
        added += 1

    print(f"Existing: {len(existing)}")
    print(f"Corpus added: {added}")
    print(f"Corpus skipped (duplicate): {skipped_dupe}")
    print(f"Corpus skipped (needs_review): {skipped_review}")
    print(f"Merged total: {len(merged)}")

    return merged


def balance_classes(
    records: list[dict],
    clean_cap: float = 0.60,
    seed: int = 42,
) -> list[dict]:
    """Balance classes: oversample minority, cap clean to prevent dominance."""
    rng = random.Random(seed)

    by_class: dict[int, list[dict]] = {i: [] for i in range(len(CLASS_NAMES))}
    for rec in records:
        by_class[rec["label"]].append(rec)

    counts = {i: len(v) for i, v in by_class.items()}
    total = sum(counts.values())

    print(f"\nPre-balance distribution:")
    for i, name in enumerate(CLASS_NAMES):
        pct = counts[i] / total * 100 if total else 0
        print(f"  {name}: {counts[i]} ({pct:.1f}%)")

    # Cap clean class
    max_clean = int(total * clean_cap)
    if counts[0] > max_clean:
        rng.shuffle(by_class[0])
        by_class[0] = by_class[0][:max_clean]
        print(f"\nCapped clean from {counts[0]} to {max_clean}")

    # Oversample minority classes to reach median count
    non_clean_counts = [len(by_class[i]) for i in range(1, len(CLASS_NAMES)) if by_class[i]]
    if non_clean_counts:
        target = int(sorted(non_clean_counts)[len(non_clean_counts) // 2] * 1.2)  # 120% of median
        for i in range(1, len(CLASS_NAMES)):
            cls_recs = by_class[i]
            if not cls_recs:
                continue
            if len(cls_recs) < target:
                # Oversample with replacement
                extra_needed = target - len(cls_recs)
                oversampled = rng.choices(cls_recs, k=extra_needed)
                by_class[i] = cls_recs + oversampled
                print(f"Oversampled {CLASS_NAMES[i]}: {len(cls_recs)} → {len(by_class[i])}")

    # Reassemble
    balanced = []
    for i in range(len(CLASS_NAMES)):
        balanced.extend(by_class[i])
    rng.shuffle(balanced)

    counts = Counter(rec["label"] for rec in balanced)
    total = len(balanced)
    print(f"\nPost-balance distribution ({total} total):")
    for i, name in enumerate(CLASS_NAMES):
        pct = counts[i] / total * 100 if total else 0
        print(f"  {name}: {counts[i]} ({pct:.1f}%)")

    return balanced


def main():
    parser = argparse.ArgumentParser(description="Merge and balance training datasets")
    parser.add_argument("--labeled", type=Path, default=DEFAULT_LABELED, help="Bootstrap-labeled JSONL")
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING, help="Existing training JSONL")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output merged JSONL")
    parser.add_argument("--clean-cap", type=float, default=0.60, help="Max fraction of clean class (default: 0.60)")
    parser.add_argument("--include-review", action="store_true", help="Include needs_review records (default: skip)")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without writing")
    args = parser.parse_args()

    if not args.labeled.exists():
        print(f"Labeled file not found: {args.labeled}", file=sys.stderr)
        print("Run label_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    if not args.existing.exists():
        print(f"Existing dataset not found: {args.existing}", file=sys.stderr)
        sys.exit(1)

    existing = load_jsonl(args.existing)
    labeled = load_jsonl(args.labeled)

    merged = merge_and_dedupe(existing, labeled, skip_review=not args.include_review)
    balanced = balance_classes(merged, clean_cap=args.clean_cap)

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Back up existing dataset
    backup = args.output.with_suffix(".jsonl.bak")
    if args.output.exists():
        import shutil
        shutil.copy2(args.output, backup)
        print(f"\nBacked up existing dataset to {backup}")

    with open(args.output, "w") as f:
        for rec in balanced:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(balanced)} records to {args.output}")


if __name__ == "__main__":
    main()
