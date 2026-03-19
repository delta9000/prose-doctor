#!/usr/bin/env python3
"""Bootstrap-label the LLM corpus for slop classifier training.

Step 1: Split corpus files into paragraphs with context windows.
Step 2: Label each paragraph using rules + existing ML classifier.

Usage:
    uv run --with prose-doctor[ml] python label_corpus.py
    uv run --with prose-doctor[ml] python label_corpus.py --corpus corpus/ --out corpus/labeled.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

from prose_doctor.ml.slop_scorer import SlopScorer
from prose_doctor.patterns.rules import check_rules, build_rule_patterns
from prose_doctor.patterns.taxonomy import CLASS_NAMES
from prose_doctor.text import split_paragraphs

CORPUS_DIR = Path(__file__).parent / "corpus"
LABELED_OUT = CORPUS_DIR / "labeled.jsonl"
REVIEW_OUT = CORPUS_DIR / "needs_review.jsonl"

# Thresholds for auto-labeling
ML_HIGH_CONF = 0.85   # auto-accept ML label above this
ML_LOW_CONF = 0.30    # auto-label clean below this (slop_prob)
RULE_CONF = 0.95      # confidence for rule-matched labels

# Patterns to strip from raw corpus files
THINKING_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)
PREAMBLE_RE = re.compile(
    r"^(?:Here (?:is|are)|Sure[,!]|Certainly[,!]|I'(?:ll|d)|Of course|Below is|"
    r"Let me|Okay|Alright|Got it).*?(?:\n\n|\n(?=[A-Z]))",
    re.IGNORECASE | re.DOTALL,
)
MD_HEADER_RE = re.compile(r"^#{1,3}\s+.*$", re.MULTILINE)


def strip_meta(text: str) -> str:
    """Remove thinking tags, LLM preamble, and markdown headers."""
    text = THINKING_RE.sub("", text)
    text = PREAMBLE_RE.sub("", text)
    text = MD_HEADER_RE.sub("", text)
    return text.strip()


def extract_source_info(filepath: Path) -> tuple[str, str]:
    """Extract (model_name, prompt_id) from corpus file path."""
    model = filepath.parent.name
    prompt = filepath.stem  # e.g. "prompt_00"
    return model, prompt


def extract_paragraphs(corpus_dir: Path) -> list[dict]:
    """Read all corpus files and split into paragraphs with context."""
    records = []
    files = sorted(corpus_dir.glob("*/prompt_*.md"))

    if not files:
        print(f"No corpus files found in {corpus_dir}/*/prompt_*.md", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} corpus files")

    for filepath in files:
        model, prompt = extract_source_info(filepath)
        raw = filepath.read_text()
        cleaned = strip_meta(raw)
        paragraphs = split_paragraphs(cleaned)

        for idx, para in enumerate(paragraphs):
            # Skip very short paragraphs (< 10 words)
            if len(para.split()) < 10:
                continue

            ctx_before = paragraphs[idx - 1] if idx > 0 else ""
            ctx_after = paragraphs[idx + 1] if idx < len(paragraphs) - 1 else ""

            records.append({
                "text": para,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "source_model": model,
                "source_prompt": prompt,
                "paragraph_idx": idx,
            })

    return records


def label_paragraphs(records: list[dict]) -> list[dict]:
    """Bootstrap labels using rules + ML classifier."""
    print("Loading SlopScorer...")
    scorer = SlopScorer()
    rule_patterns = build_rule_patterns()

    labeled = []
    total = len(records)

    for i, rec in enumerate(records):
        if (i + 1) % 200 == 0:
            print(f"  Labeled {i + 1}/{total}...")

        text = rec["text"]
        ctx_before = rec["context_before"]
        ctx_after = rec["context_after"]

        # Rule-based check
        rule_matches = check_rules(text, rule_patterns)
        rule_label = None
        rule_pattern = None
        if rule_matches:
            best = rule_matches[0]
            rule_label = best["class_id"]
            rule_pattern = best["pattern_name"]

        # ML classifier
        ml_result = scorer.score_text_with_context(ctx_before, text, ctx_after)
        ml_prob = ml_result["slop_prob"]
        ml_class_id = ml_result["class_id"]
        ml_class_name = ml_result["class_name"]
        ml_class_prob = ml_result["class_prob"]

        # Auto-assign label
        if rule_label is not None:
            # Rule match → high confidence
            label = rule_label
            class_name = CLASS_NAMES[label]
            confidence = RULE_CONF
            method = "rule"
        elif ml_prob > ML_HIGH_CONF and ml_class_id != 0:
            # ML says slop with high confidence
            label = ml_class_id
            class_name = ml_class_name
            confidence = ml_class_prob
            method = "ml_high"
        elif ml_prob < ML_LOW_CONF:
            # ML says clean with high confidence
            label = 0
            class_name = "clean"
            confidence = 1.0 - ml_prob
            method = "ml_clean"
        else:
            # Ambiguous → needs review
            label = ml_class_id
            class_name = ml_class_name
            confidence = ml_class_prob
            method = "needs_review"

        rec.update({
            "label": label,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "method": method,
            "pattern": rule_pattern,
            "ml_prob": round(ml_prob, 4),
            "ml_class_name": ml_class_name,
            "ml_class_prob": round(ml_class_prob, 4),
        })
        labeled.append(rec)

    return labeled


def write_outputs(labeled: list[dict], labeled_path: Path, review_path: Path):
    """Write labeled.jsonl and needs_review.jsonl."""
    needs_review = [r for r in labeled if r["method"] == "needs_review"]
    auto_labeled = [r for r in labeled if r["method"] != "needs_review"]

    with open(labeled_path, "w") as f:
        for rec in labeled:
            f.write(json.dumps(rec) + "\n")

    with open(review_path, "w") as f:
        for rec in needs_review:
            f.write(json.dumps(rec) + "\n")

    # Stats
    from collections import Counter
    method_counts = Counter(r["method"] for r in labeled)
    class_counts = Counter(r["class_name"] for r in labeled)

    print(f"\nTotal paragraphs: {len(labeled)}")
    print(f"\nLabeling method breakdown:")
    for method, count in sorted(method_counts.items()):
        pct = count / len(labeled) * 100
        print(f"  {method}: {count} ({pct:.1f}%)")

    print(f"\nClass distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = count / len(labeled) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")

    print(f"\nWrote {len(labeled)} records to {labeled_path}")
    print(f"Wrote {len(needs_review)} records to {review_path}")
    review_pct = len(needs_review) / len(labeled) * 100
    print(f"Review queue: {review_pct:.1f}% of total")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap-label corpus for slop classifier")
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR, help="Corpus directory")
    parser.add_argument("--out", type=Path, default=LABELED_OUT, help="Output labeled JSONL")
    parser.add_argument("--review", type=Path, default=REVIEW_OUT, help="Output review queue JSONL")
    parser.add_argument("--extract-only", action="store_true", help="Only extract paragraphs, skip labeling")
    args = parser.parse_args()

    records = extract_paragraphs(args.corpus)
    print(f"Extracted {len(records)} paragraphs (>= 10 words)")

    if args.extract_only:
        with open(args.out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(records)} records to {args.out}")
        return

    labeled = label_paragraphs(records)
    write_outputs(labeled, args.out, args.review)


if __name__ == "__main__":
    main()
