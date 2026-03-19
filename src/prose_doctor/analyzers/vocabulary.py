"""Vocabulary crutch detection: find overused words."""

from __future__ import annotations

import re
from collections import Counter

# Words that are ALWAYS fine at any frequency
_EXEMPT_WORDS = {
    "the", "a", "an", "and", "but", "or", "in", "on", "at", "to", "for",
    "of", "with", "from", "by", "as", "is", "was", "were", "be", "been",
    "are", "am", "has", "had", "have", "do", "did", "does", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "if", "then", "than", "so", "up", "out", "into", "just", "that", "this",
    "it", "its", "he", "she", "his", "her", "him", "they", "them", "their",
    "we", "us", "our", "you", "your", "i", "my", "me", "who", "what",
    "which", "when", "where", "how", "all", "each", "every", "both",
    "more", "most", "some", "any", "one", "two", "three", "first", "last",
    "said", "like", "through", "still", "back", "over", "down", "there",
    "here", "now", "then", "about", "after", "before", "between", "under",
    "again", "against", "once", "only", "very", "too", "also", "own",
    "other", "another", "same", "such", "even", "new", "old",
}


def find_vocabulary_crutches(
    text: str,
    threshold_per_1k: float = 1.2,
    exempt_words: set[str] | None = None,
) -> list[dict]:
    """Find ANY word used more often than threshold per 1000 words.

    Returns list of {word, count, budget, excess, locations} dicts,
    sorted by excess descending.
    """
    all_exempt = _EXEMPT_WORDS | (exempt_words or set())

    words_raw = re.findall(r"[a-z]+", text.lower())
    total = len(words_raw)
    if total < 100:
        return []

    counts = Counter(words_raw)
    budget_base = max(2, int(threshold_per_1k * total / 1000))

    crutches = []
    for word, count in counts.most_common():
        if len(word) < 4:
            continue
        if word in all_exempt:
            continue
        if count <= budget_base:
            continue

        # Find line numbers where this word appears
        locations = []
        for i, line in enumerate(text.split("\n"), 1):
            if re.search(rf"\b{re.escape(word)}\b", line, re.IGNORECASE):
                locations.append(i)

        crutches.append({
            "word": word,
            "count": count,
            "budget": budget_base,
            "excess": count - budget_base,
            "locations": locations,
        })

    crutches.sort(key=lambda x: -x["excess"])
    return crutches
