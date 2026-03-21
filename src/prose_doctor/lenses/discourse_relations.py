"""Discourse relations lens -- connective-based relation classification.

Classifies inter-sentence discourse relations (causal, contrastive, temporal,
additive, implicit) by detecting connective phrases near sentence boundaries.
Reports relation entropy, per-paragraph diversity, and additive-only zones.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

CAUSAL = {
    "because", "since", "so", "therefore", "consequently", "thus",
    "hence", "accordingly", "as a result",
}
CONTRASTIVE = {
    "but", "however", "although", "yet", "nevertheless", "despite",
    "instead", "whereas", "though", "even so", "on the other hand",
}
TEMPORAL = {
    "then", "next", "before", "after", "while", "meanwhile", "finally",
    "subsequently", "afterward", "previously", "simultaneously", "until",
}
ADDITIVE = {
    "and", "also", "moreover", "furthermore", "in addition", "besides",
    "likewise", "similarly", "equally",
}

_RELATION_SETS: list[tuple[str, set[str], float]] = [
    ("causal", CAUSAL, 1.0),
    ("contrastive", CONTRASTIVE, 2.0),
    ("temporal", TEMPORAL, 3.0),
    ("additive", ADDITIVE, 4.0),
]


def _classify_sentence(text: str) -> tuple[str, float, str | None]:
    """Classify a sentence's discourse relation based on connectives."""
    lowered = text.lower().strip()

    # Check multi-word connectives first
    for rel_name, connectives, code in _RELATION_SETS:
        for conn in sorted(connectives, key=len, reverse=True):
            if " " in conn:
                if lowered.startswith(conn) and (
                    len(lowered) == len(conn)
                    or not lowered[len(conn)].isalpha()
                ):
                    return rel_name, code, conn

    # Single-word connectives in first few words
    words = lowered.split(None, 6)[:6]
    for word in words[:4]:
        clean = word.rstrip(".,;:!?")
        for rel_name, connectives, code in _RELATION_SETS:
            if clean in connectives and " " not in clean:
                return rel_name, code, clean

    return "implicit", 0.0, None


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


class DiscourseRelationsLens(Lens):
    """Analyse discourse relation diversity and connective usage patterns."""

    name = "discourse_relations"
    requires_providers = ["spacy"]
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        nlp = providers.spacy
        paragraphs = split_paragraphs(text)

        all_relation_types: list[float] = []
        all_relation_labels: list[str] = []
        all_evidence: list[str | None] = []
        para_diversity: list[float] = []
        additive_only_flags: list[bool] = []

        for para in paragraphs:
            doc = nlp(para)
            sents = list(doc.sents)
            para_relations: set[str] = set()

            for sent in sents:
                rel_name, rel_code, evidence = _classify_sentence(sent.text)
                all_relation_types.append(rel_code)
                all_relation_labels.append(rel_name)
                all_evidence.append(evidence)
                para_relations.add(rel_name)

            n_types = len(para_relations)
            para_diversity.append(min(n_types / 4.0, 1.0))

            has_other = para_relations - {"additive", "implicit"}
            additive_only_flags.append(len(has_other) == 0)

        # Count additive-only zones: runs of 2+ consecutive additive-only paragraphs
        additive_only_zones = 0
        run_length = 0
        for flag in additive_only_flags:
            if flag:
                run_length += 1
            else:
                if run_length >= 2:
                    additive_only_zones += 1
                run_length = 0
        if run_length >= 2:
            additive_only_zones += 1

        counts = Counter(all_relation_labels)
        total = len(all_relation_labels) if all_relation_labels else 1
        entropy = _shannon_entropy(counts)

        return LensResult(
            lens_name=self.name,
            per_sentence={"relation_type": all_relation_types},
            per_paragraph={"relation_diversity": para_diversity},
            per_chapter={
                "relation_entropy": round(entropy, 4),
                "causal_ratio": round(counts.get("causal", 0) / total, 4),
                "contrastive_ratio": round(counts.get("contrastive", 0) / total, 4),
                "temporal_ratio": round(counts.get("temporal", 0) / total, 4),
                "additive_ratio": round(counts.get("additive", 0) / total, 4),
                "implicit_ratio": round(counts.get("implicit", 0) / total, 4),
                "additive_only_zones": additive_only_zones,
            },
            raw={
                "sentence_labels": all_relation_labels,
                "sentence_evidence": all_evidence,
            },
        )
