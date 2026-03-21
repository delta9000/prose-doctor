"""Fragment classifier lens -- craft vs crutch short-sentence detection.

Classifies short sentences (< 5 spaCy tokens) as either deliberate
literary craft or LLM-style filler crutches.

Craft signals: dialogue, em-dash constructions, 3+ fragment runs,
anaphoric echo, concrete sensory detail in a pair.

Crutch signals: vague abstractions, isolated fragments without
concrete detail.

Extracted from agent_issues.py (find_fragment_issues, _is_vague_fragment,
_has_concrete_detail).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


# ---------- Heuristic helpers ----------

# Vague/abstract content patterns -- almost always crutches
_VAGUE_FRAGMENTS = {
    "the horizon shimmered", "the air changed", "the world shifted",
    "the ground trembled", "the light faded", "the darkness deepened",
    "the silence stretched", "the sound stopped", "the weight settled",
    "the distance grew", "something changed", "everything changed",
    "nothing moved", "time passed", "time stopped",
}

_BODY_PARTS = frozenset({
    "teeth", "jaw", "throat", "chest", "ribs", "spine", "skull",
    "fingers", "knuckles", "wrist", "palm", "hand", "hands",
    "stomach", "gut", "lungs", "knees", "ankle", "bone", "bones",
    "skin", "muscle", "nerve", "ear", "ears", "eye", "eyes",
    "shoulder", "shoulders", "neck", "elbow", "hip", "tongue",
})


def _has_concrete_detail(text: str) -> bool:
    """Does this fragment contain concrete sensory/physical detail?"""
    words = set(text.lower().rstrip(".!?").split())

    # Body parts = concrete
    if words & _BODY_PARTS:
        return True

    # Named character (capitalized word that isn't sentence-start)
    split = text.split()
    if len(split) >= 2 and any(w[0].isupper() and w.isalpha() for w in split[1:]):
        return True

    # Possessive proper noun at start ("Fen's voice") = specific
    if split and "'" in split[0] and split[0][0].isupper():
        return True

    return False


def _is_vague_fragment(text: str) -> bool:
    """Is this fragment generic/abstract with no concrete detail?"""
    normalized = text.lower().rstrip(".!?").strip()
    if normalized in _VAGUE_FRAGMENTS:
        return True

    # "The [abstract noun] [verb]ed." pattern
    words = normalized.split()
    if len(words) <= 3 and words[0] in ("the", "a", "an", "it"):
        return True

    return False


# ---------- Lens ----------


class FragmentClassifierLens(Lens):
    """Classify short sentences as craft or crutch fragments."""

    name = "fragment_classifier"
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

        craft_count = 0
        crutch_count = 0
        fragment_ratios: list[float] = []
        raw_fragments: list[dict] = []

        for pi, para in enumerate(paragraphs):
            doc = nlp(para)
            sents = list(doc.sents)
            sent_lengths = [len(s) for s in sents]
            total_sents = len(sents)
            para_fragments = 0

            for si, sent in enumerate(sents):
                if len(sent) >= 5:
                    continue

                sent_text = sent.text.strip()
                if not sent_text or len(sent_text) < 3:
                    continue

                para_fragments += 1

                # --- Hard preserve: dialogue and em-dash ---
                if any(c in sent_text for c in ('"', '\u201c', '\u201d')):
                    craft_count += 1
                    raw_fragments.append({
                        "paragraph": pi, "text": sent_text,
                        "classification": "craft", "reason": "dialogue fragment",
                    })
                    continue

                if '\u2014' in sent_text or '--' in sent_text:
                    craft_count += 1
                    raw_fragments.append({
                        "paragraph": pi, "text": sent_text,
                        "classification": "craft", "reason": "em-dash continuation",
                    })
                    continue

                # --- Fragment run analysis ---
                run_start = si
                while run_start > 0 and sent_lengths[run_start - 1] < 5:
                    run_start -= 1
                run_end = si
                while run_end < len(sents) - 1 and sent_lengths[run_end + 1] < 5:
                    run_end += 1
                run_length = run_end - run_start + 1

                # 3+ sequence: craft (montage/list)
                if run_length >= 3:
                    craft_count += 1
                    raw_fragments.append({
                        "paragraph": pi, "text": sent_text,
                        "classification": "craft",
                        "reason": f"part of {run_length}-fragment sequence",
                    })
                    continue

                # Anaphoric echo
                if run_length >= 2 and si > 0 and sent_lengths[si - 1] < 5:
                    prev_first = sents[si - 1].text.strip().split()[0].lower()
                    curr_first = sent_text.split()[0].lower()
                    if prev_first == curr_first:
                        craft_count += 1
                        raw_fragments.append({
                            "paragraph": pi, "text": sent_text,
                            "classification": "craft",
                            "reason": f"anaphoric echo ('{curr_first}...')",
                        })
                        continue

                # --- Content-based classification ---
                is_concrete = _has_concrete_detail(sent_text)
                is_vague = _is_vague_fragment(sent_text)

                if run_length == 2 and is_concrete:
                    craft_count += 1
                    raw_fragments.append({
                        "paragraph": pi, "text": sent_text,
                        "classification": "craft",
                        "reason": "fragment pair with concrete detail",
                    })
                elif is_vague:
                    crutch_count += 1
                    raw_fragments.append({
                        "paragraph": pi, "text": sent_text,
                        "classification": "crutch",
                        "reason": "vague/generic fragment",
                    })
                elif is_concrete:
                    crutch_count += 1
                    raw_fragments.append({
                        "paragraph": pi, "text": sent_text,
                        "classification": "crutch",
                        "reason": "isolated fragment with concrete detail — check if earned",
                    })
                else:
                    crutch_count += 1
                    raw_fragments.append({
                        "paragraph": pi, "text": sent_text,
                        "classification": "crutch",
                        "reason": "isolated fragment",
                    })

            ratio = para_fragments / max(total_sents, 1)
            fragment_ratios.append(ratio)

        total_fragments = craft_count + crutch_count
        total_sents_all = sum(
            len(list(nlp(p).sents)) for p in paragraphs
        ) if paragraphs else 1

        return LensResult(
            lens_name=self.name,
            per_paragraph={"fragment_ratio": fragment_ratios},
            per_chapter={
                "craft_count": float(craft_count),
                "crutch_count": float(crutch_count),
                "total_fragments": float(total_fragments),
                "fragment_pct": total_fragments / max(total_sents_all, 1) * 100,
            },
            raw={"fragments": raw_fragments},
        )
