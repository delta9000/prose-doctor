"""Structural repetition detector: find repeated patterns across chapters.

Goes beyond vocabulary crutches (word frequency) to detect repeated
sentence architectures, emotional beats, scene transitions, and
paragraph-level templates that LLMs reuse unconsciously.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import norm


@dataclass
class RepetitionPattern:
    """A structural pattern that repeats too often."""
    pattern_type: str  # "opener", "transition", "beat", "architecture"
    template: str  # the repeated pattern
    count: int
    locations: list[tuple[str, int]]  # (filename, paragraph_idx)
    examples: list[str]  # actual text instances


@dataclass
class RepetitionReport:
    """Structural repetition analysis across chapters."""
    file_count: int
    paragraph_count: int
    patterns: list[RepetitionPattern]

    @property
    def prescription(self) -> str:
        if not self.patterns:
            return ""
        worst = self.patterns[0]
        return (
            f"'{worst.template}' appears {worst.count} times across "
            f"{self.file_count} chapters. Vary the {worst.pattern_type} — "
            f"each instance should feel like a choice, not a default."
        )


def _extract_openers(paragraphs: list[str]) -> list[str]:
    """Extract sentence openers (first 3-4 words)."""
    openers = []
    for para in paragraphs:
        # First sentence
        sent = re.split(r'[.!?]', para)[0].strip()
        words = sent.split()[:4]
        if len(words) >= 3:
            # Normalize: replace proper nouns with NOUN
            normalized = []
            for w in words:
                if w[0].isupper() and words.index(w) > 0:
                    normalized.append("NOUN")
                else:
                    normalized.append(w.lower())
            openers.append(" ".join(normalized))
    return openers


def _extract_transitions(paragraphs: list[str]) -> list[str]:
    """Extract paragraph transition patterns."""
    transitions = []
    for i in range(1, len(paragraphs)):
        prev = paragraphs[i - 1]
        curr = paragraphs[i]

        # Check for common LLM transition templates
        first_sent = re.split(r'[.!?]', curr)[0].strip().lower()

        # Time transitions
        if re.match(r'(later|after|when|by the time|hours|minutes|the next)', first_sent):
            transitions.append("TIME_TRANSITION")

        # Silence/pause
        if re.match(r'(silence|a (long|brief|heavy) (silence|pause|moment))', first_sent):
            transitions.append("SILENCE_BEAT")

        # Character looked/turned/glanced
        if re.match(r'\w+ (looked|turned|glanced|stared|gazed)', first_sent):
            transitions.append("NOUN_LOOKED")

        # "And then" / "And so"
        if re.match(r'and (then|so|yet)', first_sent):
            transitions.append("AND_THEN")

    return transitions


def _extract_emotional_beats(paragraphs: list[str]) -> list[str]:
    """Extract emotional beat patterns."""
    beats = []
    for para in paragraphs:
        lower = para.lower()

        # "Something in NOUN shifted/changed/broke"
        if re.search(r'something in (him|her|them|\w+) (shifted|changed|broke|stirred)', lower):
            beats.append("SOMETHING_SHIFTED")

        # "NOUN didn't answer/respond"
        if re.search(r"\w+ didn't (answer|respond|reply|speak|move)", lower):
            beats.append("DIDNT_RESPOND")

        # "Silence stretched/hung/fell"
        if re.search(r'(silence|quiet|stillness) (stretched|hung|fell|settled|descended)', lower):
            beats.append("SILENCE_STRETCHED")

        # "For a (long) moment"
        if re.search(r'for a (long |brief |)moment', lower):
            beats.append("FOR_A_MOMENT")

        # "NOUN let out a breath"
        if re.search(r'\w+ (let out|released|exhaled|blew out) a (breath|sigh)', lower):
            beats.append("LET_OUT_BREATH")

        # "The weight of" abstraction
        if re.search(r'the (weight|burden|gravity|heaviness) of', lower):
            beats.append("WEIGHT_OF")

        # "Eyes met" / "gazes locked"
        if re.search(r'(eyes met|gazes? (locked|held|met)|looked at each other)', lower):
            beats.append("EYES_MET")

    return beats


def _extract_architecture(paragraphs: list[str], nlp) -> list[str]:
    """Extract sentence architecture patterns using spaCy."""
    architectures = []
    for para in paragraphs:
        doc = nlp(para[:500])  # cap for speed
        sents = list(doc.sents)
        if not sents:
            continue

        # Check first sentence architecture
        sent = sents[0]
        tokens = [t for t in sent if not t.is_space]
        if len(tokens) < 4:
            continue

        # POS pattern for first 5 tokens
        pos_pattern = " ".join(t.pos_ for t in tokens[:5])
        architectures.append(pos_pattern)

    return architectures


def analyze_repetition(
    texts: dict[str, str],  # filename -> text
    model_manager=None,
    min_count: int = 3,
) -> RepetitionReport:
    """Analyze structural repetition across multiple chapters.

    Args:
        texts: dict mapping filename to chapter text
        model_manager: provides spaCy (optional, for architecture analysis)
        min_count: minimum occurrences to flag
    """
    from prose_doctor.text import split_paragraphs

    all_openers: list[tuple[str, str, int]] = []  # (pattern, filename, para_idx)
    all_transitions: list[tuple[str, str]] = []
    all_beats: list[tuple[str, str, int, str]] = []  # (pattern, filename, para_idx, text)
    total_paras = 0

    for filename, text in texts.items():
        paragraphs = split_paragraphs(text)
        total_paras += len(paragraphs)

        # Openers
        openers = _extract_openers(paragraphs)
        for i, op in enumerate(openers):
            all_openers.append((op, filename, i))

        # Transitions
        transitions = _extract_transitions(paragraphs)
        for t in transitions:
            all_transitions.append((t, filename))

        # Emotional beats
        beats = _extract_emotional_beats(paragraphs)
        for i, para in enumerate(paragraphs):
            for beat in _extract_emotional_beats([para]):
                all_beats.append((beat, filename, i, para[:100]))

    patterns: list[RepetitionPattern] = []

    # Opener repetition
    opener_counts = Counter(op for op, _, _ in all_openers)
    for template, count in opener_counts.most_common():
        if count >= min_count:
            locs = [(fn, idx) for op, fn, idx in all_openers if op == template]
            examples = [f"[{fn}:{idx}]" for fn, idx in locs[:3]]
            patterns.append(RepetitionPattern(
                pattern_type="opener",
                template=template,
                count=count,
                locations=locs,
                examples=examples,
            ))

    # Transition repetition
    trans_counts = Counter(t for t, _ in all_transitions)
    for template, count in trans_counts.most_common():
        if count >= min_count:
            locs = [(fn, 0) for t, fn in all_transitions if t == template]
            patterns.append(RepetitionPattern(
                pattern_type="transition",
                template=template,
                count=count,
                locations=locs,
                examples=[],
            ))

    # Beat repetition
    beat_counts = Counter(b for b, _, _, _ in all_beats)
    for template, count in beat_counts.most_common():
        if count >= min_count:
            locs = [(fn, idx) for b, fn, idx, _ in all_beats if b == template]
            examples = [txt for b, _, _, txt in all_beats if b == template][:3]
            patterns.append(RepetitionPattern(
                pattern_type="beat",
                template=template,
                count=count,
                locations=locs,
                examples=examples,
            ))

    # Sort by count descending
    patterns.sort(key=lambda p: -p.count)

    return RepetitionReport(
        file_count=len(texts),
        paragraph_count=total_paras,
        patterns=patterns[:20],  # top 20
    )
