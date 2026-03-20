"""Dialogue voice separation: do characters sound different from each other?

Extracts dialogue lines, attributes them to speakers (heuristic), embeds
per-character dialogue, and measures whether characters have distinct
speech patterns or all sound like the same LLM.

Also measures dialogue/narration ratio and scene-level balance.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import norm

from prose_doctor.text import split_paragraphs


# Regex for extracting quoted speech
_DIALOGUE_RE = re.compile(
    r'["\u201c]([^"\u201d]{10,})["\u201d]',
)

# Attribution patterns: "said X", "X said", "X asked", etc.
_ATTRIBUTION_RE = re.compile(
    r'(?:'
    r'(?:said|asked|replied|answered|whispered|shouted|muttered|murmured|'
    r'called|cried|exclaimed|demanded|insisted|suggested|offered|added|'
    r'continued|began|started|finished|interrupted|snapped|growled|hissed|'
    r'sighed|groaned|laughed|screamed|yelled|bellowed|stammered|'
    r'declared|announced|observed|noted|remarked|commented|explained)\s+'
    r'([A-Z][a-z]+)'
    r'|'
    r'([A-Z][a-z]+)\s+'
    r'(?:said|asked|replied|answered|whispered|shouted|muttered|murmured|'
    r'called|cried|exclaimed|demanded|insisted|suggested|offered|added|'
    r'continued|began|started|finished|interrupted|snapped|growled|hissed|'
    r'sighed|groaned|laughed|screamed|yelled|bellowed|stammered|'
    r'declared|announced|observed|noted|remarked|commented|explained)'
    r')',
    re.IGNORECASE,
)


@dataclass
class DialogueLine:
    """A single line of dialogue with attribution."""
    text: str
    speaker: str | None
    paragraph_idx: int


@dataclass
class DialogueProfile:
    """Dialogue analysis for a chapter."""
    filename: str
    total_paragraphs: int
    dialogue_paragraphs: int
    dialogue_ratio: float
    speakers: dict[str, int]  # speaker -> line count
    # Voice separation metrics
    speaker_separation: float  # mean pairwise distance between speaker centroids (0-1)
    speaker_similarities: dict[str, float]  # pairwise "speakerA<->speakerB": similarity
    # Scene balance
    longest_dialogue_run: int  # consecutive paragraphs with dialogue
    longest_narration_run: int  # consecutive paragraphs without dialogue
    # Per-speaker vocab diversity
    speaker_vocab_diversity: dict[str, float]  # type-token ratio per speaker

    @property
    def all_same_voice(self) -> bool:
        """Are all speakers essentially indistinguishable?"""
        if not self.speaker_similarities:
            return False
        return self.speaker_separation < 0.15

    @property
    def talking_heads(self) -> bool:
        """Is there a long dialogue run with no narration?"""
        return self.longest_dialogue_run > 8

    @property
    def prescription(self) -> str:
        issues = []
        if self.all_same_voice:
            issues.append(
                "All characters sound identical in dialogue. Give each speaker "
                "a distinctive verbal tic, sentence length preference, or vocabulary "
                "register. One character uses short declaratives, another asks "
                "questions, another hedges."
            )
        if self.talking_heads:
            issues.append(
                f"You have {self.longest_dialogue_run} consecutive dialogue paragraphs "
                f"with no action, setting, or interiority. Break up the talking heads: "
                f"insert a gesture, a sensory detail, or an interior thought every "
                f"3-4 lines of dialogue."
            )
        return " ".join(issues)


def extract_dialogue(text: str) -> list[DialogueLine]:
    """Extract dialogue lines with speaker attribution."""
    paragraphs = split_paragraphs(text)
    lines = []

    for pi, para in enumerate(paragraphs):
        # Find all quoted speech in this paragraph
        matches = _DIALOGUE_RE.findall(para)
        if not matches:
            continue

        # Try to find speaker attribution
        attr = _ATTRIBUTION_RE.search(para)
        speaker = None
        if attr:
            speaker = attr.group(1) or attr.group(2)

        for match in matches:
            lines.append(DialogueLine(
                text=match.strip(),
                speaker=speaker,
                paragraph_idx=pi,
            ))

    return lines


def _compute_runs(text: str) -> tuple[int, int]:
    """Compute longest consecutive dialogue and narration runs."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return 0, 0

    max_dial, max_narr = 0, 0
    cur_dial, cur_narr = 0, 0

    for para in paragraphs:
        has_dialogue = bool(_DIALOGUE_RE.search(para))
        if has_dialogue:
            cur_dial += 1
            if cur_narr > max_narr:
                max_narr = cur_narr
            cur_narr = 0
        else:
            cur_narr += 1
            if cur_dial > max_dial:
                max_dial = cur_dial
            cur_dial = 0

    max_dial = max(max_dial, cur_dial)
    max_narr = max(max_narr, cur_narr)
    return max_dial, max_narr


def analyze_dialogue(
    text: str,
    filename: str,
    model_manager,
    min_lines_per_speaker: int = 3,
) -> DialogueProfile:
    """Analyze dialogue voice separation in a chapter."""
    paragraphs = split_paragraphs(text)
    lines = extract_dialogue(text)

    # Count dialogue vs narration paragraphs
    dial_paras = set()
    for line in lines:
        dial_paras.add(line.paragraph_idx)

    total = len(paragraphs)
    n_dial = len(dial_paras)
    ratio = n_dial / max(total, 1)

    # Group by speaker
    by_speaker: dict[str, list[str]] = {}
    for line in lines:
        if line.speaker:
            by_speaker.setdefault(line.speaker, []).append(line.text)

    # Filter speakers with too few lines
    by_speaker = {k: v for k, v in by_speaker.items() if len(v) >= min_lines_per_speaker}
    speaker_counts = {k: len(v) for k, v in by_speaker.items()}

    # Compute runs
    max_dial_run, max_narr_run = _compute_runs(text)

    # Voice separation via embeddings
    separation = 0.0
    similarities: dict[str, float] = {}
    vocab_diversity: dict[str, float] = {}

    if len(by_speaker) >= 2:
        st = model_manager.sentence_transformer

        # Embed each speaker's dialogue
        centroids = {}
        for speaker, texts in by_speaker.items():
            embs = st.encode(texts, show_progress_bar=False, batch_size=64)
            centroids[speaker] = np.mean(embs, axis=0)

            # Vocab diversity: type-token ratio
            all_words = " ".join(texts).lower().split()
            if all_words:
                vocab_diversity[speaker] = len(set(all_words)) / len(all_words)

        # Pairwise similarities
        speakers = list(centroids.keys())
        dists = []
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                a = centroids[speakers[i]]
                b = centroids[speakers[j]]
                sim = float(np.dot(a, b) / (norm(a) * norm(b)))
                key = f"{speakers[i]}<->{speakers[j]}"
                similarities[key] = round(sim, 4)
                dists.append(1 - sim)

        separation = float(np.mean(dists)) if dists else 0.0
    elif len(by_speaker) == 1:
        speaker = list(by_speaker.keys())[0]
        all_words = " ".join(by_speaker[speaker]).lower().split()
        if all_words:
            vocab_diversity[speaker] = len(set(all_words)) / len(all_words)

    return DialogueProfile(
        filename=filename,
        total_paragraphs=total,
        dialogue_paragraphs=n_dial,
        dialogue_ratio=round(ratio, 3),
        speakers=speaker_counts,
        speaker_separation=round(separation, 4),
        speaker_similarities=similarities,
        longest_dialogue_run=max_dial_run,
        longest_narration_run=max_narr_run,
        speaker_vocab_diversity=vocab_diversity,
    )
