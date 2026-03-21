"""Dialogue voice lens — do characters sound different from each other?

Extracts dialogue lines, attributes them to speakers (heuristic), embeds
per-character dialogue, and measures whether characters have distinct
speech patterns or all sound like the same LLM.

Also measures dialogue/narration ratio and talking-heads runs.

Ported from prose_doctor.ml.dialogue into the Lens interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import norm

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


# ---------- Dialogue extraction helpers ----------

_DIALOGUE_RE = re.compile(
    r'["\u201c]([^"\u201d]{10,})["\u201d]',
)

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


def _extract_dialogue(text: str) -> list[DialogueLine]:
    """Extract dialogue lines with speaker attribution."""
    paragraphs = split_paragraphs(text)
    lines = []

    for pi, para in enumerate(paragraphs):
        matches = _DIALOGUE_RE.findall(para)
        if not matches:
            continue

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


# ---------- Lens class ----------


class DialogueVoiceLens(Lens):
    """Analyze dialogue voice separation and talking-heads patterns."""

    name = "dialogue_voice"
    requires_providers: list[str] = ["spacy", "sentence_transformer"]
    consumes_lenses: list[str] = []

    MIN_LINES_PER_SPEAKER = 3
    TALKING_HEADS_THRESHOLD = 8

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        paragraphs = split_paragraphs(text)
        lines = _extract_dialogue(text)

        # Count dialogue vs narration paragraphs
        dial_paras = {line.paragraph_idx for line in lines}
        total = len(paragraphs)
        n_dial = len(dial_paras)
        ratio = n_dial / max(total, 1)

        # Group by speaker
        by_speaker: dict[str, list[str]] = {}
        for line in lines:
            if line.speaker:
                by_speaker.setdefault(line.speaker, []).append(line.text)

        # Filter speakers with too few lines
        by_speaker = {
            k: v for k, v in by_speaker.items()
            if len(v) >= self.MIN_LINES_PER_SPEAKER
        }
        speaker_counts = {k: len(v) for k, v in by_speaker.items()}

        # Compute runs
        max_dial_run, max_narr_run = _compute_runs(text)

        # Voice separation via embeddings
        separation = 0.0
        similarities: dict[str, float] = {}
        vocab_diversity: dict[str, float] = {}
        all_same_voice = False

        if len(by_speaker) >= 2:
            st = providers.sentence_transformer

            centroids = {}
            for speaker, texts in by_speaker.items():
                embs = st.encode(texts, show_progress_bar=False, batch_size=64)
                centroids[speaker] = np.mean(embs, axis=0)

                all_words = " ".join(texts).lower().split()
                if all_words:
                    vocab_diversity[speaker] = len(set(all_words)) / len(all_words)

            speakers_list = list(centroids.keys())
            dists = []
            for i in range(len(speakers_list)):
                for j in range(i + 1, len(speakers_list)):
                    a = centroids[speakers_list[i]]
                    b = centroids[speakers_list[j]]
                    sim = float(np.dot(a, b) / (norm(a) * norm(b)))
                    key = f"{speakers_list[i]}<->{speakers_list[j]}"
                    similarities[key] = round(sim, 4)
                    dists.append(1 - sim)

            separation = float(np.mean(dists)) if dists else 0.0
            all_same_voice = separation < 0.15
        elif len(by_speaker) == 1:
            speaker = list(by_speaker.keys())[0]
            all_words = " ".join(by_speaker[speaker]).lower().split()
            if all_words:
                vocab_diversity[speaker] = len(set(all_words)) / len(all_words)

        # Prescription
        issues = []
        if all_same_voice:
            issues.append(
                "All characters sound identical in dialogue. Give each speaker "
                "a distinctive verbal tic, sentence length preference, or vocabulary "
                "register."
            )
        if max_dial_run > self.TALKING_HEADS_THRESHOLD:
            issues.append(
                f"{max_dial_run} consecutive dialogue paragraphs with no action. "
                f"Break up talking heads with gestures, sensory details, or "
                f"interior thoughts every 3-4 lines."
            )
        prescription = " ".join(issues)

        per_chapter = {
            "speaker_separation": round(separation, 4),
            "dialogue_ratio": round(ratio, 3),
            "talking_heads_count": float(max_dial_run),
        }

        raw = {
            "speakers": speaker_counts,
            "speaker_similarities": similarities,
            "speaker_vocab_diversity": vocab_diversity,
            "longest_dialogue_run": max_dial_run,
            "longest_narration_run": max_narr_run,
            "all_same_voice": all_same_voice,
            "prescription": prescription,
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            raw=raw,
        )
