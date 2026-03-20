"""Scene pacing analyzer: dialogue/action/interiority/setting balance.

Classifies each paragraph by its dominant mode and measures the
distribution and transitions. Flags talking-head scenes (dialogue
with no grounding), action deserts, and interiority gaps.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import numpy as np

from prose_doctor.text import split_paragraphs

# Heuristic paragraph mode classification
_DIALOGUE_RE = re.compile(r'["\u201c][^"\u201d]{10,}["\u201d]')
_ACTION_VERBS = frozenset({
    "ran", "walked", "grabbed", "pulled", "pushed", "threw", "hit",
    "jumped", "climbed", "fell", "opened", "closed", "turned", "moved",
    "reached", "picked", "dropped", "kicked", "punched", "ducked",
    "slammed", "ripped", "cut", "shot", "fired", "swung", "caught",
    "lunged", "stumbled", "sprinted", "crawled", "lifted", "carried",
})
_INTERIORITY_MARKERS = frozenset({
    "thought", "wondered", "realized", "knew", "felt", "remembered",
    "imagined", "believed", "hoped", "feared", "wished", "suspected",
    "considered", "supposed", "understood", "decided", "noticed",
})


def _classify_paragraph(para: str) -> str:
    """Classify a paragraph as dialogue, action, interiority, or setting."""
    words = para.lower().split()
    word_set = set(words)

    has_dialogue = bool(_DIALOGUE_RE.search(para))
    action_count = len(word_set & _ACTION_VERBS)
    interior_count = len(word_set & _INTERIORITY_MARKERS)

    # Priority: dialogue > interiority > action > setting
    if has_dialogue and len(para) < 200:
        return "dialogue"
    if has_dialogue:
        return "dialogue"  # mixed, but dialogue-dominant
    if interior_count >= 2:
        return "interiority"
    if action_count >= 2:
        return "action"
    return "setting"


@dataclass
class PacingProfile:
    """Scene-level pacing analysis."""
    filename: str
    total_paragraphs: int
    mode_counts: dict[str, int]
    mode_ratios: dict[str, float]
    paragraph_modes: list[str]
    # Transition analysis
    mode_transitions: dict[str, int]  # "dialogue->action": count
    # Run analysis
    longest_runs: dict[str, int]  # mode -> longest consecutive run
    # Flags
    talking_heads: list[dict]  # stretches of pure dialogue
    action_deserts: list[dict]  # long stretches with no action
    interiority_gaps: list[dict]  # long stretches with no interiority

    @property
    def dominant_mode(self) -> str:
        return max(self.mode_ratios, key=self.mode_ratios.get) if self.mode_ratios else "unknown"

    @property
    def prescription(self) -> str:
        issues = []
        if self.talking_heads:
            n = len(self.talking_heads)
            issues.append(
                f"{n} talking-head stretch{'es' if n > 1 else ''} — dialogue with "
                f"no action or setting. Insert a gesture, sensory detail, or "
                f"interior thought every 3-4 lines."
            )
        if self.action_deserts:
            n = len(self.action_deserts)
            issues.append(
                f"{n} action desert{'s' if n > 1 else ''} — long stretch without "
                f"physical action. Add movement, gesture, or interaction with "
                f"the environment."
            )
        if self.interiority_gaps:
            n = len(self.interiority_gaps)
            issues.append(
                f"{n} interiority gap{'s' if n > 1 else ''} — the character's "
                f"inner life goes silent. Add a thought, memory, or emotional "
                f"response."
            )
        return " ".join(issues)


def analyze_pacing(
    text: str,
    filename: str,
    talking_head_threshold: int = 6,
    desert_threshold: int = 15,
    interiority_gap_threshold: int = 12,
) -> PacingProfile:
    """Analyze scene-level pacing by paragraph mode."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return PacingProfile(
            filename=filename, total_paragraphs=0,
            mode_counts={}, mode_ratios={}, paragraph_modes=[],
            mode_transitions={}, longest_runs={},
            talking_heads=[], action_deserts=[], interiority_gaps=[],
        )

    # Classify each paragraph
    modes = [_classify_paragraph(p) for p in paragraphs]
    n = len(modes)

    # Count modes
    from collections import Counter
    counts = Counter(modes)
    ratios = {k: v / n for k, v in counts.items()}

    # Transition counts
    transitions = Counter()
    for i in range(len(modes) - 1):
        transitions[f"{modes[i]}->{modes[i+1]}"] += 1

    # Longest runs per mode
    longest = {}
    for mode in set(modes):
        max_run, cur_run = 0, 0
        for m in modes:
            if m == mode:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        longest[mode] = max_run

    # Detect talking heads: consecutive dialogue paragraphs
    talking_heads = []
    run_start = None
    for i, m in enumerate(modes):
        if m == "dialogue":
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= talking_head_threshold:
                talking_heads.append({"start": run_start, "end": i - 1, "length": i - run_start})
            run_start = None
    if run_start is not None and (n - run_start) >= talking_head_threshold:
        talking_heads.append({"start": run_start, "end": n - 1, "length": n - run_start})

    # Detect action deserts: long stretches without action
    action_deserts = []
    since_action = 0
    desert_start = 0
    for i, m in enumerate(modes):
        if m == "action":
            if since_action >= desert_threshold:
                action_deserts.append({"start": desert_start, "end": i - 1, "length": since_action})
            since_action = 0
            desert_start = i + 1
        else:
            since_action += 1
    if since_action >= desert_threshold:
        action_deserts.append({"start": desert_start, "end": n - 1, "length": since_action})

    # Detect interiority gaps
    interiority_gaps = []
    since_interior = 0
    gap_start = 0
    for i, m in enumerate(modes):
        if m == "interiority":
            if since_interior >= interiority_gap_threshold:
                interiority_gaps.append({"start": gap_start, "end": i - 1, "length": since_interior})
            since_interior = 0
            gap_start = i + 1
        else:
            since_interior += 1
    if since_interior >= interiority_gap_threshold:
        interiority_gaps.append({"start": gap_start, "end": n - 1, "length": since_interior})

    return PacingProfile(
        filename=filename,
        total_paragraphs=n,
        mode_counts=dict(counts),
        mode_ratios={k: round(v, 3) for k, v in ratios.items()},
        paragraph_modes=modes,
        mode_transitions=dict(transitions),
        longest_runs=longest,
        talking_heads=talking_heads,
        action_deserts=action_deserts,
        interiority_gaps=interiority_gaps,
    )
