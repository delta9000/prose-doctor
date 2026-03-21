"""Pacing lens — scene-level dialogue/action/interiority/setting balance.

Classifies each paragraph by its dominant mode and measures the
distribution and transitions. Flags talking-head scenes (dialogue
with no grounding), action deserts, and interiority gaps.

Ported from prose_doctor.ml.pacing into the Lens interface.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

# ---------- Heuristic paragraph mode classification ----------

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

# Float codes for per_paragraph mode array
MODE_CODE = {"dialogue": 0.0, "action": 1.0, "interiority": 2.0, "setting": 3.0}


def _classify_paragraph(para: str) -> str:
    """Classify a paragraph as dialogue, action, interiority, or setting."""
    words = para.lower().split()
    word_set = set(words)

    has_dialogue = bool(_DIALOGUE_RE.search(para))
    action_count = len(word_set & _ACTION_VERBS)
    interior_count = len(word_set & _INTERIORITY_MARKERS)

    # Priority: dialogue > interiority > action > setting
    if has_dialogue:
        return "dialogue"
    if interior_count >= 2:
        return "interiority"
    if action_count >= 2:
        return "action"
    return "setting"


class PacingLens(Lens):
    """Analyze scene-level pacing by paragraph mode."""

    name = "pacing"
    requires_providers: list[str] = []
    consumes_lenses: list[str] = []

    def __init__(
        self,
        talking_head_threshold: int = 6,
        desert_threshold: int = 15,
        interiority_gap_threshold: int = 12,
    ) -> None:
        self.talking_head_threshold = talking_head_threshold
        self.desert_threshold = desert_threshold
        self.interiority_gap_threshold = interiority_gap_threshold

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return LensResult(
                lens_name=self.name,
                per_chapter={"dialogue_ratio": 0.0, "action_ratio": 0.0,
                             "interiority_ratio": 0.0, "setting_ratio": 0.0},
                per_paragraph={"mode": []},
                raw={},
            )

        modes = [_classify_paragraph(p) for p in paragraphs]
        n = len(modes)

        # --- Per-chapter: mode ratios ---
        counts = Counter(modes)
        per_chapter = {
            f"{mode}_ratio": round(counts.get(mode, 0) / n, 3)
            for mode in ("dialogue", "action", "interiority", "setting")
        }

        # --- Per-paragraph: mode codes ---
        per_paragraph = {"mode": [MODE_CODE.get(m, 3.0) for m in modes]}

        # --- Transition counts ---
        transitions = Counter()
        for i in range(n - 1):
            transitions[f"{modes[i]}->{modes[i + 1]}"] += 1

        # --- Longest runs per mode ---
        longest: dict[str, int] = {}
        for mode in set(modes):
            max_run = cur_run = 0
            for m in modes:
                if m == mode:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 0
            longest[mode] = max_run

        # --- Talking heads: consecutive dialogue paragraphs ---
        talking_heads = _detect_runs(modes, "dialogue", self.talking_head_threshold)

        # --- Action deserts: long stretches without action ---
        action_deserts = _detect_gaps(modes, "action", self.desert_threshold)

        # --- Interiority gaps ---
        interiority_gaps = _detect_gaps(modes, "interiority", self.interiority_gap_threshold)

        raw = {
            "total_paragraphs": n,
            "mode_counts": dict(counts),
            "paragraph_modes": modes,
            "mode_transitions": dict(transitions),
            "longest_runs": longest,
            "talking_heads": talking_heads,
            "action_deserts": action_deserts,
            "interiority_gaps": interiority_gaps,
        }

        return LensResult(
            lens_name=self.name,
            per_chapter=per_chapter,
            per_paragraph=per_paragraph,
            raw=raw,
        )


def _detect_runs(modes: list[str], target: str, threshold: int) -> list[dict]:
    """Detect consecutive runs of *target* mode >= threshold length."""
    results = []
    run_start = None
    n = len(modes)
    for i, m in enumerate(modes):
        if m == target:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) >= threshold:
                results.append({"start": run_start, "end": i - 1, "length": i - run_start})
            run_start = None
    if run_start is not None and (n - run_start) >= threshold:
        results.append({"start": run_start, "end": n - 1, "length": n - run_start})
    return results


def _detect_gaps(modes: list[str], target: str, threshold: int) -> list[dict]:
    """Detect stretches where *target* mode is absent for >= threshold paragraphs."""
    results = []
    since = 0
    gap_start = 0
    n = len(modes)
    for i, m in enumerate(modes):
        if m == target:
            if since >= threshold:
                results.append({"start": gap_start, "end": i - 1, "length": since})
            since = 0
            gap_start = i + 1
        else:
            since += 1
    if since >= threshold:
        results.append({"start": gap_start, "end": n - 1, "length": since})
    return results
