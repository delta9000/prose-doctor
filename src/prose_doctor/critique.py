"""Generate a structured revision prompt from prose-doctor analysis.

Runs the full analyzer suite, compares against human prose baselines,
and outputs an actionable critique that can be pasted into a writing
LLM to guide revision.

Baselines derived from 50 chapter-sized excerpts:
  - Cory Doctorow, Little Brother (2008)
  - Peter Watts, Blindsight (2006)
  - Mary Shelley, Frankenstein (1818)
  - Joseph Conrad, Heart of Darkness (1899)
  - Charles Dickens, A Tale of Two Cities (1859)
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Human prose baselines (from 50-chapter analysis, March 2026)
# Each is (mean, target_direction) where direction is "higher" or "lower"
BASELINES = {
    "pd_mean":       (0.336, "higher",  "Psychic distance"),
    "pd_std":        (0.093, "higher",  "Distance variation"),
    "fg_inversion":  (44.2,  "higher",  "Sentence inversion"),
    "fg_sl_cv":      (0.706, "higher",  "Sentence length variation"),
    "fg_fragment":   (6.7,   "lower",   "Fragment ratio"),
    "ic_rhythmicity":(0.129, "lower",   "Information rhythmicity"),
    "ic_spikes":     (7.7,   "higher",  "Surprisal spikes"),
    # Flatlines: having fewer than baseline is fine, not a problem.
    # Only flag if significantly MORE than baseline.
    "ic_flatlines":  (3.1,   "lower",   "Information flatlines"),
}


@dataclass
class CritiqueSection:
    """One section of the critique."""
    dimension: str
    severity: str  # "major", "minor", "strength"
    value: float
    baseline: float
    direction: str
    prescription: str
    evidence: list[str] = field(default_factory=list)


def _format_delta(value: float, baseline: float, direction: str) -> str:
    """Format the gap between value and baseline."""
    if direction == "higher":
        if value >= baseline:
            return f"{value:.3f} (at or above human baseline {baseline:.3f})"
        gap = baseline - value
        return f"{value:.3f} (human baseline: {baseline:.3f}, gap: {gap:.3f})"
    else:
        if value <= baseline:
            return f"{value:.3f} (at or below human baseline {baseline:.3f})"
        gap = value - baseline
        return f"{value:.3f} (human baseline: {baseline:.3f}, excess: {gap:.3f})"


def _is_problem(value: float, baseline: float, direction: str, threshold: float = 0.15) -> bool:
    """Is this metric far enough from baseline to be a problem?"""
    if direction == "higher":
        return value < baseline * (1 - threshold)
    else:
        return value > baseline * (1 + threshold)


def _is_strength(value: float, baseline: float, direction: str) -> bool:
    """Is this metric better than the human baseline?"""
    if direction == "higher":
        return value > baseline * 1.05
    else:
        return value < baseline * 0.95


def build_critique(
    report: dict,
    twins: list | None = None,
) -> list[CritiqueSection]:
    """Build critique sections from a scan --deep report."""
    sections: list[CritiqueSection] = []

    pd = report.get("psychic_distance") or {}
    ic = report.get("info_contour") or {}
    fg = report.get("foregrounding") or {}
    sensory = report.get("sensory") or {}

    # Map report fields to baseline keys
    values = {
        "pd_mean":       pd.get("mean_distance", 0),
        "pd_std":        pd.get("std_distance", 0),
        "fg_inversion":  fg.get("inversion_pct", 0),
        "fg_sl_cv":      fg.get("sentence_length_cv", 0),
        "fg_fragment":   fg.get("fragment_pct", 0),
        "ic_rhythmicity":ic.get("rhythmicity", 0),
        "ic_spikes":     ic.get("spikes", 0) if isinstance(ic.get("spikes"), (int, float)) else len(ic.get("spikes", [])),
        "ic_flatlines":  ic.get("flatlines", 0) if isinstance(ic.get("flatlines"), (int, float)) else len(ic.get("flatlines", [])),
    }

    # Prescriptions keyed by metric
    prescriptions = {
        "pd_mean": (
            "You're narrating from outside the character. Pick the 3-5 most "
            "emotionally charged moments and rewrite them from deep inside the "
            "character's perception — what they feel in their body, what they "
            "notice, what they're afraid of. Use perception verbs (saw, felt, "
            "heard) and interior thought."
        ),
        "pd_std": (
            "Your narrative distance is flat — you stay at the same zoom level "
            "throughout. Pull back to establishing-shot distance for scene "
            "transitions, then push in close for confrontations and revelations. "
            "The contrast creates rhythm."
        ),
        "fg_inversion": (
            "Too many sentences follow subject-verb-object order. Restructure "
            "5-8 sentences to lead with a prepositional phrase, a verb, or a "
            "subordinate clause. 'Down the corridor she ran' instead of "
            "'She ran down the corridor.'"
        ),
        "fg_sl_cv": (
            "Your sentences are too uniform in length. Break 3-4 long sentences "
            "into staccato fragments at high-tension moments. Merge 3-4 short "
            "sentences into complex, flowing ones during reflective passages. "
            "The variation creates reading rhythm."
        ),
        "fg_fragment": (
            "You're overusing short fragments for manufactured emphasis. "
            "Each fragment should earn its weight — save them for genuine "
            "moments of impact. Merge the weaker ones back into full sentences."
        ),
        "ic_rhythmicity": (
            "Your information density is too metronomic — every paragraph "
            "delivers roughly the same amount of new information. Vary the "
            "density: follow a dense, information-packed paragraph with a "
            "sparse, atmospheric one. Let the reader breathe."
        ),
        "ic_spikes": (
            "Your prose is too predictable — not enough moments where the "
            "language itself surprises. At 3-4 key moments, choose an "
            "unexpected word, an unusual metaphor, or a syntactic structure "
            "the reader doesn't see coming."
        ),
        "ic_flatlines": (
            "Your prose has long stretches of uniform information density — "
            "the reader zones out. Break these up by inserting a moment of "
            "high tension, a surprising detail, or a shift in narrative mode."
        ),
    }

    for key, (baseline, direction, label) in BASELINES.items():
        value = values.get(key, 0)
        if _is_problem(value, baseline, direction):
            severity = "major" if _is_problem(value, baseline, direction, 0.25) else "minor"
            sections.append(CritiqueSection(
                dimension=label,
                severity=severity,
                value=value,
                baseline=baseline,
                direction=direction,
                prescription=prescriptions.get(key, ""),
            ))
        elif _is_strength(value, baseline, direction):
            sections.append(CritiqueSection(
                dimension=label,
                severity="strength",
                value=value,
                baseline=baseline,
                direction=direction,
                prescription="",
            ))

    # Foregrounding weakest axis prescription
    if fg.get("weakest_axis") and fg.get("prescription"):
        sections.append(CritiqueSection(
            dimension=f"Texture: {fg['weakest_axis']}",
            severity="minor",
            value=0,
            baseline=0,
            direction="higher",
            prescription=fg["prescription"],
        ))

    # Sensory prescription if available
    if sensory.get("prescription") and sensory.get("weakest"):
        sections.append(CritiqueSection(
            dimension=f"Sensory: {sensory['weakest']}",
            severity="minor",
            value=0,
            baseline=0,
            direction="higher",
            prescription=sensory["prescription"],
        ))

    # Dialogue voice separation
    dialogue = report.get("dialogue") or {}
    if dialogue.get("all_same_voice"):
        sections.append(CritiqueSection(
            dimension="Dialogue voice",
            severity="major",
            value=dialogue.get("speaker_separation", 0),
            baseline=0.15,
            direction="higher",
            prescription=dialogue.get("prescription", (
                "All characters sound identical in dialogue. Give each speaker "
                "a distinctive verbal tic, sentence length preference, or "
                "vocabulary register."
            )),
        ))
    if dialogue.get("talking_heads"):
        n = len(dialogue["talking_heads"]) if isinstance(dialogue["talking_heads"], list) else dialogue["talking_heads"]
        if n > 0:
            sections.append(CritiqueSection(
                dimension="Talking heads",
                severity="minor",
                value=float(n),
                baseline=0,
                direction="lower",
                prescription=(
                    f"{n} stretch{'es' if n > 1 else ''} of unbroken dialogue. "
                    f"Insert action, setting, or interior thought every 3-4 "
                    f"lines to ground the conversation in physical space."
                ),
            ))

    # Scene pacing
    pacing = report.get("pacing") or {}
    pacing_rx = pacing.get("prescription", "")
    if pacing_rx:
        sections.append(CritiqueSection(
            dimension="Scene pacing",
            severity="minor",
            value=0,
            baseline=0,
            direction="higher",
            prescription=pacing_rx,
        ))

    # Twin-based self-referential feedback
    if twins:
        for tw in twins[:3]:
            sections.append(CritiqueSection(
                dimension="Self-reference",
                severity="minor",
                value=tw.get("flat_texture", 0),
                baseline=tw.get("twin_texture", 0),
                direction="higher",
                prescription=(
                    f"Your flattest paragraph (#{tw.get('flat_idx', '?')}) is topically "
                    f"similar to one of your best (#{tw.get('twin_idx', '?')}). "
                    f"You wrote it well there — match that texture here."
                ),
                evidence=[
                    f"FLAT: {tw.get('flat_text', '')[:150]}",
                    f"TWIN: {tw.get('twin_text', '')[:150]}",
                ],
            ))

    # Sort: major issues first, then minor, then strengths
    order = {"major": 0, "minor": 1, "strength": 2}
    sections.sort(key=lambda s: order.get(s.severity, 1))

    return sections


def format_critique_prompt(
    filename: str,
    sections: list[CritiqueSection],
    word_count: int = 0,
) -> str:
    """Format critique sections into a revision prompt for an LLM."""
    lines = []
    lines.append("# Revision Guide")
    lines.append("")
    lines.append(f"Chapter: {filename} ({word_count:,} words)")
    lines.append("")

    majors = [s for s in sections if s.severity == "major"]
    minors = [s for s in sections if s.severity == "minor"]
    strengths = [s for s in sections if s.severity == "strength"]

    if majors:
        lines.append("## Priority Issues")
        lines.append("")
        for s in majors:
            lines.append(f"**{s.dimension}** — {_format_delta(s.value, s.baseline, s.direction)}")
            lines.append(f"  {s.prescription}")
            for e in s.evidence:
                lines.append(f"  > {e}")
            lines.append("")

    if minors:
        lines.append("## Improvements")
        lines.append("")
        for s in minors:
            if s.prescription:
                lines.append(f"**{s.dimension}**")
                lines.append(f"  {s.prescription}")
                for e in s.evidence:
                    lines.append(f"  > {e}")
                lines.append("")

    if strengths:
        lines.append("## Strengths (preserve these)")
        lines.append("")
        for s in strengths:
            lines.append(f"- **{s.dimension}**: {_format_delta(s.value, s.baseline, s.direction)}")
        lines.append("")

    lines.append("## Instructions")
    lines.append("")
    lines.append("Rewrite this chapter addressing the priority issues above.")
    lines.append("Preserve the plot, characters, and dialogue content.")
    lines.append("Do not add new scenes or characters.")
    lines.append("Focus on prose technique, not story changes.")

    return "\n".join(lines)
