"""Situation shifts lens -- time/space/actor transition detection.

Tracks shifts in time, space, and actors at paragraph boundaries using
spaCy NER, dependency parsing, and word-list heuristics. Based on the
event-indexing model (Zwaan, Langston & Graesser, 1995).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

TEMPORAL_MARKERS = {
    "later", "earlier", "before", "after", "ago", "next", "previous",
    "yesterday", "tomorrow", "morning", "evening", "night", "dawn",
    "dusk", "then", "meanwhile", "hours", "days", "weeks", "months",
    "years",
}

MOTION_VERBS = {
    "went", "drove", "walked", "ran", "flew", "crossed", "entered",
    "left", "arrived", "returned", "moved",
}

TIME_JUMP_RE = re.compile(
    r"\b(?:(?:three|two|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:days?|weeks?|months?|years?|hours?|minutes?)\s+"
    r"(?:later|earlier|before|after|ago))"
    r"|(?:the\s+(?:next|previous|following)\s+(?:morning|evening|night|day|week))"
    r"|(?:(?:a|one)\s+(?:week|month|year|day)\s+(?:later|earlier|ago))",
    re.IGNORECASE,
)

LOCATION_PREP_RE = re.compile(
    r"\b(?:in|at|inside|outside|behind|beneath|above)\s+(?:the\s+)?(\w+)",
    re.IGNORECASE,
)


def _get_dominant_tense(doc) -> str:
    """Return 'past' or 'present' based on majority verb tense."""
    past = 0
    present = 0
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            morph = token.morph.get("Tense")
            if morph:
                if "Past" in morph:
                    past += 1
                elif "Pres" in morph:
                    present += 1
    return "past" if past >= present else "present"


def _get_locations(doc) -> set[str]:
    """Extract location-like entities and prepositional place nouns."""
    locs = set()
    for ent in doc.ents:
        if ent.label_ in ("LOC", "GPE", "FAC"):
            locs.add(ent.text.lower())
    for match in LOCATION_PREP_RE.finditer(doc.text):
        locs.add(match.group(1).lower())
    return locs


def _get_first_subject(doc) -> str | None:
    """Get the text of the first nsubj in the doc."""
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "nsubj":
                # Return the head noun of the subject
                if token.pos_ == "PRON":
                    return token.text.lower()
                return token.lemma_.lower()
        break  # only first sentence
    return None


def _has_temporal_markers(text: str) -> bool:
    lower = text.lower()
    for marker in TEMPORAL_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", lower):
            return True
    return bool(TIME_JUMP_RE.search(text))


def _has_motion_verbs(doc) -> bool:
    for token in doc:
        if token.lemma_.lower() in MOTION_VERBS or token.text.lower() in MOTION_VERBS:
            return True
    return False


class SituationShiftsLens(Lens):
    """Detect time, space, and actor shifts at paragraph boundaries."""

    name = "situation_shifts"
    requires_providers = ["spacy"]
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict | None = None,
    ) -> LensResult:
        nlp = providers.spacy
        paragraphs = split_paragraphs(text)

        time_shifts: list[float] = []
        space_shifts: list[float] = []
        actor_shifts: list[float] = []
        shift_details: list[dict] = []

        prev_tense = None
        prev_locs: set[str] = set()
        prev_subject: str | None = None

        for i, para in enumerate(paragraphs):
            doc = nlp(para)

            cur_tense = _get_dominant_tense(doc)
            cur_locs = _get_locations(doc)
            cur_subject = _get_first_subject(doc)

            if i == 0:
                time_shifts.append(0.0)
                space_shifts.append(0.0)
                actor_shifts.append(0.0)
                shift_details.append({})
            else:
                detail: dict = {}

                # Time shift
                t_shift = 0.0
                if _has_temporal_markers(para):
                    t_shift = 1.0
                    detail["time_evidence"] = "temporal markers"
                elif cur_tense != prev_tense:
                    t_shift = 1.0
                    detail["time_evidence"] = f"tense change: {prev_tense}->{cur_tense}"
                time_shifts.append(t_shift)

                # Space shift
                s_shift = 0.0
                if prev_locs and cur_locs and not prev_locs & cur_locs:
                    s_shift = 1.0
                    detail["space_evidence"] = f"new locations: {cur_locs - prev_locs}"
                elif _has_motion_verbs(doc) and cur_locs - prev_locs:
                    s_shift = 1.0
                    detail["space_evidence"] = "motion verb + new location"
                space_shifts.append(s_shift)

                # Actor shift
                a_shift = 0.0
                if (
                    cur_subject is not None
                    and prev_subject is not None
                    and cur_subject != prev_subject
                    # Don't flag pronoun->name if they could be the same person
                    and not (cur_subject in ("he", "she", "they", "it") or prev_subject in ("he", "she", "they", "it"))
                ):
                    a_shift = 1.0
                    detail["actor_evidence"] = f"subject change: {prev_subject}->{cur_subject}"
                actor_shifts.append(a_shift)

                shift_details.append(detail)

            prev_tense = cur_tense
            prev_locs = cur_locs
            prev_subject = cur_subject

        total_time = int(sum(time_shifts))
        total_space = int(sum(space_shifts))
        total_actor = int(sum(actor_shifts))
        total_shifts = total_time + total_space + total_actor

        # Disorientation: shifts without grounding markers (e.g., time shift
        # without explicit temporal phrase — just tense change)
        ungrounded = sum(
            1 for d in shift_details
            if d and any("tense change" in str(v) for v in d.values())
        )

        n_boundaries = max(len(paragraphs) - 1, 1)
        disorientation = round(ungrounded / n_boundaries, 3)

        return LensResult(
            lens_name="situation_shifts",
            per_paragraph={
                "time_shift": time_shifts,
                "space_shift": space_shifts,
                "actor_shift": actor_shifts,
            },
            per_chapter={
                "total_shifts": total_shifts,
                "time_shifts": total_time,
                "space_shifts": total_space,
                "actor_shifts": total_actor,
                "disorientation_score": disorientation,
            },
            raw={"shift_details": shift_details},
        )
