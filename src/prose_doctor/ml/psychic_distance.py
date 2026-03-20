"""Psychic distance tracker: narrative zoom level per sentence.

Based on John Gardner's psychic distance scale — a continuous gradient from
distant, objective narration to deep interiority. Measures pronoun type,
perception verbs, deictic markers, sensory concreteness (via distilled
sensory probe), and tense to produce a per-sentence distance score.

Flags "zoom jumps" — unintentional 2+ level shifts within a short window
without a scene break.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from prose_doctor.text import split_paragraphs_with_breaks

# Perception/cognition verbs that pull the reader into a character's head
PERCEPTION_VERBS = frozenset({
    "saw", "see", "sees", "seeing", "seen",
    "heard", "hear", "hears", "hearing",
    "felt", "feel", "feels", "feeling",
    "smelled", "smell", "smells", "smelling",
    "tasted", "taste", "tastes", "tasting",
    "touched", "touch", "touches", "touching",
    "noticed", "notice", "notices", "noticing",
    "watched", "watch", "watches", "watching",
    "sensed", "sense", "senses", "sensing",
    "glimpsed", "glimpse", "glimpses",
})

COGNITION_VERBS = frozenset({
    "thought", "think", "thinks", "thinking",
    "knew", "know", "knows", "knowing",
    "realized", "realize", "realizes", "realizing",
    "understood", "understand", "understands",
    "wondered", "wonder", "wonders", "wondering",
    "believed", "believe", "believes", "believing",
    "supposed", "suppose", "supposes",
    "imagined", "imagine", "imagines", "imagining",
    "remembered", "remember", "remembers",
    "forgot", "forget", "forgets",
    "decided", "decide", "decides",
    "hoped", "hope", "hopes", "hoping",
    "feared", "fear", "fears", "fearing",
    "wished", "wish", "wishes", "wishing",
    "doubted", "doubt", "doubts",
    "suspected", "suspect", "suspects",
    "considered", "consider", "considers",
})

# Proximal deictics = close to character's here-and-now
PROXIMAL_DEICTICS = frozenset({
    "this", "these", "here", "now", "just", "come", "coming",
})

# Distal deictics = distant, narrator perspective
DISTAL_DEICTICS = frozenset({
    "that", "those", "there", "then", "once", "ago", "go", "going", "went",
})


_SENSORY_PROBE = None


def _get_sensory_probe():
    """Lazy-load the sensory modality probe."""
    global _SENSORY_PROBE
    if _SENSORY_PROBE is None:
        from prose_doctor.ml.sensory import SensoryProbe
        _SENSORY_PROBE = SensoryProbe()
    return _SENSORY_PROBE


def _sensory_concreteness(words: list[str]) -> float:
    """Score a list of words on sensory concreteness (0-1).

    Uses the distilled sensory probe — max modality score per word, averaged.
    """
    if not words:
        return 0.3
    probe = _get_sensory_probe()
    try:
        scores = probe.score_words(words)  # (N, 6)
        # Max across modalities per word, then mean
        return float(scores.max(axis=1).mean())
    except Exception:
        return 0.3


def score_sentence(sent, _unused=None) -> float:
    """Score a spaCy sentence span on psychic distance (0=distant, 1=deep interior).

    Features (all 0-1, weighted):
      - pronoun_score:    first person > third person > proper noun > no person
      - perception_score: density of perception/cognition verbs
      - deictic_score:    proximal vs distal deictic ratio
      - sensory_score:    sensory concreteness via distilled probe
      - tense_score:      present > past > past perfect
      - interiority:      exclamations, questions in narration, italics markers
    """
    tokens = [t for t in sent if not t.is_space]
    if len(tokens) < 3:
        return 0.5  # too short to judge

    words = [t.text.lower() for t in tokens]
    n = len(tokens)

    # --- Pronoun / person score ---
    first_person = sum(1 for w in words if w in ("i", "me", "my", "mine", "myself",
                                                   "we", "us", "our", "ours"))
    third_person = sum(1 for t in tokens if t.pos_ == "PRON" and
                       t.text.lower() not in ("i", "me", "my", "mine", "myself",
                                               "we", "us", "our", "ours",
                                               "you", "your", "yours",
                                               "it", "its"))
    if first_person > 0:
        pronoun_score = 0.9
    elif third_person > 0:
        pronoun_score = 0.6
    else:
        pronoun_score = 0.2

    # --- Perception/cognition verbs ---
    perc_count = sum(1 for w in words if w in PERCEPTION_VERBS)
    cog_count = sum(1 for w in words if w in COGNITION_VERBS)
    perception_score = min((perc_count * 0.7 + cog_count * 1.0) / max(n / 10, 1), 1.0)

    # --- Deictic markers ---
    proximal = sum(1 for w in words if w in PROXIMAL_DEICTICS)
    distal = sum(1 for w in words if w in DISTAL_DEICTICS)
    total_deictic = proximal + distal
    if total_deictic > 0:
        deictic_score = proximal / total_deictic
    else:
        deictic_score = 0.5

    # --- Sensory concreteness ---
    content_words = [w for w in words if len(w) > 3 and w.isalpha()]
    sensory_score = _sensory_concreteness(content_words)

    # --- Tense ---
    tense_score = 0.4  # default: neutral
    for t in tokens:
        if t.pos_ == "VERB":
            morph = t.morph.to_dict()
            tense = morph.get("Tense", "")
            if tense == "Pres":
                tense_score = 0.7
                break
            elif tense == "Past":
                tense_score = 0.4
                break

    # --- Interiority markers ---
    text = sent.text
    interiority = 0.0
    if "!" in text and not any(t.text == '"' for t in tokens[:3]):
        interiority += 0.3  # exclamation outside dialogue
    if "?" in text and not any(t.text == '"' for t in tokens[:3]):
        interiority += 0.3  # question in narration
    if "*" in text or "_" in text:
        interiority += 0.2  # italic markers (internal thought)

    # Thought content: extended reasoning/evaluation in third person.
    # "she thought about how X" or "she understood that Y" followed by
    # substantial clause content signals deep third, not just a perception tag.
    # Detect: cognition verb + subordinate clause (that/how/whether/if/about)
    thought_connectors = frozenset({
        "that", "how", "whether", "if", "about", "why", "what",
    })
    evaluation_words = frozenset({
        "meant", "meant", "mean", "means", "should", "shouldn't",
        "could", "couldn't", "would", "wouldn't", "might",
        "must", "enough", "wrong", "right", "true", "possible",
        "impossible", "certain", "uncertain", "sure", "unsure",
        "perhaps", "maybe", "probably", "never", "always",
        "every", "nothing", "everything", "anyone", "nobody",
    })
    # Check for cognition verb followed by thought connector
    for i, w in enumerate(words):
        if w in COGNITION_VERBS and i < len(words) - 2:
            # Look ahead for a thought connector within 3 words
            lookahead = words[i+1:i+4]
            if any(la in thought_connectors for la in lookahead):
                # Extended thought — boost based on sentence length
                # (longer = more reasoning = deeper interiority)
                clause_length = len(words) - i
                if clause_length > 10:
                    interiority += 0.4  # substantial extended thought
                elif clause_length > 5:
                    interiority += 0.2
                break

    # Evaluation language density — words that express judgment/assessment
    eval_count = sum(1 for w in words if w in evaluation_words)
    if eval_count >= 2:
        interiority += 0.2  # multiple evaluative terms = character judging

    interiority = min(interiority, 1.0)

    # Weighted combination
    distance = (
        pronoun_score * 0.20
        + perception_score * 0.25
        + deictic_score * 0.10
        + sensory_score * 0.15
        + tense_score * 0.10
        + interiority * 0.20
    )

    return min(max(distance, 0.0), 1.0)


@dataclass
class ZoomJump:
    """An unintentional psychic distance jump."""
    paragraph_idx: int
    sentence_idx: int
    text: str
    distance_before: float
    distance_after: float
    delta: float


@dataclass
class PsychicDistanceResult:
    """Full psychic distance analysis for a chapter."""
    filename: str
    sentence_scores: list[float]
    paragraph_means: list[float]
    mean_distance: float
    std_distance: float
    zoom_jumps: list[ZoomJump] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.mean_distance > 0.7:
            return "deep interior"
        elif self.mean_distance > 0.55:
            return "close third"
        elif self.mean_distance > 0.4:
            return "middle distance"
        else:
            return "establishing shot"


def analyze_chapter(
    text: str,
    filename: str,
    model_manager,
    jump_threshold: float = 0.25,
    jump_window: int = 3,
) -> PsychicDistanceResult:
    """Analyze psychic distance across an entire chapter.

    Args:
        text: chapter text
        filename: for reporting
        model_manager: provides spaCy
        jump_threshold: minimum delta to flag as zoom jump
        jump_window: sentences within this window count as a jump
    """
    nlp = model_manager.spacy

    items = split_paragraphs_with_breaks(text)
    all_scores: list[float] = []
    para_means: list[float] = []
    # Track (paragraph_idx, sentence_idx_in_para, score, sentence_text)
    score_map: list[tuple[int, int, float, str]] = []

    para_idx = 0
    for item in items:
        if item is None:
            # Scene break — reset jump detection window
            para_idx += 1
            continue

        doc = nlp(item)
        sents = list(doc.sents)
        para_scores = []

        for si, sent in enumerate(sents):
            s = score_sentence(sent)
            all_scores.append(s)
            para_scores.append(s)
            score_map.append((para_idx, si, s, sent.text.strip()))

        if para_scores:
            para_means.append(sum(para_scores) / len(para_scores))
        para_idx += 1

    if not all_scores:
        return PsychicDistanceResult(
            filename=filename,
            sentence_scores=[],
            paragraph_means=[],
            mean_distance=0.5,
            std_distance=0.0,
        )

    import numpy as np
    mean_d = float(np.mean(all_scores))
    std_d = float(np.std(all_scores))

    # Detect zoom jumps: large distance change within window, not across scene breaks
    jumps: list[ZoomJump] = []
    for i in range(jump_window, len(score_map)):
        pi_now, si_now, s_now, text_now = score_map[i]
        pi_prev, si_prev, s_prev, text_prev = score_map[i - 1]

        # Skip if crossing a scene break (different paragraph with gap)
        delta = abs(s_now - s_prev)
        if delta >= jump_threshold and pi_now == pi_prev or (pi_now == pi_prev + 1):
            # Check it's not just noise — look at the trend
            window_before = [score_map[j][2] for j in range(max(0, i - jump_window), i)]
            if window_before:
                mean_before = sum(window_before) / len(window_before)
                if abs(s_now - mean_before) >= jump_threshold:
                    jumps.append(ZoomJump(
                        paragraph_idx=pi_now,
                        sentence_idx=si_now,
                        text=text_now[:120],
                        distance_before=round(mean_before, 3),
                        distance_after=round(s_now, 3),
                        delta=round(s_now - mean_before, 3),
                    ))

    # Deduplicate jumps that are very close together
    deduped: list[ZoomJump] = []
    for j in jumps:
        if not deduped or abs(j.paragraph_idx - deduped[-1].paragraph_idx) > 1:
            deduped.append(j)

    return PsychicDistanceResult(
        filename=filename,
        sentence_scores=all_scores,
        paragraph_means=para_means,
        mean_distance=round(mean_d, 3),
        std_distance=round(std_d, 3),
        zoom_jumps=deduped,
    )
