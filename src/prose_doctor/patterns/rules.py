"""Canonical pattern registry for prose analysis.

Consolidates all regex patterns from slop_scorer, proof_scanner, and
density_analyzer into one registry. Each pattern has a name, category,
severity, and compiled regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from prose_doctor.patterns.taxonomy import CLASS_NAMES


@dataclass(frozen=True)
class RulePattern:
    """A single prose pattern rule."""

    name: str
    regex: re.Pattern
    category: str  # maps to SlopCategory value
    severity: str  # "high", "medium", "density"


# ---------------------------------------------------------------------------
# Rule-based patterns from slop_scorer (classifier supplement)
# ---------------------------------------------------------------------------

def build_rule_patterns(character_names: list[str] | None = None) -> list[RulePattern]:
    """Build the canonical list of all prose patterns.

    Character names are injected into patterns that reference them.
    If no names provided, uses pronoun-only variants.
    """
    names = character_names or []
    # Build alternation for character names + pronouns
    pronoun_alts = "her|him|his|them|their"
    if names:
        name_alts = "|".join(re.escape(n) for n in names)
        subject_alts = pronoun_alts + "|" + name_alts
    else:
        subject_alts = pronoun_alts

    # Build name alternation for he/she/name patterns
    he_she_alts = "He|She|It"
    if names:
        he_she_alts += "|" + "|".join(re.escape(n) for n in names)

    patterns: list[RulePattern] = []

    # === SLOP SCORER PATTERNS (classifier supplement) ===

    # "Not X. Y." -- bare negation fragment -> positive assertion
    patterns.append(RulePattern(
        name="not_x_period_y",
        regex=re.compile(
            r'(?:^|(?<=\. )|(?<=\.\n))Not '
            r'(?!(?:everyone|everything|everybody|anymore|now|yet|quite|exactly|entirely|'
            r'necessarily|just|only|always|here|there|long|far|much|many|all|even|once)\b)'
            r'(?:a |an |the )?[a-z]\w(?:(?!,?\s*but\b)[^.]){0,40}\.\s+[A-Z]',
            re.MULTILINE),
        category="thesis",
        severity="medium",
    ))

    # "Not X. Not Y. Z." -- triple-beat negation
    patterns.append(RulePattern(
        name="not_x_not_y_z",
        regex=re.compile(r'\bNot [^.]{1,40}\.\s+Not [^.]{1,40}\.\s+[A-Z]'),
        category="thesis",
        severity="medium",
    ))

    # "It/She/He wasn't X. It/She/He was Y."
    patterns.append(RulePattern(
        name="wasnt_it_was",
        regex=re.compile(
            r"\b(?:It|This|That|She|He) wasn[\u2019']?t [^.]+\.\s+"
            r"(?:It|This|That|She|He) was "),
        category="thesis",
        severity="medium",
    ))

    # "was not X. It was Y." -- formal variant
    patterns.append(RulePattern(
        name="was_not_it_was",
        regex=re.compile(r"\bwas not [^.]+\.\s+(?:It|This|That) was "),
        category="thesis",
        severity="medium",
    ))

    # "Not the X. The Y." -- definite article contrast
    patterns.append(RulePattern(
        name="not_the_x_the_y",
        regex=re.compile(r'\bNot the [^.]{3,50}\.\s+The [^.]{3,50}'),
        category="thesis",
        severity="medium",
    ))

    # "Not X -- Y" -- dash-separated rhetorical correction
    patterns.append(RulePattern(
        name="not_x_dash_y",
        regex=re.compile(
            r'(?:^|(?<=\. ))Not [^.—\u2013\n]{2,30}[—\u2013]\s*\w',
            re.MULTILINE),
        category="thesis",
        severity="medium",
    ))

    # "began to" / "started to" -- weak verb crutch
    patterns.append(RulePattern(
        name="began_started_to",
        regex=re.compile(r'\b(?:began|started) to \w+'),
        category="padding",
        severity="medium",
    ))

    # "as if" x2+ in one paragraph -- simile density overload
    patterns.append(RulePattern(
        name="as_if_density",
        regex=re.compile(r'(?:as if).*(?:as if)'),
        category="dead_figure",
        severity="medium",
    ))

    # "seemed to [verb]" -- hedging/distancing
    patterns.append(RulePattern(
        name="seemed_to",
        regex=re.compile(r'\bseemed to \w+'),
        category="padding",
        severity="medium",
    ))

    # "the weight of [abstraction]"
    patterns.append(RulePattern(
        name="weight_of_abstraction",
        regex=re.compile(
            r'\bthe\s+weight\s+of\s+(?:the\s+)?(?:observation|responsibility|it|'
            r'the\s+moment|everything|the\s+decision|what|his|her|their|'
            r'the\s+silence|the\s+loss|the\s+knowledge|the\s+truth|'
            r'the\s+question|the\s+realization|the\s+admission|'
            r'the\s+confession|the\s+accusation|the\s+implication)\b',
            re.IGNORECASE),
        category="emotion",
        severity="medium",
    ))

    # "something in [pronoun/name]" -- vague subject dodging specificity
    patterns.append(RulePattern(
        name="something_in_pronoun",
        regex=re.compile(
            rf'\bsomething\s+in\s+(?:{subject_alts})\b',
            re.IGNORECASE),
        category="narrator_gloss",
        severity="medium",
    ))

    # "deep, resonant [noun]" -- adjective pair wallpaper
    patterns.append(RulePattern(
        name="deep_resonant",
        regex=re.compile(r'\bdeep,?\s+resonant\b', re.IGNORECASE),
        category="padding",
        severity="density",
    ))

    # "patient [non-person noun]" -- personifying non-person things
    patterns.append(RulePattern(
        name="patient_nonperson",
        regex=re.compile(
            r'\bpatient\s+(?:hum|vibration|pulse|rhythm|pressure|tone|drone|'
            r'persistence|attention|cycling|heat|glow|accretion|indifference|'
            r'waiting|regard|presence|probe|signal|machine|silence|dark|growth|'
            r'warmth|cold|light|current|frequency|erosion|observation|process|'
            r'work|note|ticking|measurement|pace)\b',
            re.IGNORECASE),
        category="padding",
        severity="density",
    ))

    # "[abstract noun] sat/settled/lodged in [body part]" -- phantom sensation
    patterns.append(RulePattern(
        name="phantom_sensation",
        regex=re.compile(
            r'\b(?:The\s+)?(?:fact|truth|name|question|knowledge|lie|word|words|'
            r'silence|loss|grief|fear|thought|idea|realization|admission|weight|'
            r'certainty|wrongness|rightness|loneliness|absence|dread|shame|guilt|'
            r'anger|relief)\s+'
            r'(?:sat|settled|lodged|lived|landed|pooled|gathered|coiled)\s+'
            r'(?:in|on|behind|beneath|against|at|between)\s+'
            r'(?:h(?:is|er)|the)\s+'
            r'(?:chest|ribs|sternum|jaw|throat|stomach|spine|skull|temple|'
            r'temples|molars|teeth|belly|gut|breastbone|lungs|diaphragm)\b'),
        category="emotion",
        severity="density",
    ))

    # "a [role]'s [quality]" -- appositional label shortcut
    patterns.append(RulePattern(
        name="role_quality_label",
        regex=re.compile(
            r"\ba\s+\w+(?:'s|\u2019s)\s+(?:\w+\s+)?"
            r'(?:habit|patience|instinct|reflex|eye|precision|care|attention|'
            r'caution|discipline|pragmatism|gesture|certainty|economy|efficiency|'
            r'assessment|calculation|curiosity|steadiness|competence|awareness|'
            r'detachment|clarity|focus|calm|stubbornness|grip)\b'),
        category="narrator_gloss",
        severity="density",
    ))

    # "something that had nothing to do with [X]"
    patterns.append(RulePattern(
        name="something_nothing_to_do",
        regex=re.compile(
            r'\bsomething\s+that\s+had\s+nothing\s+to\s+do\s+with\b',
            re.IGNORECASE),
        category="narrator_gloss",
        severity="medium",
    ))

    # "He/She didn't [verb]. He/She [verbed]." -- negation-then-positive pair
    patterns.append(RulePattern(
        name="didnt_verb_verbed",
        regex=re.compile(
            rf'\b(?:{he_she_alts})'
            r"\s+didn[\u2019']?t\s+\w+[^.]*\.\s+"
            rf'(?:{he_she_alts})'
            r'\s+\w+ed\b'),
        category="thesis",
        severity="medium",
    ))

    # "That was what it meant" / "That was what [noun] was"
    patterns.append(RulePattern(
        name="that_was_what",
        regex=re.compile(
            r'\bThat\s+was\s+(?:what\s+(?:it|he|she|they|the\s+\w+)\s+'
            r'(?:meant|was|felt\s+like|looked\s+like|sounded\s+like)|'
            r'the\s+(?:word|truth|answer|cost|price|difference|problem|thing))\b'),
        category="narrator_gloss",
        severity="medium",
    ))

    # === PROOF SCANNER PATTERNS (high severity) ===

    patterns.append(RulePattern(
        name="forbidden_word",
        regex=re.compile(r"\b(tapestry|symphony)\b", re.IGNORECASE),
        category="forbidden",
        severity="high",
    ))

    patterns.append(RulePattern(
        name="emotion_naming",
        regex=re.compile(
            r"\b(?:felt\s+(?:scared|afraid|happy|sad|peaceful|anxious|"
            r"a\s+surge|a\s+wave|a\s+sense)"
            r"|sense\s+of\s+(?:peace|dread|fear|joy|unease|wonder)"
            r"|a\s+feeling\s+of)\b",
            re.IGNORECASE),
        category="emotion",
        severity="high",
    ))

    patterns.append(RulePattern(
        name="realization_verb",
        regex=re.compile(
            rf"\b(?:(?:{he_she_alts.lower()}|{he_she_alts})\s+"
            r"(?:realized|understood)|it\s+dawned\s+on)\b",
            re.IGNORECASE),
        category="narrator_gloss",
        severity="high",
    ))

    patterns.append(RulePattern(
        name="grammar_a_an",
        regex=re.compile(
            r"\ba\s+(?:urgent|unusual|un\w+|utter|open|old|ear\w*|eye\w*|"
            r"hour|honest|honor)",
            re.IGNORECASE),
        category="forbidden",
        severity="high",
    ))

    patterns.append(RulePattern(
        name="adverb_dialogue_tag",
        regex=re.compile(
            r"\bsaid\s+(?:quietly|softly|loudly|gently|firmly|angrily|"
            r"sadly|happily|quickly|slowly)\b",
            re.IGNORECASE),
        category="padding",
        severity="high",
    ))

    patterns.append(RulePattern(
        name="scene_header",
        regex=re.compile(r"^#{2,3}\s+", re.MULTILINE),
        category="forbidden",
        severity="high",
    ))

    patterns.append(RulePattern(
        name="star_divider",
        regex=re.compile(r"^\*{3}\s*$", re.MULTILINE),
        category="forbidden",
        severity="high",
    ))

    # === PROOF SCANNER PATTERNS (medium severity) ===

    patterns.append(RulePattern(
        name="thesis_statement",
        regex=re.compile(
            r"\bwas\s+not\s+.{3,30}\.\s+(?:It|They|She|He)\s+was\b"),
        category="thesis",
        severity="medium",
    ))

    patterns.append(RulePattern(
        name="escalation_metaphor",
        regex=re.compile(
            r"\bno\s+longer\s+.{3,30}\.\s+(?:It|They|She|He)\s+was\b"),
        category="thesis",
        severity="medium",
    ))

    patterns.append(RulePattern(
        name="not_x_but_y",
        regex=re.compile(r"\bnot\s+.{3,30},?\s+but\s+", re.IGNORECASE),
        category="thesis",
        severity="medium",
    ))

    patterns.append(RulePattern(
        name="symmetrical_closer",
        regex=re.compile(
            r"\bDifferent\s+.{3,20},\s+same\s+.{3,20}\.",
            re.IGNORECASE),
        category="thesis",
        severity="medium",
    ))

    patterns.append(RulePattern(
        name="something_like",
        regex=re.compile(r"\bsomething\s+like\b", re.IGNORECASE),
        category="narrator_gloss",
        severity="medium",
    ))

    patterns.append(RulePattern(
        name="before_now_temporal",
        regex=re.compile(r"\bBefore[,.]"),
        category="thesis",
        severity="medium",
    ))

    # === DENSITY PATTERNS (structural) ===

    # Prepositional stacking: "through X, into Y, against Z"
    patterns.append(RulePattern(
        name="prepositional_stacking",
        regex=re.compile(
            r'\b(?:through|into|against|beneath|behind|across|along|beyond)\s+\w+'
            r'(?:,|\s+and)\s+'
            r'(?:through|into|against|beneath|behind|across|along|beyond)\s+\w+'
            r'(?:,|\s+and)\s+'
            r'(?:through|into|against|beneath|behind|across|along|beyond)\s+\w+'),
        category="padding",
        severity="density",
    ))

    # Over-resolution: substantial clause + trailing "like X" simile
    patterns.append(RulePattern(
        name="over_resolution",
        regex=re.compile(
            r'([^.]{40,}?)'
            r',?\s+like\s+'
            r'([^.]{5,40})'
            r'\.'),
        category="dead_figure",
        severity="density",
    ))

    # Cost-recovery cadence: ability -> symptom -> wipe -> continue
    patterns.append(RulePattern(
        name="cost_recovery",
        regex=re.compile(
            r'\b(?:iron taste|tasted (?:of )?iron|copper taste|tasted (?:of )?copper|'
            r'salt on (?:her|his) (?:tongue|lips)|nosebleed|blood (?:from|on) (?:his|her) '
            r'(?:nose|lip|nostril)|wiped (?:the )?blood|the taste faded|'
            r'the (?:iron|copper|salt) faded)\b',
            re.IGNORECASE),
        category="emotion",
        severity="density",
    ))

    return patterns


# Pre-built default patterns (no character names)
DEFAULT_PATTERNS = build_rule_patterns()


def check_rules(text: str, patterns: list[RulePattern] | None = None) -> list[dict]:
    """Run rule-based pattern checks on text.

    Returns list of matched rules: [{pattern_name, category, severity}]
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    matches = []
    for rule in patterns:
        if rule.regex.search(text):
            class_id = CLASS_NAMES.index(rule.category) if rule.category in CLASS_NAMES else 1
            matches.append({
                "pattern_name": rule.name,
                "category": rule.category,
                "class_name": rule.category,
                "class_id": class_id,
                "severity": rule.severity,
            })
    return matches
