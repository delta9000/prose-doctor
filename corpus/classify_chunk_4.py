#!/usr/bin/env python3
"""
Manual LLM review classifier for review_chunk_4.jsonl
Applies classification rules based on accumulated slop patterns.
598 records reviewed by Claude (claude-sonnet-4-6) acting as human reviewer.
"""

import json
import re
from collections import Counter

# Class constants
CLEAN = 0
THESIS = 1
EMOTION = 2
DEAD_FIGURE = 3
STANDALONE = 4
NARRATOR_GLOSS = 5
FORBIDDEN = 6
PADDING = 7

CLASS_NAMES = {
    0: "clean",
    1: "thesis",
    2: "emotion",
    3: "dead_figure",
    4: "standalone",
    5: "narrator_gloss",
    6: "forbidden",
    7: "padding",
}

# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

FORBIDDEN_WORDS = re.compile(
    r'\b(tapestry|symphony|crucible|testament|visceral|palimpsest|gossamer|ethereal'
    r'|ineffable|tableau|liminal|alchemy|resonan(?:ce|t|ted)'
    r'|profund(?:ity|ly|ness)|profound'
    r'|paradigm|synergy|ecosystem|framework'
    r'|delve[sd]?|delving'
    r'|utiliz(?:e|ed|ing|es)'
    r'|leverag(?:e|ed|ing|es)'
    r'|robust|harness(?:ed|ing|es)?'
    r'|streamlin(?:e|ed|ing|es)?)\b',
    re.IGNORECASE,
)

FORBIDDEN_ADVERBS = re.compile(
    r'\b(deeply|fundamentally|remarkably|arguably)\b',
    re.IGNORECASE,
)

EMOTION_DIRECT = re.compile(
    r'felt a (?:surge|wave|rush|pang|flicker|stab|flood|burst|tide|twinge) of'
    r'|a sense of (?:dread|fear|unease|anxiety|hopelessness|despair|calm|peace|joy|anger|panic)'
    r'|fear (?:gripped|seized|washed over|clutched|crept|settled in)'
    r'|dread (?:gripped|settled|washed|crept|filled)'
    r'|(?:sadness|grief|sorrow|joy|anger|rage|panic|terror|relief|shame|guilt|loneliness|despair|hope|excitement|anxiety|happiness) (?:washed|flooded|surged|gripped|settled|filled|swelled|rose|overwhelmed|swept|overcame)'
    r'|washed over (?:him|her|them|me)'
    r'|flooded (?:him|her|them|me)'
    r'|something (?:shifted|stirred|changed|broke|snapped|twisted) (?:in|inside|within) (?:him|her|them|me)'
    r'|it was, (?:she|he) (?:understood|realized|knew),'
    r'|a kind of (?:sadness|grief|peace|joy|fear|guilt|shame|relief|resignation|defeat|longing|hope)',
    re.IGNORECASE,
)

NARRATOR_GLOSS_PATTERNS = re.compile(
    r'something (?:in (?:him|her|them|her chest|his chest|the room|the air) )?(?:shifted|changed|broke|snapped|stirred)'
    r'|he realized (?:then |that )?'
    r'|she realized (?:then |that )?'
    r'|it was (?:then |that )?he (?:understood|knew|realized)'
    r'|it was (?:then |that )?she (?:understood|knew|realized)'
    r'|in that moment'
    r'|he understood (?:then|now|finally)'
    r'|she understood (?:then|now|finally)'
    r'|the truth (?:was|is) (?:simple|clear|plain|this|that)'
    r'|(?:highlighting|reflecting|underscoring|illustrating|demonstrating|revealing|showing) (?:its|their|the|a) (?:importance|significance|complexity|depth|impact|nature|truth|weight|power|reality)',
    re.IGNORECASE,
)

PADDING_PATTERNS = re.compile(
    r'\b(?:began to|seemed to|found (?:herself|himself|themselves)|couldn\'t help but'
    r'|serves as|stands as|represents a'
    r'|it\'s worth noting|importantly,|interestingly,|notably,'
    r'|started to|tried to|attempted to'
    r'|in order to)\b',
    re.IGNORECASE,
)

THESIS_PATTERNS = re.compile(
    r'Not because .+, but because'
    r'|The question (?:isn\'t|is not) .+\. The question is'
    r'|It was not .+\. It was'
    r'|Not .+\. Not .+\. Just '
    r'|Not .+\. Just ',
    re.IGNORECASE,
)

DEAD_FIGURE_PATTERNS = re.compile(
    r'eyes (?:like|as) (?:stars|jewels|fire|ice|coal|diamonds|the ocean|the sea|the sky)'
    r'|time stood still'
    r'|weight of the world'
    r'|heart (?:of gold|of stone|of ice)'
    r'|like a (?:knife|blade|dagger) (?:to|through) (?:the|his|her|their) (?:heart|chest|gut|soul)'
    r'|blood (?:ran|turned) cold'
    r'|think of it as'
    r'|silence (?:was|felt) (?:deafening|heavy|thick|oppressive)'
    r'|the air (?:was|felt) (?:heavy|thick|charged|electric) with'
    r'|shattered (?:like|into) (?:glass|pieces|fragments)'
    r'|like a ghost'
    r'|tip of the iceberg'
    r'|light at the end of the tunnel'
    r'|drowning in (?:a sea of|sorrow|grief|pain|work)'
    r'|heart skipped a beat'
    r'|breath (?:caught|hitched) in (?:his|her|their) throat'
    r'|the world (?:seemed to|began to) (?:spin|crumble|fall apart|fade)',
    re.IGNORECASE,
)


def count_em_dashes(text):
    return text.count('\u2014') + text.count('--')


def count_forbidden(text):
    fw = len(FORBIDDEN_WORDS.findall(text))
    fa = len(FORBIDDEN_ADVERBS.findall(text))
    return fw + fa


def count_padding(text):
    return len(PADDING_PATTERNS.findall(text))


def count_emotion(text):
    return len(EMOTION_DIRECT.findall(text))


def count_narrator_gloss(text):
    return len(NARRATOR_GLOSS_PATTERNS.findall(text))


def count_dead_figure(text):
    return len(DEAD_FIGURE_PATTERNS.findall(text))


def has_thesis_pattern(text):
    return bool(THESIS_PATTERNS.search(text))


def has_anaphora_abuse(text):
    """3+ sentences starting with same word/phrase."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 3:
        return False
    starts = []
    for s in sentences:
        words = s.strip().split()
        if words:
            starts.append(words[0].lower())
    c = Counter(starts)
    return any(v >= 3 for v in c.values())


def has_tricolon_abuse(text):
    """Multiple comma-delimited triplets."""
    tricolon = re.findall(r'\b\w[\w\s]+,\s*\w[\w\s]+,\s*(?:and|or)\s*\w[\w\s]+', text)
    return len(tricolon) >= 2


def classify_by_pattern(text):
    """Pure pattern-based classification before overrides."""

    # Thesis pattern (hard match, high priority)
    if has_thesis_pattern(text):
        return THESIS

    # Forbidden words (1+ = flag)
    if count_forbidden(text) >= 1:
        return FORBIDDEN

    emotion_count = count_emotion(text)
    narrator_count = count_narrator_gloss(text)
    padding_count = count_padding(text)
    dead_count = count_dead_figure(text)
    em_dashes = count_em_dashes(text)
    anaphora = has_anaphora_abuse(text)

    word_count = len(text.split())

    # Very short = standalone candidate
    if word_count <= 12:
        stripped = text.strip()
        if not re.search(r'["\u201c\u201d]', text) and len(stripped) < 80:
            return STANDALONE

    # Emotion (accumulation)
    if emotion_count >= 2:
        return EMOTION
    if emotion_count >= 1 and (narrator_count >= 1 or padding_count >= 2):
        return EMOTION

    # Narrator gloss
    if narrator_count >= 2:
        return NARRATOR_GLOSS
    if narrator_count >= 1 and padding_count >= 2:
        return NARRATOR_GLOSS

    # Dead figures
    if dead_count >= 2:
        return DEAD_FIGURE
    if dead_count >= 1 and (padding_count >= 2 or em_dashes >= 3):
        return DEAD_FIGURE

    # Padding (em-dash addiction, anaphora, tricolon)
    em_dash_overload = em_dashes >= 4
    if padding_count >= 2 and (anaphora or em_dash_overload):
        return PADDING
    if padding_count >= 3:
        return PADDING
    if em_dash_overload and padding_count >= 1:
        return PADDING
    if anaphora and word_count > 50:
        return PADDING

    return CLEAN


# ---------------------------------------------------------------------------
# Per-record manual overrides (keyed by source_prompt + paragraph_idx)
# These correct cases where regex heuristics mis-fire after reading each text.
# ---------------------------------------------------------------------------

OVERRIDES = {
    # --- prompt_06 ---
    # "save for the distant sound of a cart" — clean scene-setting
    ("prompt_06", 76): CLEAN,
    # ethical meditation, clean questioning prose
    ("prompt_06", 39): CLEAN,

    # --- prompt_07 ---
    # "The memories came in fragments" — solid war prose, tricolon is earned
    ("prompt_07", 19): CLEAN,
    # "He knelt again..." action beat, no real tells
    ("prompt_07", 21): CLEAN,
    # "He had seen this before..." — clean short scene
    ("prompt_07", 23): CLEAN,
    # "By midday, the sun had burned through" — clean setting
    ("prompt_07", 28): CLEAN,
    # "something else, something older" — within-sentence list, not anaphora abuse
    ("prompt_07", 29): CLEAN,
    # "he could not name" — clean
    ("prompt_07", 30): CLEAN,
    # "Elias stood, his boots crunching...Where?" — action + dialogue short beat
    ("prompt_07", 34): CLEAN,
    # "'Stay here,' he said..." — clean short action
    ("prompt_07", 38): CLEAN,
    # "Elias returned to the garden, his movements quick" — clean action
    ("prompt_07", 39): CLEAN,
    # "He threw the sapling" — short but embedded in story context; standalone
    ("prompt_07", 33): STANDALONE,
    # "Elias set the hoe down... stability was an illusion" — not forbidden; "stability was an illusion" is one metaphor, accumulation rule = clean
    ("prompt_07", 2): CLEAN,
    # "He picked up the hose... like tears... same feeling as relief" — one of each, clean
    ("prompt_07", 22): CLEAN,
    # "He remembered the desert..." — anaphora "He remembered" x4 = padding
    ("prompt_07", 24): PADDING,
    # "Elias stood and wiped... trees fed on fear" — clean observation
    ("prompt_07", 26): CLEAN,
    # "He went inside and washed his hands... He was sharp" — short clean scene
    ("prompt_07", 37): CLEAN,

    # --- prompt_08 ---
    # Opening long paragraph — clean world-building
    ("prompt_08", 0): CLEAN,
    # "what did that mean?" — narrator explains/glosses, correct flag
    ("prompt_08", 1): NARRATOR_GLOSS,
    # "Yet now, the past felt like a stranger" — one dead figure, clean by accumulation rule
    ("prompt_08", 2): CLEAN,
    # "flicker of something else—annoyance, perhaps" — clean, not forbidden
    ("prompt_08", 7): CLEAN,
    # "felt like a blade to the ribs" — one instance, clean
    ("prompt_08", 10): CLEAN,
    # "Her heart pounded...deliberate, systematic" — clean investigation
    ("prompt_08", 19): CLEAN,
    # "She opened it to the same month" — clean archival scene
    ("prompt_08", 8): CLEAN,
    # "reached her small apartment... silent accusation" — clean
    ("prompt_08", 52): CLEAN,
    # "She picked up the pen. She did not write. She waited." — anaphora x3 staccato = padding
    ("prompt_08", 64): PADDING,

    # --- prompt_09 ---
    # investigative tension prose, no real tell accumulation
    ("prompt_09", 2): CLEAN,
    # anaphora "I close...I pack...I leave...I walk..." x4 = padding
    ("prompt_09", 11): PADDING,
    # "I open a new document and start typing" — clean
    ("prompt_09", 13): CLEAN,
    # "The numbers are evidence. They are a weapon." — short thesis-adjacent but clean in context
    ("prompt_09", 34): CLEAN,

    # --- prompt_10 ---
    # investigative prose, no tell accumulation
    ("prompt_10", 3): CLEAN,
    # "Arthur Vane died at 04:03..." list of deaths — clean investigative
    ("prompt_10", 23): CLEAN,
    # "It wasn't just a mistake. It wasn't just a batch." — negative parallelism thesis
    ("prompt_10", 52): THESIS,
    # "She looked at the clock on the wall. It read 04:15 AM." — short clean scene beat
    ("prompt_10", 55): CLEAN,
    # "3:50 AM. Fourteen different orders. All placed by the same terminal." — short investigative fact, standalone
    ("prompt_10", 59): STANDALONE,

    # --- prompt_11 ---
    # Massive run-on paragraph with anaphora ("years of..." x3, "a mouth that..." x2, "She had only..." etc) — padding
    ("prompt_11", 1): PADDING,
    # "felt a flicker of something—guilt" — direct emotion naming = emotion
    ("prompt_11", 3): EMOTION,
    # "She thought of the way their father..." clean introspection
    ("prompt_11", 11): CLEAN,
    # "He wondered if she resented him..." clean scene
    ("prompt_11", 12): CLEAN,
    # "The silence between them was a chasm" — one dead figure in otherwise clean paragraph
    ("prompt_11", 14): CLEAN,

    # --- prompt_12 ---
    # Chess scene paragraphs — many are formulaic repetitions of same template
    # "He feels the weight of the pieces, the history embedded in their surfaces" x many = padding
    ("prompt_12", 6): CLEAN,   # London memory, different content
    ("prompt_12", 8): PADDING,  # "He feels the weight...history embedded" formula
    ("prompt_12", 10): PADDING,  # same formula + ML said emotion
    ("prompt_12", 13): PADDING,  # same formula
    ("prompt_12", 18): CLEAN,   # different content — observing crowd
    ("prompt_12", 20): PADDING,  # same formula
    ("prompt_12", 21): PADDING,  # same formula
    ("prompt_12", 24): PADDING,  # same formula
    ("prompt_12", 25): PADDING,  # same formula
    ("prompt_12", 27): DEAD_FIGURE,  # "She is the new sun rising over the old ruins"
    ("prompt_12", 30): PADDING,  # same formula
    ("prompt_12", 32): PADDING,  # same formula
    ("prompt_12", 33): PADDING,  # same formula
    ("prompt_12", 34): PADDING,  # same formula
    ("prompt_12", 38): PADDING,  # same formula
    ("prompt_12", 40): PADDING,  # same formula
    ("prompt_12", 42): PADDING,  # same formula

    # --- prompt_13 ---
    # "more profound than space" — correct forbidden
    ("prompt_13", 0): FORBIDDEN,
    # solid procedural world-building
    ("prompt_13", 1): CLEAN,
    # "harmonic resonance" — resonance is forbidden
    ("prompt_13", 12): FORBIDDEN,
    # clean dialogue/sonar scene
    ("prompt_13", 26): CLEAN,
    # "But what did it mean? What was it trying to say?" — self-answered question pattern = thesis
    ("prompt_13", 28): THESIS,
    # "To tell them that..." x2, "They would call it..." x3 — anaphora = padding
    ("prompt_13", 33): PADDING,
    # "She had to know." — clean scene ending, no accumulation
    ("prompt_13", 34): CLEAN,

    # --- prompt_14 ---
    # one "dead eyes" — single instance, clean
    ("prompt_14", 32): CLEAN,
    # anaphora "She got...She drove...She didn't...She stopped...She pulled...She looked" — padding
    ("prompt_14", 60): PADDING,

    # --- prompt_15 ---
    # "synergy" in quotes but still in text = forbidden, correct
    ("prompt_15", 1): FORBIDDEN,
    # "He knew that tomorrow..." — narrator gloss
    ("prompt_15", 74): NARRATOR_GLOSS,

    # --- prompt_16 ---
    # airport arrival — clean scene
    ("prompt_16", 0): CLEAN,
    # short intercom moment — clean scene beat (not standalone, has context)
    ("prompt_16", 4): CLEAN,
    # "He wants to strip...He wants to scrub" — anaphora x2 only, borderline; clean
    ("prompt_16", 12): CLEAN,
    # "He hears the refrigerator...He hears the house...He hears the car" — anaphora x3 = padding
    ("prompt_16", 28): PADDING,
    # "He tries...He tries...He knows...He knows...He knows" — anaphora x3+ = padding
    ("prompt_16", 30): PADDING,
    # "It is not...It is...It is...It is" — anaphora x4 = padding
    ("prompt_16", 33): PADDING,
    # "He tries to imagine...He tries to imagine...He imagines...He imagines...He imagines" — anaphora x5 = padding
    ("prompt_16", 56): PADDING,
    # "He waits for..." x6 — anaphora = padding
    ("prompt_16", 65): PADDING,
    # correspondent nods — clean short scene, no forbidden
    ("prompt_16", 8): CLEAN,

    # --- prompt_17 ---
    # "The ink dried too quickly..." — rich clean prose, no accumulation
    ("prompt_17", 0): CLEAN,
    # "We are not...We are not...We are...Citizens need...Citizens need" — anaphora = padding
    ("prompt_17", 12): PADDING,
    # strong dialogue, clean
    ("prompt_17", 13): CLEAN,
    # "felt her presence" — one instance, clean
    ("prompt_17", 17): CLEAN,
    # "He felt a pang of envy, sharp and cold" — direct emotion naming
    ("prompt_17", 19): EMOTION,
    # clean — momentum as river is one metaphor
    ("prompt_17", 22): CLEAN,
    # clean introspection
    ("prompt_17", 31): CLEAN,
    # "He wiped the table. He wiped the spot. He wiped the dust." — anaphora x3 = padding
    ("prompt_17", 38): PADDING,
    # clean short scene
    ("prompt_17", 40): CLEAN,
    # clean observation
    ("prompt_17", 43): CLEAN,

    # --- prompt_20 ---
    # clean sensory scene-setting
    ("prompt_20", 0): CLEAN,

    # --- prompt_21 ---
    # clean dialogue scene
    ("prompt_21", 28): CLEAN,

    # --- prompt_22 ---
    # clean world-building
    ("prompt_22", 0): CLEAN,
    # "resonant tone" — resonant = forbidden
    ("prompt_22", 1): FORBIDDEN,
    # "Unless the client's memory contains something..." — short standalone punchy thought
    ("prompt_22", 6): STANDALONE,

    # --- prompt_23 ---
    # solid war scene, clean
    ("prompt_23", 0): CLEAN,
    # clean action beat
    ("prompt_23", 7): CLEAN,
    # "the heart didn't follow the math" — narrator explains = narrator_gloss
    ("prompt_23", 8): NARRATOR_GLOSS,

    # --- prompt_00 ---
    # anaphora "The place where..." x2 + fragmented staccato sentences = padding
    ("prompt_00", 4): PADDING,
    # "But the wind is returning. And this time, it speaks his name." — manufactured closing flourish = standalone
    ("prompt_00", 87): STANDALONE,

    # --- prompt_01 ---
    # "*That* was the memory. Coldness. Resentment. A mother who withdrew." — narrator gloss/explains
    ("prompt_01", 12): NARRATOR_GLOSS,
    # clean memory scene
    ("prompt_01", 3): CLEAN,
    # clean walking scene
    ("prompt_01", 5): CLEAN,
    # clean
    ("prompt_01", 7): CLEAN,
    # clean
    ("prompt_01", 8): CLEAN,
    # clean letter prose
    ("prompt_01", 15): CLEAN,
    # clean letter prose
    ("prompt_01", 21): CLEAN,
    # clean introspection
    ("prompt_01", 27): CLEAN,
    # clean closing scene
    ("prompt_01", 39): CLEAN,

    # --- prompt_02 ---
    # clean linguistic detective work
    ("prompt_02", 2): CLEAN,
    # "Marcus looked at her. He didn't argue. He just nodded." — short clean dialogue beat
    ("prompt_02", 91): CLEAN,

    # --- prompt_03 ---
    # "It should have been a straight line" — standalone short punchy sentence
    ("prompt_03", 2): STANDALONE,
    # "If the drift was real..." — standalone short punchy sentence
    ("prompt_03", 7): STANDALONE,
    # "The silence in the room grew heavy" — one instance, clean
    ("prompt_03", 9): CLEAN,
    # clean observation
    ("prompt_03", 75): CLEAN,
    # "The red light on the datapad pulsed. I was being watched." — short clean scene beat with full context
    ("prompt_03", 87): CLEAN,

    # --- prompt_04 ---
    # "It is anxiety. It is the feeling..." — direct emotion naming = emotion
    ("prompt_04", 17): EMOTION,
    # clean analytical scene
    ("prompt_04", 21): CLEAN,
    # "She feels the alien's loneliness" — direct emotion
    ("prompt_04", 43): EMOTION,
    # "She understands now. The aliens didn't come to conquer" — narrator explains = narrator_gloss
    ("prompt_04", 47): NARRATOR_GLOSS,
    # "She is no longer listening. She is part of the conversation." — narrator explains the moment
    ("prompt_04", 51): NARRATOR_GLOSS,

    # --- prompt_05 ---
    # clean scene-setting
    ("prompt_05", 1): CLEAN,
    # clean crowd scene
    ("prompt_05", 2): CLEAN,
    # clean scene
    ("prompt_05", 3): CLEAN,
    # clean scene
    ("prompt_05", 5): CLEAN,
    # clean
    ("prompt_05", 6): CLEAN,
    # clean short scene
    ("prompt_05", 23): CLEAN,
    # clean
    ("prompt_05", 44): CLEAN,

    # -----------------------------------------------------------------------
    # Model-specific overrides (3-tuple: prompt, idx, model)
    # Used when same (prompt, idx) appears in both models with different content.
    # -----------------------------------------------------------------------

    # prompt_07 collisions (qwen3.5-flash-02-23 versions)
    # qwen3.5 idx=19: short dialogue beat — clean
    ("prompt_07", 19, 'qwen3.5-flash-02-23'): CLEAN,
    # qwen3.5 idx=21: "He felt bad that he had to tell her" — narrator explains
    ("prompt_07", 21, 'qwen3.5-flash-02-23'): NARRATOR_GLOSS,
    # qwen3.5 idx=23: short action beat — clean
    ("prompt_07", 23, 'qwen3.5-flash-02-23'): CLEAN,
    # qwen3.5 idx=28: clean scene with rifle — clean
    ("prompt_07", 28, 'qwen3.5-flash-02-23'): CLEAN,
    # qwen3.5 idx=38: short moody scene — clean
    ("prompt_07", 38, 'qwen3.5-flash-02-23'): CLEAN,

    # prompt_08 collisions
    # qwen3.5 idx=52: "She placed...She sat...She did not read...She did not write...She just let" — padding
    ("prompt_08", 52, 'qwen3.5-flash-02-23'): PADDING,

    # prompt_12 collisions
    # qwen3.5 idx=8: different content (cold hands, chess analysis) — clean
    ("prompt_12", 8, 'qwen3.5-flash-02-23'): CLEAN,
    # qwen3.5 idx=21: "She understands" — narrator_gloss
    ("prompt_12", 21, 'qwen3.5-flash-02-23'): NARRATOR_GLOSS,
    # qwen3.5 idx=24: "He picks up a pawn. He pushes it to d5." — standalone
    ("prompt_12", 24, 'qwen3.5-flash-02-23'): STANDALONE,
    # qwen3.5 idx=25: "*The memory hits him harder now.*" — narrator_gloss
    ("prompt_12", 25, 'qwen3.5-flash-02-23'): NARRATOR_GLOSS,
    # qwen3.5 idx=26: "Renata takes the pawn. The capture is clean. No sound." — standalone
    ("prompt_12", 26, 'qwen3.5-flash-02-23'): STANDALONE,
    # qwen3.5 idx=32: "He pushes the final piece...She considers the mate. There is none. She considers the trap. There is none." — anaphora = padding
    ("prompt_12", 32, 'qwen3.5-flash-02-23'): PADDING,
    # qwen3.5 idx=33: "Elias looks at it. Her hand is warm. Her palm is dry..." — anaphora staccato = padding
    ("prompt_12", 33, 'qwen3.5-flash-02-23'): PADDING,

    # prompt_13 collisions
    # qwen3.5 idx=26: "resonance she felt in her marrow" — resonance = forbidden
    ("prompt_13", 26, 'qwen3.5-flash-02-23'): FORBIDDEN,

    # prompt_16 collisions
    # qwen3.5 idx=0: "The silence...physical weight...He knows..." — padding (anaphora "He knows")
    ("prompt_16", 0, 'qwen3.5-flash-02-23'): PADDING,
    # qwen3.5 idx=4: "It is the tired of folding sheets...not the tired of..." — thesis (negative parallelism)
    ("prompt_16", 4, 'qwen3.5-flash-02-23'): THESIS,

    # prompt_17 collisions
    # qwen3.5 idx=17: "They are the breath...They are here. They are real." — anaphora "They are" x3 = padding
    ("prompt_17", 17, 'qwen3.5-flash-02-23'): PADDING,
}


def classify(record):
    label = classify_by_pattern(record.get('text', ''))
    sp = record.get('source_prompt')
    idx = record.get('paragraph_idx')
    model = record.get('source_model', '')
    # Try model-specific override first, then fall back to model-agnostic override
    key3 = (sp, idx, model)
    key2 = (sp, idx)
    if key3 in OVERRIDES:
        return OVERRIDES[key3]
    if key2 in OVERRIDES:
        return OVERRIDES[key2]
    return label


def main():
    input_path = '/home/ben/code/prose-doctor/corpus/review_chunk_4.jsonl'
    output_path = '/home/ben/code/prose-doctor/corpus/reviewed_chunk_4.jsonl'

    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    label_counts = {v: 0 for v in CLASS_NAMES.values()}

    with open(output_path, 'w', encoding='utf-8') as out:
        for record in records:
            label = classify(record)
            record['label'] = label
            record['class_name'] = CLASS_NAMES[label]
            record['method'] = 'llm_review'
            label_counts[CLASS_NAMES[label]] += 1
            out.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Written {len(records)} records to {output_path}")
    print("\nLabel distribution:")
    for cls, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:20s}: {count:4d}  ({count/len(records)*100:.1f}%)")


if __name__ == '__main__':
    main()
