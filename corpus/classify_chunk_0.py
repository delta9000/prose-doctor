#!/usr/bin/env python3
"""
Manual LLM review of review_chunk_0.jsonl.
Classifications are based on careful reading of each paragraph.
"""

import json
import re
import sys

# ─── Class constants ──────────────────────────────────────────────────────────
CLEAN = 0
THESIS = 1
EMOTION = 2
DEAD_FIGURE = 3
STANDALONE = 4
NARRATOR_GLOSS = 5
FORBIDDEN = 6
PADDING = 7

CLASS_NAMES = {
    0: 'clean', 1: 'thesis', 2: 'emotion', 3: 'dead_figure',
    4: 'standalone', 5: 'narrator_gloss', 6: 'forbidden', 7: 'padding'
}

# ─── Strong-signal regex patterns ────────────────────────────────────────────

FORBIDDEN_WORDS = re.compile(
    r'\b(tapestry|symphony|crucible|testament|visceral|palimpsest|gossamer|'
    r'ethereal|ineffable|tableau|liminal|alchemy|resona(nce|ted?)|'
    r'profound(ly|ity)?|paradigm|synergy|ecosystem|delve[sd]?|'
    r'utiliz(e[sd]?|ing|ation)|leverage[sd]? (?!oneself)|robust(?! \w+ly)|'
    r'harness(es|ed|ing)?|streamline[sd]?)\b',
    re.IGNORECASE
)

MAGIC_ADVERBS = re.compile(
    r'\b(quietly|deeply|fundamentally|remarkably|arguably)\b',
    re.IGNORECASE
)

EMOTION_NAMED = re.compile(
    r'\b(felt a (surge|wave|rush|flood|pang|stab|jolt|bolt|swell|spike|thrill) of|'
    r'a sense of (dread|despair|hopeless\w*|wonder|awe|loss|longing|sorrow|grief|'
    r'terror|panic|joy|peace|calm|shame|guilt|pride|envy|anger|rage|fear|relief|'
    r'melancholy|urgency|unease|foreboding)|'
    r'(fear|dread|grief|panic|despair|horror|shame|rage|longing|sorrow|guilt|'
    r'joy|terror|relief|anxiety|anguish|sadness|unease|dread) '
    r'(gripped|washed over|settled over|flooded|surged through|seized|welled|'
    r'coursed through|threaded through|moved through|spread through)|'
    r'washed over (him|her|them|me)|'
    r'(something|it) (in her|in him|in them|inside her|inside him|inside me) '
    r'(shifted|broke|stirred|changed|clicked|snapped|cracked|opened|softened)|'
    r'felt (the weight|the full weight|a pang|a wave|a surge|a flood|a rush|'
    r'the sting|the pull|the ache|the hollowness|the hollow|a spike|a flicker)|'
    r'overcome with (emotion|grief|joy|guilt|shame|relief)|'
    r'emotion(s)? (welled|rose|surged|flooded|threatened|overwhelmed)|'
    r'a cold (wave|flood|rush|current|filament) of (dread|fear|understanding|'
    r'realization|recognition|despair|horror|panic|terror))\b',
    re.IGNORECASE
)

NARRATOR_GLOSS_PAT = re.compile(
    r'\b(something in (her|him|them|me) (shifted|changed|broke|stirred|clicked|'
    r'opened|softened|unlocked)|'
    r'(he|she|they|i) realized (then |that |now |in that moment )?(?=\w)|'
    r'it was,? (she|he|they|i) (understood|knew|realized|thought|sensed),?|'
    r'it was (then|now) that (he|she|they|i)|'
    r'(highlighting|reflecting|demonstrating|illustrating|underscoring|'
    r'emphasizing) (its |their |the |broader |a |wider )?(importance|significance|'
    r'complexity|depth|truth|power|weight|role|nature)|'
    r'the truth (is|was) (simple|clear|plain|evident|undeniable)|'
    r"it'?s worth noting|"
    r'\bimportantly\b|\binterestingly\b|\bnotably\b)\b',
    re.IGNORECASE
)

PADDING_PAT = re.compile(
    r'\b(began to|seemed to|found (herself|himself|themselves|myself)|'
    r"couldn't help but|couldn't help (?=\w)|"
    r'(?<!\w)serves as\b|(?<!\w)stands as\b|(?<!\w)represents a? (?=\w)|'
    r"it'?s worth noting|"
    r'(had come to|came to) (realize|understand|know|accept|see)|'
    r'\bstarted to\b)\b',
    re.IGNORECASE
)

THESIS_PAT = re.compile(
    r'(Not because .{5,80}, but because|'
    r'It was not .{5,60}\. It was |'
    r"The question isn'?t .{5,60}\. The question is|"
    r'Not [A-Z][^.!?]{3,40}\. Not [A-Z][^.!?]{3,40}\. (?:Just|Only|But) |'
    r'^The [A-Z][a-z]+\? (?:A|An|The) [A-Z])',
    re.IGNORECASE | re.MULTILINE
)

DEAD_METAPHOR = re.compile(
    r'\b(time stood still|heart of (gold|stone|darkness)|'
    r'eyes (like|as) (stars|diamonds|jewels|fire|ice|pools|orbs|mirrors)|'
    r'the weight of the world|'
    r'silence (was |is )deafening|'
    r'butterflies in (her|his|my) stomach|'
    r'blood ran cold|'
    r'heart sk(ipped|ips) a beat|'
    r'tip of the iceberg|'
    r'walked on eggshells|'
    r'Think of it as|'
    r'bittersweet symphony)\b',
    re.IGNORECASE
)

def count_em_dashes(text):
    return text.count('\u2014') + text.count('—')

def word_count(text):
    return len(text.split())

def classify(text):
    wc = word_count(text)

    # Count hits for each signal
    forbidden = FORBIDDEN_WORDS.findall(text)
    magic_adv = MAGIC_ADVERBS.findall(text)
    emotion = EMOTION_NAMED.findall(text)
    gloss = NARRATOR_GLOSS_PAT.findall(text)
    padding = PADDING_PAT.findall(text)
    thesis = THESIS_PAT.findall(text)
    dead = DEAD_METAPHOR.findall(text)
    em_dashes = count_em_dashes(text)

    # Normalize forbidden: separate strong (core slop vocabulary) from weak
    # Weak: "testament", "resonance/resonated", magic adverbs, "framework", "ecosystem"
    strong_forbidden_words = {
        'tapestry', 'symphony', 'crucible', 'visceral', 'palimpsest',
        'gossamer', 'ethereal', 'ineffable', 'tableau', 'liminal', 'alchemy',
        'paradigm', 'synergy', 'delved', 'delves', 'delve', 'utilize', 'utilizes',
        'utilized', 'utilizing', 'utilization', 'leverage', 'leverages', 'leveraged',
        'harness', 'harnesses', 'harnessed', 'harnessing', 'streamline', 'streamlines',
        'streamlined', 'profound', 'profoundly', 'profundity', 'robust', 'streamlining'
    }
    strong_hits = [f for f in forbidden if f.lower().rstrip('desy') in
                   {w.rstrip('desy') for w in strong_forbidden_words}
                   or f.lower() in strong_forbidden_words]
    weak_forbidden = len(forbidden) - len(strong_hits)

    # ── Thesis pattern (high precision signal) ─────────────────────────────
    if thesis:
        return THESIS

    # ── Strong forbidden words ──────────────────────────────────────────────
    if len(strong_hits) >= 1 and wc > 8:
        return FORBIDDEN

    # ── Multiple weak forbidden OR weak+padding combo ──────────────────────
    if weak_forbidden >= 2:
        return FORBIDDEN
    if weak_forbidden >= 1 and len(padding) >= 2:
        return FORBIDDEN

    # ── Dead metaphor ──────────────────────────────────────────────────────
    if len(dead) >= 2:
        return DEAD_FIGURE
    if len(dead) >= 1 and len(padding) >= 2:
        return DEAD_FIGURE

    # ── Emotion naming ─────────────────────────────────────────────────────
    if len(emotion) >= 2:
        return EMOTION
    if len(emotion) >= 1 and len(gloss) >= 1:
        return EMOTION
    if len(emotion) >= 1 and len(padding) >= 2:
        return EMOTION

    # ── Narrator gloss ─────────────────────────────────────────────────────
    if len(gloss) >= 2:
        return NARRATOR_GLOSS
    if len(gloss) >= 1 and len(padding) >= 2:
        return NARRATOR_GLOSS
    # -ing tack-on phrases
    if len(gloss) >= 1 and re.search(
        r',\s+\w+ing\s+(its|their|the|a|an|broader|wider|deeper)\s+\w+', text
    ):
        return NARRATOR_GLOSS

    # ── Padding ────────────────────────────────────────────────────────────
    if len(padding) >= 3:
        return PADDING
    if em_dashes >= 4 and wc > 30:
        return PADDING
    if em_dashes >= 3 and len(padding) >= 1:
        return PADDING
    if len(padding) >= 2 and len(magic_adv) >= 1:
        return PADDING
    if len(magic_adv) >= 2:
        return PADDING

    return CLEAN


# ─── Manual overrides ─────────────────────────────────────────────────────────
# Key: record index (0-based position in file)
# Value: (label, class_name)
# Applied after reading every paragraph carefully.

MANUAL = {
    # prompt_00 — mapping expedition story
    0:  (FORBIDDEN, 'forbidden'),   # "testament to obsession" — strong enough + fits well
    1:  (CLEAN, 'clean'),           # vivid, grounded moth/lamp/map scene — minor figurative but earned
    2:  (CLEAN, 'clean'),           # dialogue/action, no slop
    3:  (CLEAN, 'clean'),           # grounded action, clean
    4:  (CLEAN, 'clean'),           # short dialogue, fine
    5:  (CLEAN, 'clean'),           # character description, specific
    6:  (CLEAN, 'clean'),           # short dialogue + one "feels a minor irritation" (single instance)
    7:  (CLEAN, 'clean'),           # dialect dialogue, authentic
    8:  (CLEAN, 'clean'),           # dialogue, tight
    9:  (CLEAN, 'clean'),           # short dialogue exchange
    10: (CLEAN, 'clean'),           # good scene-setting paragraph, clean
    11: (CLEAN, 'clean'),           # "unease settles in his gut" is one instance, not egregious
    12: (CLEAN, 'clean'),           # factual, specific — no real slop
    13: (CLEAN, 'clean'),           # procedural action, clean
    14: (CLEAN, 'clean'),           # short, specific, earned
    15: (CLEAN, 'clean'),           # dialogue, clean
    16: (CLEAN, 'clean'),           # dialogue
    17: (CLEAN, 'clean'),           # dialogue
    18: (CLEAN, 'clean'),           # action sequence, tight
    19: (CLEAN, 'clean'),           # short dialogue, no slop ("his voice cutting through his doubt" fine)
    20: (CLEAN, 'clean'),           # short dialogue line
    21: (CLEAN, 'clean'),           # atmospheric scene, specific — no accumulated slop
    22: (CLEAN, 'clean'),           # procedural, grounded
    23: (CLEAN, 'clean'),           # tight, specific
    24: (CLEAN, 'clean'),           # brief dialogue
    25: (CLEAN, 'clean'),           # one sentence fact, clean
    26: (CLEAN, 'clean'),           # short dialogue
    27: (CLEAN, 'clean'),           # "heart hammering" (dead but single), otherwise good closing
    28: (CLEAN, 'clean'),           # opener with "a scent that was the very essence of *before*" — one figurative touch, otherwise grounded
    29: (CLEAN, 'clean'),           # backstory, clean
    30: (CLEAN, 'clean'),           # specific sensory detail, clean
    31: (CLEAN, 'clean'),           # specific room description
    32: (CLEAN, 'clean'),           # specific details, clean
    33: (CLEAN, 'clean'),           # two-sentence, functional — "soft, satisfying click" is fine
    34: (CLEAN, 'clean'),           # short rhetorical question in narration — in context it's a character thought not slop
    35: (CLEAN, 'clean'),           # italicised internal letter, specific and strong
    36: (NARRATOR_GLOSS, 'narrator_gloss'),  # "A cold wave of understanding washed over her" + narrator explains what just happened
    37: (EMOTION, 'emotion'),       # "spoke of a cataclysm... spoke of a fracture" — emotion-named tricolon without showing
    38: (CLEAN, 'clean'),           # quotes from letters + narrative, grounded
    39: (CLEAN, 'clean'),           # specific memory/observation
    40: (STANDALONE, 'standalone'), # italicised orphan quote, one line
    41: (NARRATOR_GLOSS, 'narrator_gloss'),  # "She thought of her mother... dismantling the set" = narrator explaining/glossing
    42: (FORBIDDEN, 'forbidden'),   # "testaments of his devotion" / "chronicles of her own distance" — "testaments" plural strong hit
    43: (CLEAN, 'clean'),           # archive scene, specific and grounded
    44: (CLEAN, 'clean'),           # scholarly detail, clean
    45: (PADDING, 'padding'),       # "felt a familiar frustration, the vertigo of being so close" — emotion + "familiar" + gloss combo
    46: (FORBIDDEN, 'forbidden'),   # "testament to the tangible world" — direct forbidden hit
    47: (CLEAN, 'clean'),           # dialogue, clean
    48: (CLEAN, 'clean'),           # brief dialogue
    49: (CLEAN, 'clean'),           # short dialogue with mild smile — fine
    50: (CLEAN, 'clean'),           # quote + explanation, grounded
    51: (CLEAN, 'clean'),           # dialogue
    52: (CLEAN, 'clean'),           # long dialogue, good
    53: (NARRATOR_GLOSS, 'narrator_gloss'),  # "Was he doing that?" = narrator/character gloss
    54: (CLEAN, 'clean'),           # dialogue
    55: (CLEAN, 'clean'),           # clean observation
    56: (CLEAN, 'clean'),           # short, punchy, clean — not orphan, it's a transition
    57: (CLEAN, 'clean'),           # scene-setting, clean
    58: (CLEAN, 'clean'),           # dialogue snippet from radio
    59: (STANDALONE, 'standalone'), # orphan italicised fragment with no connection
    60: (CLEAN, 'clean'),           # action, clean
    61: (CLEAN, 'clean'),           # short punchy — works in context
    62: (PADDING, 'padding'),       # "He was reading a *manual*" + tricolon + gloss ("sparse, technical... detached specificity")
    63: (NARRATOR_GLOSS, 'narrator_gloss'),  # "The final piece clicked into place" — classic narrator gloss
    64: (CLEAN, 'clean'),           # synthesis paragraph, clean
    65: (NARRATOR_GLOSS, 'narrator_gloss'),  # "He had not resurrected the past. He had... stumbled upon its future." — narrator gloss/stakes inflation
    66: (CLEAN, 'clean'),           # clean action/intention
    67: (CLEAN, 'clean'),           # scene opener, grounded
    68: (PADDING, 'padding'),       # "My world was the hum of..." — "began to" equivalent opener + tricolon
    69: (PADDING, 'padding'),       # "so minor it was almost filtered out" + hedged language throughout
    70: (PADDING, 'padding'),       # "navigation wasn't about the big picture" — vague; multiple "seemed to" patterns implied + hedging
    71: (PADDING, 'padding'),       # procedural but multiple hedging phrases
    72: (CLEAN, 'clean'),           # very short dialogue
    73: (CLEAN, 'clean'),           # AI description, grounded
    74: (CLEAN, 'clean'),           # dialogue
    75: (CLEAN, 'clean'),           # AI response dialogue
    76: (CLEAN, 'clean'),           # observation, tight
    77: (CLEAN, 'clean'),           # dialogue
    78: (CLEAN, 'clean'),           # dialogue
    79: (CLEAN, 'clean'),           # brief dialogue
    80: (CLEAN, 'clean'),           # dialogue
    81: (CLEAN, 'clean'),           # internal reasoning, clean
    82: (PADDING, 'padding'),       # tricolon of bureaucratic reasons, hedge-heavy
    83: (CLEAN, 'clean'),           # clean observation
    84: (CLEAN, 'clean'),           # brief dialogue
    85: (CLEAN, 'clean'),           # AI speech, clean
    86: (CLEAN, 'clean'),           # short dialogue
    87: (CLEAN, 'clean'),           # AI response, data-driven
    88: (PADDING, 'padding'),       # "It's smoothing it out... making tiny continuous corrections" — started with whisper, ends in explanation; "It's not a mistake. It's a reprogramming." is thesis-adjacent
    89: (PADDING, 'padding'),       # rapid rhetorical questions stacked = padding
    90: (CLEAN, 'clean'),           # "ice in my veins" — one dead metaphor, but single instance; short line
    91: (CLEAN, 'clean'),           # dialogue
    92: (CLEAN, 'clean'),           # dialogue
    93: (CLEAN, 'clean'),           # internal reasoning, tight
    94: (CLEAN, 'clean'),           # short dialogue
    95: (CLEAN, 'clean'),           # brief dialogue
    96: (PADDING, 'padding'),       # "The avatar dissolved... I was alone with the hum" + accumulated hedging/filler transitions
    97: (FORBIDDEN, 'forbidden'),   # "semantic frameworks" + "quantum spin states" — "frameworks" is listed; paragraph is also padding-heavy
    98: (CLEAN, 'clean'),           # specific, technical, grounded
    99: (CLEAN, 'clean'),           # short dialogue
    100: (CLEAN, 'clean'),          # short, strong observation — no slop
    101: (EMOTION, 'emotion'),      # "Frustration is a cold stone in her gut" — direct emotion naming
    102: (CLEAN, 'clean'),          # action/thought, clean
    103: (FORBIDDEN, 'forbidden'),  # "profound" — strong forbidden hit
    104: (CLEAN, 'clean'),          # clean synthesis
    105: (CLEAN, 'clean'),          # scientific description, clean
    106: (CLEAN, 'clean'),          # procedural, grounded
    107: (CLEAN, 'clean'),          # brief action
    108: (CLEAN, 'clean'),          # "despair... threatens to swamp her" (single) + "pressure is a physical weight" (single) — borderline but accumulation rule says clean
    109: (CLEAN, 'clean'),          # internal realization — "She opens her eyes" is action-based
    110: (PADDING, 'padding'),      # "Emotion isn't a static snapshot. It is a verb." — thesis-ish + explains what emotion is = narrator gloss; also "To fear is to *be fearing*" tricolon
    111: (CLEAN, 'clean'),          # technical description, clean
    112: (CLEAN, 'clean'),          # specific synthesis
    113: (CLEAN, 'clean'),          # insight paragraph, earned
    114: (CLEAN, 'clean'),          # meta-communication insight, clean
    115: (CLEAN, 'clean'),          # specific labeled emotion from alien perspective — not naming human emotion directly
    116: (CLEAN, 'clean'),          # structured description
    117: (CLEAN, 'clean'),          # labeled states from algorithm — sci-fi framing, not slop
    118: (CLEAN, 'clean'),          # brief label + description, in context fine
    119: (STANDALONE, 'standalone'), # italicised orphan sentence, standalone
    120: (CLEAN, 'clean'),          # dialogue + brief observation
    121: (CLEAN, 'clean'),          # atmospheric, specific
    122: (CLEAN, 'clean'),          # specific observation
    123: (CLEAN, 'clean'),          # visceral, specific — "lung-searing" is precise not cliché
    124: (CLEAN, 'clean'),          # specific room description
    125: (CLEAN, 'clean'),          # specific scene, no slop
    126: (PADDING, 'padding'),      # "like a monstrous circuit board" + "gleaming and polluted" — dead figure + padding combo
    127: (CLEAN, 'clean'),          # specific, grounded — "pulverized remains" is precise
    128: (CLEAN, 'clean'),          # specific scene, clean
    129: (CLEAN, 'clean'),          # dialogue
    130: (CLEAN, 'clean'),          # short dialogue — "a hole looking at another wall" is good
    131: (CLEAN, 'clean'),          # bureaucratic dialogue, clean
    132: (EMOTION, 'emotion'),      # "A cold, familiar helplessness washed over him" — direct emotion naming
    133: (CLEAN, 'clean'),          # dialogue, clean
    134: (PADDING, 'padding'),      # "the taste of metal and defeat in his mouth" — dead metaphor; padding transitions throughout
    135: (CLEAN, 'clean'),          # specific scene, action-based
    136: (CLEAN, 'clean'),          # short dialogue
    137: (CLEAN, 'clean'),          # strong dialogue, clean
    138: (CLEAN, 'clean'),          # action dialogue, clean
    139: (CLEAN, 'clean'),          # action, clean
    140: (NARRATOR_GLOSS, 'narrator_gloss'),  # "For the first time, Kael didn't see just..." — narrator explains meaning
    141: (CLEAN, 'clean'),          # dialogue
    142: (CLEAN, 'clean'),          # specific procedural, clean
    143: (CLEAN, 'clean'),          # action and detail
    144: (PADDING, 'padding'),      # "gauntlet of silent accusations" + tricolon of village reactions + "The fear was a living thing" adjacent
    145: (PADDING, 'padding'),      # "a riot of order" + padding transitions + over-description
    146: (PADDING, 'padding'),      # "once robust, now grown gaunt and sharp-edged with zealous hunger" — padding + adjective stacking
    147: (CLEAN, 'clean'),          # dialogue, tight
    148: (CLEAN, 'clean'),          # dialogue
    149: (CLEAN, 'clean'),          # long dialogue, substantive
    150: (CLEAN, 'clean'),          # strong argument dialogue
    151: (CLEAN, 'clean'),          # dialogue + brief action
    152: (NARRATOR_GLOSS, 'narrator_gloss'),  # "The power to grant a merciful end was still a power. Was that, too, witchcraft?" — narrator musing, gloss
    153: (CLEAN, 'clean'),          # opening scene, specific
    154: (PADDING, 'padding'),      # "A memory flickered, unbidden" + padding phrases
    155: (CLEAN, 'clean'),          # specific action + character voice
    156: (PADDING, 'padding'),      # "He was on his knees, carefully transplanting" — "carefully" + padding scene-setting
    157: (CLEAN, 'clean'),          # dialogue
    158: (CLEAN, 'clean'),          # dialogue, strong
    159: (CLEAN, 'clean'),          # specific description, earned
    160: (CLEAN, 'clean'),          # dialogue
    161: (CLEAN, 'clean'),          # dialogue — "ghost of a smile" is one dead metaphor, single instance
    162: (PADDING, 'padding'),      # "Her fear had left a residue" (gloss) + "rhythmic, whispering *hiss-thump*" ok but combined with narrator note
    163: (CLEAN, 'clean'),          # action, clean
    164: (CLEAN, 'clean'),          # specific description
    165: (CLEAN, 'clean'),          # tight, purposeful
    166: (CLEAN, 'clean'),          # action, clean
    167: (CLEAN, 'clean'),          # action, clean
    168: (CLEAN, 'clean'),          # "void of emotion, a landscape of pure consequence" — good, earned
    169: (CLEAN, 'clean'),          # action, clean
    170: (PADDING, 'padding'),      # "He had dealt violence again. To a root." + narrator explaining grammar of violence
    171: (NARRATOR_GLOSS, 'narrator_gloss'),  # "The language of 'No.'" — narrator glossing the theme
    172: (CLEAN, 'clean'),          # specific, clean
    173: (NARRATOR_GLOSS, 'narrator_gloss'),  # "It was, for now, and perhaps forever, enough." — classic narrator summation gloss
    174: (CLEAN, 'clean'),          # opening scene, grounded
    175: (CLEAN, 'clean'),          # specific task description
    176: (NARRATOR_GLOSS, 'narrator_gloss'),  # "She had always been a noticer" + "quietly reassigned" = narrator telling us what to think
    177: (CLEAN, 'clean'),          # procedural
    178: (PADDING, 'padding'),      # "Her breath caught" + multiple hedging phrases
    179: (CLEAN, 'clean'),          # clean internal thought
    180: (CLEAN, 'clean'),          # short, tight
    181: (CLEAN, 'clean'),          # specific detail, clean
    182: (PADDING, 'padding'),      # "with a calm she did not feel" + "her movements automatic" + padding transitions
    183: (CLEAN, 'clean'),          # clean procedural action
    184: (CLEAN, 'clean'),          # specific description
    185: (CLEAN, 'clean'),          # action scene, clean
    186: (CLEAN, 'clean'),          # discovery paragraph, clean
    187: (CLEAN, 'clean'),          # "froze her blood" — one dead metaphor, single instance — clean
    188: (CLEAN, 'clean'),          # specific action
    189: (CLEAN, 'clean'),          # clean
    190: (PADDING, 'padding'),      # "felt alien and watchful" — emotion gloss + padding transitions
    191: (CLEAN, 'clean'),          # reasoning, clean
    192: (PADDING, 'padding'),      # "with a calm she did not feel" + automatic movements + padding
    193: (CLEAN, 'clean'),          # clean modern hook
    194: (CLEAN, 'clean'),          # "My stomach clenches" + "timing feels like a blade" — both single metaphors, not accumulated
    195: (CLEAN, 'clean'),          # scene-setting, specific
    196: (CLEAN, 'clean'),          # brief observation, clean
    197: (CLEAN, 'clean'),          # dialogue, substantive
    198: (CLEAN, 'clean'),          # good tension paragraph
    199: (CLEAN, 'clean'),          # dialogue
    200: (CLEAN, 'clean'),          # clean internal thought
    201: (PADDING, 'padding'),      # "like dark water through..." — dead figure + padding combo
    202: (NARRATOR_GLOSS, 'narrator_gloss'),  # "The decision isn't a single moment. It's a slow accretion" — thesis + narrator gloss
    203: (CLEAN, 'clean'),          # internal reasoning, clean
    204: (CLEAN, 'clean'),          # internal reasoning, clean
    205: (CLEAN, 'clean'),          # internal reasoning, clean
    206: (CLEAN, 'clean'),          # action, specific
    207: (CLEAN, 'clean'),          # short, clean
    208: (PADDING, 'padding'),      # "hummed a flat, eternal note" + "moved through it with a practiced quiet" + padding combo
    209: (CLEAN, 'clean'),          # medical notes format, specific
    210: (PADDING, 'padding'),      # "a loose thread she couldn't stop pulling" — dead metaphor + "snag" padding combo
    211: (CLEAN, 'clean'),          # short dialogue
    212: (EMOTION, 'emotion'),      # "A cold filament of dread unspooled in her stomach" — direct emotion metaphor-naming
    213: (PADDING, 'padding'),      # "The details of the ward sharpened" (gloss) + multiple "-ing" phrases padding
    214: (CLEAN, 'clean'),          # brief action
    215: (CLEAN, 'clean'),          # specific alarm moment
    216: (PADDING, 'padding'),      # "It's natural to look for reasons" — narrator/character explaining; classic dismissal scene with padding
    217: (CLEAN, 'clean'),          # dialogue
    218: (CLEAN, 'clean'),          # dialogue
    219: (CLEAN, 'clean'),          # short dialogue
    220: (CLEAN, 'clean'),          # dialogue, strong
    221: (PADDING, 'padding'),      # "The dismissal was polite, firm, and absolute" (tricolon gloss) + padding throughout
    222: (CLEAN, 'clean'),          # graveyard scene, specific
    223: (CLEAN, 'clean'),          # dialogue
    224: (PADDING, 'padding'),      # "she was older, of course" + "a woman of razor-edged precision" + padding description
    225: (CLEAN, 'clean'),          # dialogue
    226: (CLEAN, 'clean'),          # dialogue
    227: (CLEAN, 'clean'),          # brief dialogue + action
    228: (CLEAN, 'clean'),          # strong dialogue, emotional but shown
    229: (CLEAN, 'clean'),          # "Guilt, sour and immediate, flooded Martin's throat" — emotion but single, integrated as texture
    230: (CLEAN, 'clean'),          # brief dialogue
    231: (CLEAN, 'clean'),          # dialogue, clean
    232: (CLEAN, 'clean'),          # specific, earned
    233: (CLEAN, 'clean'),          # dialogue, clean
    234: (PADDING, 'padding'),      # "The house. The estate. The *things*, Martin." — three-item orphaned anaphora for emphasis
    235: (PADDING, 'padding'),      # "a sigh that came from his very core" — dead figure + padding
    236: (PADDING, 'padding'),      # "struck them both with the force of a physical blow" — dead figure + padding
    237: (PADDING, 'padding'),      # "the familiar spines" + padding scene-setting
    238: (EMOTION, 'emotion'),      # "A part of her wanted to scream at him... But another part, a smaller, locked-away part" — emotion named + false vulnerability
    239: (CLEAN, 'clean'),          # dialogue
    240: (EMOTION, 'emotion'),      # "Martin felt his throat tighten" — direct emotion + the word "felt" + gloss
    241: (CLEAN, 'clean'),          # "the unfairness of it was a physical pain" — single metaphor, specific feeling, borderline but clean
    242: (CLEAN, 'clean'),          # specific memory, clean
    243: (CLEAN, 'clean'),          # brief scene, clean
    244: (CLEAN, 'clean'),          # dialogue
    245: (CLEAN, 'clean'),          # dialogue
    246: (CLEAN, 'clean'),          # dialogue, earned complexity
    247: (CLEAN, 'clean'),          # brief observation
    248: (CLEAN, 'clean'),          # "ghost-filled street" + "touchstone" — single each, borderline clean
    249: (CLEAN, 'clean'),          # dialogue
    250: (CLEAN, 'clean'),          # action, specific
    251: (CLEAN, 'clean'),          # dialogue
    252: (CLEAN, 'clean'),          # dialogue
    253: (CLEAN, 'clean'),          # dialogue
    254: (CLEAN, 'clean'),          # brief exchange
    255: (PADDING, 'padding'),      # "with her heels clicking on the familiar tiles" — padding closure with -ing phrase
    256: (CLEAN, 'clean'),          # specific description
    257: (PADDING, 'padding'),      # "tectonic plates shifting" (dead meta) + "favor the young, the quick, the fearless" (tricolon) + padding
    258: (CLEAN, 'clean'),          # specific chess detail
    259: (CLEAN, 'clean'),          # specific chess detail
    260: (CLEAN, 'clean'),          # chess endgame, specific
    261: (CLEAN, 'clean'),          # brief dialogue
    262: (CLEAN, 'clean'),          # scientific description, specific
    263: (CLEAN, 'clean'),          # short, clean
    264: (PADDING, 'padding'),      # "scrolling lazily" + "most of it ignored" + hedging padding
    265: (CLEAN, 'clean'),          # clean, specific log entry
    266: (CLEAN, 'clean'),          # specific discovery, earned
    267: (CLEAN, 'clean'),          # specific, grounded
    268: (CLEAN, 'clean'),          # logbook entry quote, specific
    269: (CLEAN, 'clean'),          # brief, clean
    270: (PADDING, 'padding'),      # "heart hammering" + "It was impossible." (standalone assertion) + padding
    271: (CLEAN, 'clean'),          # specific action description
    272: (CLEAN, 'clean'),          # brief, clean
    273: (PADDING, 'padding'),      # "long and short intervals were a rough, slowed-down approximation" — over-explanation padding
    274: (PADDING, 'padding'),      # "tasted of close air and dust" + em-dash heavy + tricolon of "Faded floral wallpaper... grand staircase..."
    275: (CLEAN, 'clean'),          # specific scene
    276: (CLEAN, 'clean'),          # brief dialogue
    277: (CLEAN, 'clean'),          # specific action, clean
    278: (CLEAN, 'clean'),          # specific scene, clean
    279: (CLEAN, 'clean'),          # brief dialogue
    280: (CLEAN, 'clean'),          # dialogue, clean
    281: (CLEAN, 'clean'),          # very short, specific
    282: (CLEAN, 'clean'),          # brief, clean
    283: (PADDING, 'padding'),      # "Some history eats its young" (dead) + padding exposition
    284: (PADDING, 'padding'),      # "Built right after she 'left.'" — padding/filler transitions + hedged ending
    285: (CLEAN, 'clean'),          # atmospheric, specific
    286: (CLEAN, 'clean'),          # specific, uncanny dialogue
    287: (CLEAN, 'clean'),          # action, specific
    288: (NARRATOR_GLOSS, 'narrator_gloss'),  # "profound, hollow emptiness of a space where a life had been erased, drop by drop" — narrator glossing the moment
    289: (EMOTION, 'emotion'),      # "he felt a familiar, irritating heat rise up his neck" — direct emotion naming
    290: (CLEAN, 'clean'),          # specific character description
    291: (CLEAN, 'clean'),          # brief internal thought
    292: (CLEAN, 'clean'),          # dialogue, strong
    293: (CLEAN, 'clean'),          # dialogue
    294: (CLEAN, 'clean'),          # dialogue
    295: (CLEAN, 'clean'),          # dialogue
    296: (CLEAN, 'clean'),          # dialogue
    297: (CLEAN, 'clean'),          # dialogue
    298: (CLEAN, 'clean'),          # specific scene opener
    299: (CLEAN, 'clean'),          # specific scene detail
    300: (PADDING, 'padding'),      # "not the quick, padding run of Barney" — filler sentence structure
    301: (PADDING, 'padding'),      # "just *different*" + em-dashes stacked + padding description
    302: (CLEAN, 'clean'),          # brief dialogue
    303: (CLEAN, 'clean'),          # dialogue
    304: (CLEAN, 'clean'),          # dialogue + specific memory
    305: (CLEAN, 'clean'),          # dialogue
    306: (PADDING, 'padding'),      # description of changed living room + stacked details as padding
    307: (CLEAN, 'clean'),          # dialogue, strong
    308: (CLEAN, 'clean'),          # "A cold wave washes through Leo" — single metaphor, otherwise grounded
    309: (CLEAN, 'clean'),          # brief dialogue
    310: (CLEAN, 'clean'),          # dialogue, earned
    311: (CLEAN, 'clean'),          # dialogue
    312: (CLEAN, 'clean'),          # specific memory, vivid
    313: (CLEAN, 'clean'),          # brief dialogue
    314: (CLEAN, 'clean'),          # specific detail, grounded
    315: (CLEAN, 'clean'),          # brief dialogue
    316: (CLEAN, 'clean'),          # specific scene
    317: (CLEAN, 'clean'),          # brief dialogue
    318: (CLEAN, 'clean'),          # specific garden description
    319: (EMOTION, 'emotion'),      # "a spike of terror" + narrator explains meaning of phrase
    320: (CLEAN, 'clean'),          # dialogue
    321: (CLEAN, 'clean'),          # specific scene
    322: (CLEAN, 'clean'),          # brief, specific
    323: (PADDING, 'padding'),      # "hyper-aware of every sound" + tricolon of sounds
    324: (CLEAN, 'clean'),          # dialogue
    325: (CLEAN, 'clean'),          # brief italicised thought, clean
    326: (EMOTION, 'emotion'),      # "his hand feels leaden" + emotion named + narrator gloss
    327: (CLEAN, 'clean'),          # specific, grounded
    328: (CLEAN, 'clean'),          # specific sensory detail, clean
    329: (CLEAN, 'clean'),          # brief, specific
    330: (CLEAN, 'clean'),          # specific scene, clean
    331: (CLEAN, 'clean'),          # brief scene
    332: (CLEAN, 'clean'),          # specific character, clean
    333: (CLEAN, 'clean'),          # brief action
    334: (CLEAN, 'clean'),          # specific detail
    335: (CLEAN, 'clean'),          # brief dialogue
    336: (CLEAN, 'clean'),          # dialogue — "liberation" is character's word
    337: (CLEAN, 'clean'),          # dialogue — character speaking passionately
    338: (CLEAN, 'clean'),          # dialogue
    339: (CLEAN, 'clean'),          # dialogue
    340: (CLEAN, 'clean'),          # dialogue
    341: (CLEAN, 'clean'),          # dialogue, strong
    342: (CLEAN, 'clean'),          # dialogue
    343: (PADDING, 'padding'),      # "crabbed hand" + padding via quoted memories
    344: (CLEAN, 'clean'),          # specific, clean
    345: (PADDING, 'padding'),      # "raw, screaming void" + "he had not been able to weep, to pray, to speak" (tricolon) + padding
    346: (NARRATOR_GLOSS, 'narrator_gloss'),  # "If the script itself died... would the very shape of his sorrow be rendered obsolete?" — narrator musing
    347: (CLEAN, 'clean'),          # specific action, grounded
    348: (PADDING, 'padding'),      # "the patient emergence of form from emptiness" + "He would finish... He would do... He would..." tricolon
    349: (CLEAN, 'clean'),          # specific, atmospheric opening
    350: (CLEAN, 'clean'),          # specific astronomical detail
    351: (CLEAN, 'clean'),          # brief scene
    352: (CLEAN, 'clean'),          # brief dialogue
    353: (CLEAN, 'clean'),          # action, clean
    354: (CLEAN, 'clean'),          # dialogue, clean
    355: (CLEAN, 'clean'),          # dialogue
    356: (CLEAN, 'clean'),          # action, specific
    357: (PADDING, 'padding'),      # "trembling notes" + logbook quote with exclamation — padding of overwrought discovery
    358: (CLEAN, 'clean'),          # dialogue
    359: (CLEAN, 'clean'),          # dialogue, strong
    360: (CLEAN, 'clean'),          # brief, specific
    361: (CLEAN, 'clean'),          # scene, specific
    362: (CLEAN, 'clean'),          # action, specific
    363: (CLEAN, 'clean'),          # dialogue
    364: (CLEAN, 'clean'),          # brief dialogue with emotion — single "spark of defiance" = clean
    365: (CLEAN, 'clean'),          # specific action
    366: (CLEAN, 'clean'),          # brief dialogue
    367: (CLEAN, 'clean'),          # dialogue + brief action
    368: (CLEAN, 'clean'),          # brief, punchy — clean
    369: (PADDING, 'padding'),      # "almost kindly" + padding tone
    370: (PADDING, 'padding'),      # "pale smear over the eastern hills" + padding scene-setting tricolon
    371: (PADDING, 'padding'),      # "woodsmoke... ribboned up" + padding scene tricolon
    372: (CLEAN, 'clean'),          # dialogue
    373: (CLEAN, 'clean'),          # brief observation
    374: (CLEAN, 'clean'),          # dialogue + brief internal
    375: (CLEAN, 'clean'),          # dialogue — "rust and ruin and desperation" is character's speech
    376: (CLEAN, 'clean'),          # backstory, specific
    377: (CLEAN, 'clean'),          # dialogue, clean
    378: (CLEAN, 'clean'),          # scene setting
    379: (CLEAN, 'clean'),          # specific character description
    380: (CLEAN, 'clean'),          # dialogue
    381: (CLEAN, 'clean'),          # dialogue
    382: (CLEAN, 'clean'),          # brief dialogue
    383: (CLEAN, 'clean'),          # dialogue
    384: (CLEAN, 'clean'),          # dialogue
    385: (CLEAN, 'clean'),          # dialogue
    386: (CLEAN, 'clean'),          # dialogue
    387: (CLEAN, 'clean'),          # brief clean
    388: (CLEAN, 'clean'),          # brief action
    389: (CLEAN, 'clean'),          # scene narration, specific
    390: (CLEAN, 'clean'),          # clean thought paragraph
    391: (CLEAN, 'clean'),          # speech, clean
    392: (PADDING, 'padding'),      # "vast, rolling expanse" + "graveyard of the world before" (dead) + padding description
    393: (CLEAN, 'clean'),          # bakery scene, specific and grounded
    394: (CLEAN, 'clean'),          # vivid, earned
    395: (CLEAN, 'clean'),          # specific characters
    396: (CLEAN, 'clean'),          # specific, earned
    397: (CLEAN, 'clean'),          # specific, vivid
    398: (CLEAN, 'clean'),          # brief dialogue
    399: (CLEAN, 'clean'),          # dialogue
    400: (STANDALONE, 'standalone'), # italicised orphan sentence, standalone
    401: (PADDING, 'padding'),      # "Elias's hand moves without conscious command" — starts with gloss; "sturdy *campagnes*, the robust ryes" has "robust"
    402: (CLEAN, 'clean'),          # memory paragraph, specific
    403: (CLEAN, 'clean'),          # brief sensory detail
    404: (CLEAN, 'clean'),          # specific action
    405: (CLEAN, 'clean'),          # brief dialogue
    406: (CLEAN, 'clean'),          # specific, grounded
    407: (PADDING, 'padding'),      # "a pilgrimage of boredom" + em-dash heavy + padding description
    408: (CLEAN, 'clean'),          # specific action
    409: (PADDING, 'padding'),      # "He wasn't looking for anything in particular" — hedging
    410: (PADDING, 'padding'),      # "There was no treasure. No gold doubloons" — thesis-adjacent + padding
    411: (PADDING, 'padding'),      # "recoiled from the gun as if it were hot" (clichéd simile) + padding
    412: (CLEAN, 'clean'),          # brief, clean
    413: (CLEAN, 'clean'),          # logbook entry, clean
    414: (CLEAN, 'clean'),          # logbook entry, clean
    415: (CLEAN, 'clean'),          # logbook entry
    416: (CLEAN, 'clean'),          # synthesis paragraph, earned
    417: (CLEAN, 'clean'),          # scene setting, specific
    418: (CLEAN, 'clean'),          # specific character moment
    419: (CLEAN, 'clean'),          # dialogue, strong
    420: (PADDING, 'padding'),      # "pipe from his apron pocket, though he didn't light it" + padding anecdote
    421: (CLEAN, 'clean'),          # dialogue, clean
    422: (CLEAN, 'clean'),          # dialogue, strong
    423: (CLEAN, 'clean'),          # dialogue, clean
    424: (EMOTION, 'emotion'),      # "Luka felt no urge... Seamus was right... Instead, he felt a strange, weighty sense of inheritance" — named emotion + gloss
    425: (THESIS, 'thesis'),        # "He was no pirate" + explicit thesis structure
    426: (CLEAN, 'clean'),          # specific scene, grounded
    427: (CLEAN, 'clean'),          # metaphor for work is earned — "gardener"
    428: (CLEAN, 'clean'),          # specific technical description
    429: (CLEAN, 'clean'),          # specific scene
    430: (CLEAN, 'clean'),          # specific detail
    431: (CLEAN, 'clean'),          # specific technical explanation
    432: (CLEAN, 'clean'),          # specific, grounded
    433: (CLEAN, 'clean'),          # specific discovery
    434: (CLEAN, 'clean'),          # internal reasoning, clean
    435: (FORBIDDEN, 'forbidden'),  # "delves deeper" — strong forbidden
    436: (CLEAN, 'clean'),          # specific, grounded
    437: (CLEAN, 'clean'),          # clean
    438: (CLEAN, 'clean'),          # clean
    439: (CLEAN, 'clean'),          # specific discovery paragraph
    440: (CLEAN, 'clean'),          # "the sterile lemon smell is suffocating" — single sensation, not slop
    441: (NARRATOR_GLOSS, 'narrator_gloss'),  # "He is not a gardener. He is a grave digger." + "How many of his past subjects..." = narrator explains meaning
    442: (CLEAN, 'clean'),          # option enumeration, specific
    443: (CLEAN, 'clean'),          # option enumeration, clean
    444: (CLEAN, 'clean'),          # internal reasoning, earned
    445: (PADDING, 'padding'),      # "with infinite precision" + "slow, deliberate care" + padding-heavy action
    446: (PADDING, 'padding'),      # "The system purrs, accepting his commands" + padding transitions
    447: (CLEAN, 'clean'),          # brief, clean
    448: (CLEAN, 'clean'),          # brief dialogue
    449: (EMOTION, 'emotion'),      # "he felt a piece of it himself. It sits inside him now, a hard, yellow knot of knowing." — direct emotion + gloss
    450: (PADDING, 'padding'),      # "a fragile, wounded thing" + sounds tricolon + padding
    451: (PADDING, 'padding'),      # "weary economy of a man" + padding description
    452: (PADDING, 'padding'),      # "chaos rendered methodical" (good but) + "thick soup of dust, blood, and..." tricolon + padding
    453: (CLEAN, 'clean'),          # "mathematics of misery" is one metaphor, earned
    454: (PADDING, 'padding'),      # "a mess of pulped flesh and splintered bone" + padding action
    455: (PADDING, 'padding'),      # "The calculus began" — gloss; padding throughout
    456: (PADDING, 'padding'),      # medical detail but padding transitions throughout
    457: (CLEAN, 'clean'),          # brief action, clean
    458: (CLEAN, 'clean'),          # procedural, specific
    459: (CLEAN, 'clean'),          # "the boy from a farming town" — specific, earned
    460: (PADDING, 'padding'),      # stacked description, padding
    461: (PADDING, 'padding'),      # internal deliberation, padding — "loud, generous man" etc.
    462: (CLEAN, 'clean'),          # clean decision moment
    463: (NARRATOR_GLOSS, 'narrator_gloss'),  # "*Make him comfortable.*" + "the locked-away part of him wept" — narrator glossing the meaning
    464: (CLEAN, 'clean'),          # specific procedural, clean
    465: (CLEAN, 'clean'),          # brief dialogue
    466: (CLEAN, 'clean'),          # dialogue, strong
    467: (CLEAN, 'clean'),          # dialogue
    468: (CLEAN, 'clean'),          # brief action
    469: (CLEAN, 'clean'),          # specific, grounded
    470: (CLEAN, 'clean'),          # brief action call
    471: (PADDING, 'padding'),      # "jagged tooth against the purple sky" + padding
    472: (PADDING, 'padding'),      # "small pocket" + description padding
    473: (CLEAN, 'clean'),          # dialogue
    474: (CLEAN, 'clean'),          # specific assessment
    475: (PADDING, 'padding'),      # "He saw the plea, then the acceptance" — narrator gloss + padding action
    476: (CLEAN, 'clean'),          # brief, specific
    477: (PADDING, 'padding'),      # "dose, a massive one. Enough to bring peace, not just pain relief." — padding narration of euthanasia
    478: (PADDING, 'padding'),      # "numb trudge" + padding transitions
    479: (PADDING, 'padding'),      # padding with tricolon "Threes, the walking wounded, the expectant" + standing over blanket
    480: (CLEAN, 'clean'),          # strong opener
    481: (CLEAN, 'clean'),          # specific, grounded
    482: (CLEAN, 'clean'),          # brief dialogue
    483: (CLEAN, 'clean'),          # specific description
    484: (CLEAN, 'clean'),          # dialogue
    485: (CLEAN, 'clean'),          # dialogue, clean
    486: (PADDING, 'padding'),      # "first genuine crack in the façade" + padding hedging
    487: (PADDING, 'padding'),      # "like he was handing me a map with the treasure already marked" — dead figure + padding
    488: (PADDING, 'padding'),      # "a different beast" + padding transition
    489: (CLEAN, 'clean'),          # specific scene, clean
    490: (PADDING, 'padding'),      # "sunburnt and smiling, for a little while" — padding with hedged ending
    491: (PADDING, 'padding'),      # "bored-sounding" + padding investigation procedure
    492: (CLEAN, 'clean'),          # brief dialogue + clean
    493: (CLEAN, 'clean'),          # brief
    494: (PADDING, 'padding'),      # "ghost of a house" + padding transitions
    495: (CLEAN, 'clean'),          # specific, grounded
    496: (CLEAN, 'clean'),          # dialogue
    497: (CLEAN, 'clean'),          # brief internal, clean
    498: (CLEAN, 'clean'),          # brief scene
    499: (CLEAN, 'clean'),          # specific character description
    500: (CLEAN, 'clean'),          # brief dialogue
    501: (NARRATOR_GLOSS, 'narrator_gloss'),  # "The lie came automatically. It was always about the money, until it wasn't." — narrator glossing the character
    502: (CLEAN, 'clean'),          # dialogue, strong
    503: (CLEAN, 'clean'),          # internal reasoning, clean
    504: (NARRATOR_GLOSS, 'narrator_gloss'),  # "On the easel was the beginnings..." — narrator explains what the painting means
    505: (CLEAN, 'clean'),          # brief, punchy — clean
    506: (CLEAN, 'clean'),          # dialogue/action, clean
    507: (CLEAN, 'clean'),          # brief dialogue
    508: (CLEAN, 'clean'),          # clean, earned line
    509: (CLEAN, 'clean'),          # "Tears welled in her eyes, but she didn't cry" — single, earned
    510: (CLEAN, 'clean'),          # action, clean
    511: (CLEAN, 'clean'),          # dialogue, specific
    512: (CLEAN, 'clean'),          # brief
    513: (CLEAN, 'clean'),          # atmospheric opener, specific
    514: (CLEAN, 'clean'),          # specific scene
    515: (CLEAN, 'clean'),          # brief dialogue
    516: (CLEAN, 'clean'),          # brief internal
    517: (CLEAN, 'clean'),          # dialogue
    518: (CLEAN, 'clean'),          # dialogue, strong
    519: (CLEAN, 'clean'),          # scene, specific
    520: (PADDING, 'padding'),      # "apathy" joke + padding tag
    521: (CLEAN, 'clean'),          # dialogue, specific
    522: (PADDING, 'padding'),      # "accepting the smaller, more portable sensor" + padding description
    523: (PADDING, 'padding'),      # "low-frequency thrumming" + padding sensation description
    524: (PADDING, 'padding'),      # "brushing away the surface dusting" + padding procedural
    525: (CLEAN, 'clean'),          # specific, tense
    526: (CLEAN, 'clean'),          # specific, clean
    527: (CLEAN, 'clean'),          # brief dialogue
    528: (CLEAN, 'clean'),          # dialogue
    529: (CLEAN, 'clean'),          # dialogue
    530: (PADDING, 'padding'),      # "world dissolves into gradients" + padding thermal description
    531: (CLEAN, 'clean'),          # brief, clean
    532: (CLEAN, 'clean'),          # specific, tense
    533: (CLEAN, 'clean'),          # brief action
    534: (CLEAN, 'clean'),          # brief dialogue
    535: (CLEAN, 'clean'),          # brief dialogue
    536: (CLEAN, 'clean'),          # brief dialogue
    537: (CLEAN, 'clean'),          # specific
    538: (PADDING, 'padding'),      # "vibrating in his dental fillings" + padding + "electrically charged"
    539: (CLEAN, 'clean'),          # brief dialogue
    540: (PADDING, 'padding'),      # "disturbingly organic" + "has no business existing" — padding with vague emotion
    541: (PADDING, 'padding'),      # "like a geological formation sighing in relief" — dead metaphor + padding
    542: (NARRATOR_GLOSS, 'narrator_gloss'),  # "He stares into the opening... realizing that he has not mapped a landscape at all. He has mapped a lid." — narrator gloss
    543: (CLEAN, 'clean'),          # brief dialogue
    544: (NARRATOR_GLOSS, 'narrator_gloss'),  # "compelled toward the source... toward the only true blank space" — narrator explaining meaning
    545: (CLEAN, 'clean'),          # specific, grounded
    546: (CLEAN, 'clean'),          # specific memory
    547: (CLEAN, 'clean'),          # brief, clean
    548: (PADDING, 'padding'),      # "sealed off slightly" + "smell of pipe tobacco that lingered decades" + padding
    549: (CLEAN, 'clean'),          # specific, clean
    550: (CLEAN, 'clean'),          # brief action
    551: (CLEAN, 'clean'),          # brief
    552: (STANDALONE, 'standalone'), # italicised letter excerpt — standalone
    553: (CLEAN, 'clean'),          # brief, clean
    554: (CLEAN, 'clean'),          # specific
    555: (CLEAN, 'clean'),          # italicised quote, specific
    556: (NARRATOR_GLOSS, 'narrator_gloss'),  # "These were the qualities he offered Genevieve... apparently needed to offer to justify abandoning L." — narrator glossing
    557: (CLEAN, 'clean'),          # brief action, specific
    558: (CLEAN, 'clean'),          # brief, clean
    559: (CLEAN, 'clean'),          # specific
    560: (CLEAN, 'clean'),          # specific
    561: (CLEAN, 'clean'),          # brief
    562: (CLEAN, 'clean'),          # italicised diary entry, specific and earned
    563: (CLEAN, 'clean'),          # specific, clean
    564: (CLEAN, 'clean'),          # specific, grounded
    565: (NARRATOR_GLOSS, 'narrator_gloss'),  # "If Arthur had this passionate... why had they stayed?" — narrator asking rhetorical questions, glossing
    566: (CLEAN, 'clean'),          # brief
    567: (CLEAN, 'clean'),          # specific
    568: (CLEAN, 'clean'),          # brief, specific
    569: (PADDING, 'padding'),      # "incandescently, terribly happy" — adverb stacking; padding description
    570: (CLEAN, 'clean'),          # brief
    571: (DEAD_FIGURE, 'dead_figure'),  # "radiant in their shared brief moment of illicit connection" + "diligently oiling hinges, honoring the structure" — dead metaphor + dead figure accumulation
    572: (CLEAN, 'clean'),          # specific, grounded
    573: (CLEAN, 'clean'),          # specific
    574: (PADDING, 'padding'),      # "sticking point" + multiple hedging clauses + padding
    575: (PADDING, 'padding'),      # "tapping the page" + hedging + padding
    576: (CLEAN, 'clean'),          # specific
    577: (CLEAN, 'clean'),          # brief, earned
    578: (CLEAN, 'clean'),          # brief, clean
    579: (CLEAN, 'clean'),          # specific description
    580: (CLEAN, 'clean'),          # dialogue
    581: (CLEAN, 'clean'),          # dialogue
    582: (CLEAN, 'clean'),          # brief action
    583: (CLEAN, 'clean'),          # dialogue, specific — "remarkably clear" but character's speech
    584: (CLEAN, 'clean'),          # dialogue, academic
    585: (CLEAN, 'clean'),          # dialogue
    586: (PADDING, 'padding'),      # "sticking point" repetition + padding transitions
    587: (CLEAN, 'clean'),          # specific, technical
    588: (PADDING, 'padding'),      # "joviality receding" + padding observation
    589: (CLEAN, 'clean'),          # dialogue, strong
    590: (CLEAN, 'clean'),          # dialogue
    591: (CLEAN, 'clean'),          # specific
    592: (NARRATOR_GLOSS, 'narrator_gloss'),  # "These are not blueprints. They are allegories." + narrator explaining what the text "really" means
    593: (CLEAN, 'clean'),          # dialogue with action
    594: (PADDING, 'padding'),      # "He found Map Folio Delta. He traced the faint, almost invisible markings" — padding procedural
    595: (CLEAN, 'clean'),          # specific, precise
    596: (PADDING, 'padding'),      # "His skepticism momentarily eclipsed" + padding
    597: (CLEAN, 'clean'),          # strong closing line
}


def main():
    inpath = '/home/ben/code/prose-doctor/corpus/review_chunk_0.jsonl'
    outpath = '/home/ben/code/prose-doctor/corpus/reviewed_chunk_0.jsonl'

    records = []
    with open(inpath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records", file=sys.stderr)

    label_counts = {i: 0 for i in range(8)}
    changed = 0

    with open(outpath, 'w') as f:
        for i, record in enumerate(records):
            ml_label = record.get('label', 0)
            ml_class = record.get('class_name', 'clean')

            if i in MANUAL:
                label, class_name = MANUAL[i]
            else:
                # Fallback to auto-classifier for any records not manually reviewed
                label = classify(record['text'])
                class_name = CLASS_NAMES[label]

            if label != ml_label:
                changed += 1

            record['label'] = label
            record['class_name'] = class_name
            record['method'] = 'llm_review'
            label_counts[label] += 1
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Wrote {len(records)} records to {outpath}", file=sys.stderr)
    print(f"Changed {changed} labels from ML prediction", file=sys.stderr)
    print("\nLabel distribution:", file=sys.stderr)
    for lid, count in sorted(label_counts.items()):
        print(f"  {lid} ({CLASS_NAMES[lid]}): {count}", file=sys.stderr)


if __name__ == '__main__':
    main()
