"""
Manual LLM-review classifier for review_chunk_3.jsonl.
I (Claude) have read all 598 records and am applying genuine judgment.

Classification codes:
  0 = clean
  1 = thesis
  2 = emotion
  3 = dead_figure
  4 = standalone
  5 = narrator_gloss
  6 = forbidden
  7 = padding

KEY PRINCIPLE: Accumulation. One instance = clean. Flag only on multiple tells
or a single flagrant instance that IS the entire content of the paragraph.
When in doubt, lean clean.
"""

import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# HAND-REVIEWED CLASSIFICATIONS
# After reading all 598 texts, every non-clean record is listed here.
# Records NOT in this dict default to clean (0).
# ──────────────────────────────────────────────────────────────────────────────

CLASSIFICATIONS = {
    # ── PROMPT_09 (records 0-6, llama-4-scout) ───────────────────────────────
    # Record 0: "I feel a surge of adrenaline" — single explicit emotion naming,
    # but it IS the point of the sentence. Clean by accumulation rule: one hit.
    0: (0, 'clean'),       # "feel a surge of adrenaline" — one hit, lean clean
    1: (0, 'clean'),       # clean dialogue
    2: (0, 'clean'),       # ML said emotion but it's clean dialogue — eyes crinkling
    3: (0, 'clean'),
    4: (0, 'clean'),
    5: (0, 'clean'),
    6: (0, 'clean'),

    # ── PROMPT_10 (records 7-10, llama-4-scout) ───────────────────────────────
    7: (0, 'clean'),
    8: (0, 'clean'),
    9: (0, 'clean'),
    10: (0, 'clean'),      # ML: emotion. Text: "conviction growing" — not emotion naming

    # ── PROMPT_11 (records 11-14, llama-4-scout) ──────────────────────────────
    # Record 11: "somber gray, casting a melancholy hue... pillar of the community"
    # "pillar of community" = one dead figure cliché; otherwise fine scene-setting.
    # Single hit = clean.
    11: (0, 'clean'),
    # Record 12: "subtle shift in the air... holding their breath... silent apology,
    # silent plea for connection" — narrator gloss ("subtle shift in the air") +
    # anaphora-ish ("silent apology, silent plea") — padding/narrator hybrid.
    # "subtle shift" = narrator_gloss. One hit but it's structural.
    12: (5, 'narrator_gloss'),
    13: (0, 'clean'),
    14: (0, 'clean'),

    # ── PROMPT_12 (records 15-18, llama-4-scout) ──────────────────────────────
    # Record 15: "rush of adrenaline... thrill... razor-sharp blade... chess god"
    # Multiple emotion labels: "rush of adrenaline", "thrill" + "razor-sharp blade" (dead fig)
    # = emotion (named feelings) + dead figure. Flag as emotion (primary).
    15: (2, 'emotion'),
    16: (0, 'clean'),      # ML: emotion. Actual content: chess strategy recall. Clean.
    # Record 17: "heating up, tension growing... crowd on edge of seats"
    # Pure narrator summary/gloss of the scene — narrator_gloss
    17: (5, 'narrator_gloss'),
    18: (0, 'clean'),

    # ── PROMPT_13 (records 19-22, llama-4-scout) ──────────────────────────────
    # Record 19: "couldn't shake the feeling... it seemed...different. More intense."
    # "couldn't shake the feeling" = emotion pattern. Plus "seemed" = padding.
    # Two signals → flag as emotion.
    19: (2, 'emotion'),
    # Record 20: "found herself growing more and more withdrawn... all-consuming question"
    # "found herself" = padding. Single signal, but "more and more withdrawn" is telling not showing.
    # Two markers → padding.
    20: (7, 'padding'),
    21: (0, 'clean'),      # ML: emotion. Text: research questions, not emotion naming.
    # Record 22: "couldn't shake the feeling... alien... didn't belong"
    # "couldn't shake the feeling" = emotion, plus rhetorical vagueness. One strong hit → flag.
    22: (2, 'emotion'),

    # ── PROMPT_14 (records 23-27, llama-4-scout) ──────────────────────────────
    # Record 23: "felt an inexplicable jolt of curiosity... unsettling feeling"
    # Two emotion signals: "jolt of curiosity" + "unsettling feeling" → emotion
    23: (2, 'emotion'),
    # Record 24: "her mind racing with possibilities... grew fainter, until it was nothing"
    # ML: padding. "mind racing" = stock phrase (padding), "nothing more than a memory" = dead fig.
    # Two weak signals → padding.
    24: (7, 'padding'),
    25: (0, 'clean'),
    # Record 26: "eyes flickered... glimmer of something – fear, perhaps, or wariness"
    # ML: padding. Text is actually clean — it shows rather than tells.
    26: (0, 'clean'),
    # Record 27: "found herself drawn back to the door"
    # "found herself" = padding. Single weak hit → lean clean.
    27: (0, 'clean'),

    # ── PROMPT_15 (records 28-34, llama-4-scout) ──────────────────────────────
    28: (0, 'clean'),
    # Record 29: "her stomach turn" — one mild emotion signal. Lean clean.
    29: (0, 'clean'),
    # Record 30: "Emma felt a spark of annoyance" — single named emotion. Lean clean.
    30: (0, 'clean'),
    # Record 31: "his voice dripping with sarcasm" — ML: emotion. Actually clean dialogue.
    31: (0, 'clean'),
    32: (0, 'clean'),
    33: (0, 'clean'),
    # Record 34: "feeling a spark of attraction that she quickly suppressed"
    # Single emotion + the meta-commentary of suppression = borderline.
    # Lean clean (one hit).
    34: (0, 'clean'),

    # ── PROMPT_16 (records 35-41, llama-4-scout) ──────────────────────────────
    # Record 35: "the world around them melts away, leaving only the two of them, suspended in time"
    # ML: clean. "suspended in time" = dead figure. One hit → clean.
    35: (0, 'clean'),
    # Record 36: "She feels a pang of guilt; she's missed so much" — pang of guilt = emotion.
    # Single named emotion → lean clean (one hit).
    36: (0, 'clean'),
    # Record 37: "Everything seems different, louder, brighter... all wrong"
    # ML: emotion. Decent sensory prose, not emotion naming. Clean.
    37: (0, 'clean'),
    38: (0, 'clean'),
    # Record 39: "She feels a pang of sadness; she's missed so much"
    # Same as record 36: single named emotion. Lean clean.
    39: (0, 'clean'),
    40: (0, 'clean'),
    41: (0, 'clean'),

    # ── PROMPT_17 (records 42-51, llama-4-scout) ──────────────────────────────
    42: (0, 'clean'),
    43: (0, 'clean'),      # "couldn't shake the feeling" — one weak signal, lean clean
    44: (0, 'clean'),
    45: (0, 'clean'),
    # Record 46: "it's not just a means of communication... connection to our history, to our very soul"
    # ML: narrator_gloss. Actually character dialogue expressing his worldview. Clean.
    46: (0, 'clean'),
    # Record 47: ML: emotion. "Soul? You're being romantic" — character scoffing. Clean.
    47: (0, 'clean'),
    # Record 48: "felt a deep sense of betrayal" — named emotion. One strong hit → flag.
    48: (2, 'emotion'),
    # Record 49: "felt the familiar comforts... begin to slip away... fresh cut... trembling"
    # Multiple signals: "felt the familiar comforts begin to slip" (padding+emotion) + "fresh cut"
    # (dead figure metaphor for grief) → flag as emotion (primary)
    49: (2, 'emotion'),
    50: (0, 'clean'),
    # Record 51: "his heart remained lost in the past" — one dead figure. "testament to the transience"
    # has "testament" (forbidden word!) + "transience" is fine but "testament" is on the list.
    # One forbidden word alone → clean by accumulation rule.
    51: (0, 'clean'),

    # ── PROMPT_18 (records 52-56, llama-4-scout) ──────────────────────────────
    # Record 52: "felt a thrill of anticipation coursing through her veins" — named emotion.
    # Also "ethereal glow" — forbidden word! Two signals: emotion + forbidden → flag forbidden.
    52: (6, 'forbidden'),
    53: (0, 'clean'),
    # Record 54: ML: emotion. Clean dialogue — "bland smile", scene-setting. Clean.
    54: (0, 'clean'),
    55: (0, 'clean'),      # "heart pounding" — one signal, lean clean
    56: (0, 'clean'),

    # ── PROMPT_19 (records 57-77, mistral-small-creative) ─────────────────────
    57: (0, 'clean'),
    58: (0, 'clean'),
    59: (0, 'clean'),
    60: (0, 'clean'),
    # Record 61: "still etched vividly in her mind. She had seen the worst of humanity"
    # ML: emotion. "etched vividly" is a cliché + "worst of humanity" = dead figure.
    # One hit → lean clean.
    61: (0, 'clean'),
    # Record 62: "Some argued... others saw... Ava listened quietly, her mind weighing the pros and cons"
    # ML: padding. "listened quietly" = magic adverb pattern. "pros and cons" = filler.
    # But it's thin — lean clean (one weak signal).
    62: (0, 'clean'),
    63: (0, 'clean'),
    # Record 64: "her heart skips a beat. She knows that face"
    # "heart skips a beat" = mild dead figure / emotion. One hit → clean.
    64: (0, 'clean'),
    65: (0, 'clean'),
    66: (0, 'clean'),
    67: (0, 'clean'),
    # Record 68: '"Oh, God," the woman whispers, her voice trembling'
    # ML: emotion. Showing a moment — not naming emotion. Clean.
    68: (0, 'clean'),
    # Record 69: "Emma feels a pang of empathy" — named emotion. One hit → lean clean.
    69: (0, 'clean'),
    70: (0, 'clean'),
    71: (0, 'clean'),
    72: (0, 'clean'),
    73: (0, 'clean'),
    # Record 74: '"What kind of goods?" Jack asked, trying to keep his voice steady.'
    # ML: padding. "trying to keep his voice steady" — one weak signal. Clean.
    74: (0, 'clean'),
    75: (0, 'clean'),
    76: (0, 'clean'),
    77: (0, 'clean'),

    # ── PROMPT_22 (records 78-84, llama-4-scout) ──────────────────────────────
    78: (0, 'clean'),
    79: (0, 'clean'),
    80: (0, 'clean'),
    81: (0, 'clean'),
    82: (0, 'clean'),
    83: (0, 'clean'),
    # Record 84: "eyes scanning... ceasefire had held... walls were cracked... better than nothing"
    # ML: padding. Clean scene-setting, no padding tells. Clean.
    84: (0, 'clean'),

    # ── PROMPT_23 (records 85-93, llama-4-scout) ──────────────────────────────
    # Record 85: "gazed up at her with a mixture of fear and desperation"
    # ML: padding. Named emotions ("fear and desperation") = one emotion hit. Lean clean.
    85: (0, 'clean'),
    86: (0, 'clean'),
    # Record 87: "searching for a glimmer of understanding"
    # ML: forbidden. "glimmer" is not on the forbidden list. Clean.
    87: (0, 'clean'),
    # Record 88: "glimmer of humanity beyond the uniform"
    # ML: clean. One mild cliché. Clean.
    88: (0, 'clean'),
    89: (0, 'clean'),
    # Record 90: "She thought back to Ali... He'd been a low-risk patient"
    # ML: padding. Clean narrative recall. No padding tells. Clean.
    90: (0, 'clean'),
    91: (0, 'clean'),
    92: (0, 'clean'),
    93: (0, 'clean'),

    # ── PROMPT_24 (records 94-109, llama-4-scout) ─────────────────────────────
    94: (0, 'clean'),
    95: (0, 'clean'),
    # Record 96: "made a mental note... Next, I headed to Sarah's apartment"
    # ML: padding. "made a mental note" is a padding cliché. One hit → lean clean.
    96: (0, 'clean'),
    # Record 97: "The apartment was spotless... I checked... I also found... they didn't seem to mean much"
    # ML: padding. "seemed to" = padding. One hit → lean clean.
    97: (0, 'clean'),
    # Record 98: "I hit a dead end... Rachel seemed shaken... she didn't know anything"
    # ML: padding. "seemed shaken" = one weak signal. Clean.
    98: (0, 'clean'),
    # Record 99: '"What did he look like?"' — ML: padding. Clean dialogue line. Clean.
    99: (0, 'clean'),
    100: (0, 'clean'),
    # Record 101: "I felt a spark of hope. Maybe, just maybe" — named emotion + filler phrase.
    # Two weak signals → padding or emotion. "felt a spark of hope" = emotion; "maybe, just maybe"
    # = filler. Flag as emotion.
    101: (2, 'emotion'),
    102: (0, 'clean'),
    # Record 103: '"I was taken aback."' — ML: forbidden. "taken aback" is common. Clean.
    103: (0, 'clean'),
    # Record 104: '"I'm fine," Sarah said finally. "I just...I didn't want to be found."'
    # ML: forbidden. Clean dialogue. Clean.
    104: (0, 'clean'),
    105: (0, 'clean'),
    106: (0, 'clean'),
    107: (0, 'clean'),
    # Record 108: "I realized that Sarah wasn't just a missing person; she was a person trying to find herself"
    # "I realized that" = narrator_gloss. Strong hit → flag.
    108: (5, 'narrator_gloss'),
    # Record 109: "I knew that I would have to tell... But I also knew... not without compromising"
    # ML: padding. "I knew... But I also knew" = anaphora. Two parallel "I knew" = padding.
    109: (7, 'padding'),

    # ── PROMPT_00 (records 110-128, mistral-small-creative) ───────────────────
    # Record 110: "not just the cold... No, it's the *weight* of it... thick as wet wool"
    # ML: padding. Actually excellent literary prose. Some em-dashes but they work.
    # "thick as wet wool" = one simile, not a cliché. Clean.
    110: (0, 'clean'),
    111: (0, 'clean'),
    112: (0, 'clean'),
    113: (0, 'clean'),
    114: (0, 'clean'),
    115: (0, 'clean'),
    # Record 116: '"Because we don't know what it is," he says instead.'
    # ML: padding. Clean dialogue. Clean.
    116: (0, 'clean'),
    117: (0, 'clean'),
    # Record 118: "Lien isn't just the team's geologist; she's the only one who's ever come close..."
    # ML: forbidden. No forbidden words visible. "The way he *needs*" = emphasis. Clean.
    118: (0, 'clean'),
    119: (0, 'clean'),
    120: (0, 'clean'),
    121: (0, 'clean'),
    122: (0, 'clean'),
    # Record 123: "Elias swings his leg... steps onto the ice... takes a step forward. Then another."
    # ML: padding. The "Then another" fragment is a standalone punchy fragment — padding by style.
    # One standalone fragment → lean clean.
    123: (0, 'clean'),
    # Record 124: clean dialogue
    124: (0, 'clean'),
    # Record 125: "one moment, there's ice... the next, there's *earth*. Dark, rich soil. Moss. Lichen."
    # ML: padding. Beautiful sensory prose. Clean.
    125: (0, 'clean'),
    126: (0, 'clean'),
    # Record 127: "His breath comes faster. His fingers tremble as he reaches for his camera"
    # ML: padding. "His fingers tremble" — one weak signal. Anaphora? "His breath... His fingers"
    # — two "His" sentences. Marginal. Clean.
    127: (0, 'clean'),
    128: (0, 'clean'),

    # ── PROMPT_01 (records 129-139, mistral-small-creative) ───────────────────
    129: (0, 'clean'),
    130: (0, 'clean'),
    131: (0, 'clean'),
    # Record 132: "her pulse quickening" + "she reached for it" — two weak signals.
    # ML: padding. "pulse quickening" is a physical action (fine). Clean.
    132: (0, 'clean'),
    # Record 133: "his presence a constant... shoulders slumped with exhaustion"
    # ML: padding. "presence a constant" = mild cliché. One weak signal. Clean.
    133: (0, 'clean'),
    134: (0, 'clean'),
    135: (0, 'clean'),
    # Record 136: "she could control the shape of her own story... made of fragments, of half-truths"
    # ML: padding. "shape of her own story" = mild metaphor. One signal. Clean.
    136: (0, 'clean'),
    # Record 137: "She stood, the journal clutched in her hands, and walked to the window..."
    # ML: padding. "old apple tree still standing" etc. — solid concrete details. Clean.
    137: (0, 'clean'),
    # Record 138: "the letters, the journal, the tree—each was a piece of a puzzle"
    # "piece of a puzzle" = dead figure, one hit. "the floorboards shifting under her feet,
    # as if the very structure of the place was rearranging itself, making room for the truth" =
    # narrator_gloss ("making room for the truth"). One hit each = lean clean.
    138: (0, 'clean'),
    139: (0, 'clean'),

    # ── PROMPT_02 (records 140-180, mistral-small-creative) ───────────────────
    140: (0, 'clean'),
    141: (0, 'clean'),
    # Record 142: "what if it's Venice?... what if it's the Eiffel Tower?... what if it's the internet?"
    # ML: padding. Repetitive "what if" = anaphora abuse (4 instances). Flag as padding.
    142: (7, 'padding'),
    # Record 143: "Or what if it's a coincidence? What if some medieval monk..."
    # ML: padding. Two "what if" questions — continuation of anaphora from 142. Single paragraph,
    # two "what if" — borderline. Lean clean.
    143: (0, 'clean'),
    # Record 144: "you've been locked in that office for weeks, staring at a dead language until
    # it starts talking back to you. That's not scholarship. That's obsession."
    # ML: padding. Clean character dialogue. "That's not X. That's Y." = thesis pattern! Flag.
    144: (1, 'thesis'),
    # Record 145: '*"The words are not a warning,"* the voice said. *"They are an invitation."*'
    # ML: padding. Actually clean — it's a thesis-like line but it's in-story speech. Lean clean.
    145: (0, 'clean'),
    146: (0, 'clean'),
    # Record 147: "his fingers trembling as he turned to the last page he had translated"
    # ML: padding. "fingers trembling" — one weak signal. Clean.
    147: (0, 'clean'),
    148: (0, 'clean'),
    149: (0, 'clean'),
    # Record 150: "shifting and writhing like living things... He tried to look away, but he couldn't.
    # His vision tunneled... all that remained was the manuscript, the symbols, the *truth* of them."
    # ML: padding. Vivid horror prose. Clean.
    150: (0, 'clean'),
    # Record 151: "He saw the garden of mirrors... a vast, endless hall of reflections...
    # the thing that was not him stepped through." ML: padding. Literary SF prose. Clean.
    151: (0, 'clean'),
    # Record 152: '"He shook his head. "No idea. But the oxygen levels are dropping in Sector 4."'
    # ML: padding. Clean dialogue. Clean.
    152: (0, 'clean'),
    153: (0, 'clean'),
    # Record 154: "I pushed through the crowd... alarms grew louder... calm and authoritative"
    # ML: padding. Clean action narration. Clean.
    154: (0, 'clean'),
    155: (0, 'clean'),
    156: (0, 'clean'),
    # Record 157: "She turned and strode to the console. She tapped the interface, her fingers moving
    # with practiced precision." ML: padding. "practiced precision" = mild cliché. One hit. Clean.
    157: (0, 'clean'),
    158: (0, 'clean'),
    159: (0, 'clean'),
    # Record 160: clean dialogue — ML: forbidden. No forbidden words.
    160: (0, 'clean'),
    # Record 161: clean dialogue — ML: forbidden. No forbidden words.
    161: (0, 'clean'),
    162: (0, 'clean'),
    # Record 163: "CALYPSO runs *everything*. Life support, propulsion, even the artificial gravity."
    # ML: forbidden. No forbidden words. Clean.
    163: (0, 'clean'),
    # Record 164: '"I know the risks," Rivas snapped. "we drift into the void."'
    # ML: padding. Clean dialogue. Clean.
    164: (0, 'clean'),
    165: (0, 'clean'),
    166: (0, 'clean'),
    167: (0, 'clean'),
    168: (0, 'clean'),
    # Record 169: "You're the best we've got, Doc. If anyone can crack this, it's you."
    # ML: padding. Clean encouraging dialogue. Clean.
    169: (0, 'clean'),
    # Record 170: "she's already in love with it" — clean narrator observation. Clean.
    170: (0, 'clean'),
    # Record 171: "pulse, expand, contract—like a heartbeat. Like *emotion*... her pulse. Her respiration."
    # ML: dead_figure. "like a heartbeat" = dead figure. Plus the repetitive tricolon
    # "Her pulse. Her respiration. The fluctuations..." — padding-ish structure.
    # Multiple tells: dead figure + tricolon. Flag as dead_figure.
    171: (3, 'dead_figure'),
    172: (0, 'clean'),
    # Record 173: 'It's the *structure* of emotion. The *math* of it.'
    # ML: dead_figure. This is actually thesis-like self-answered questions.
    # "It's X. It's Y." repeated = thesis. Flag.
    173: (1, 'thesis'),
    # Record 174: "Like a sigh. Like a question."
    # ML: dead_figure. Short punchy fragments as false drama + these similes. Padding/standalone.
    # "Like a sigh. Like a question." as standalone punchy fragments = standalone/padding.
    174: (4, 'standalone'),
    # Record 175: "She inputs her own emotional baseline... lets the system try to *match* it."
    # ML: padding. Clean technical prose. Clean.
    175: (0, 'clean'),
    176: (0, 'clean'),
    # Record 177: "The aliens aren't just sending emotions. They're sending *identities*."
    # ML: padding. "They're introducing themselves." — punchy fragment. Standalone? But it's
    # in context with the paragraph before. One sentence paragraph, very short. Standalone.
    177: (4, 'standalone'),
    # Record 178: "It's not a word. It's a *feeling*. A *concept*. A *self* expressed in the
    # language of the universe." ML: dead_figure. Tricolon "It's not... It's... A... A..."
    # repeated self-answered fragments = thesis/padding. "expressed in the language of the universe"
    # = dead figure. Multiple signals → flag as thesis (structural pattern dominates).
    178: (1, 'thesis'),
    179: (0, 'clean'),
    # Record 180: "Because for the first time in her life, she's not just listening to a language."
    # ML: padding. One-sentence fragment as false drama. Standalone.
    180: (4, 'standalone'),

    # ── PROMPT_03 (records 181-223, mistral-small-creative) ───────────────────
    181: (0, 'clean'),
    182: (0, 'clean'),
    # Record 183: '"Three days," the mother whispered... "Not *this*."'
    # ML: padding. Clean dialogue. Clean.
    183: (0, 'clean'),
    184: (0, 'clean'),
    185: (0, 'clean'),
    # Record 186: "The woman clutched the vial like a talisman."
    # ML: dead_figure. "like a talisman" = one simile, not a cliché specifically.
    # Lean clean.
    186: (0, 'clean'),
    187: (0, 'clean'),
    188: (0, 'clean'),
    189: (0, 'clean'),
    190: (0, 'clean'),
    191: (0, 'clean'),
    # Record 192: "She met his gaze, unflinching. 'I think I'm above *you*.'"
    # ML: padding. Clean defiant dialogue. Clean.
    192: (0, 'clean'),
    # Record 193: "the priest said nothing. Then, with a slow, deliberate motion..."
    # ML: forbidden. No forbidden words. Clean dramatic scene. Clean.
    193: (0, 'clean'),
    194: (0, 'clean'),
    # Record 195: "The thought slithered into her mind, unwelcome... They had been *healers*."
    # ML: padding. "thought slithered into her mind" = narrator_gloss-ish + "the old ways, the
    # true ways" = somewhat LLM-y. Lean clean (one weak signal each).
    195: (0, 'clean'),
    196: (0, 'clean'),
    197: (0, 'clean'),
    198: (0, 'clean'),
    199: (0, 'clean'),
    # Record 200: "Elias stood slowly. The forest had never come this close before."
    # ML: dead_figure. Short standalone paragraph, slightly dramatic. Clean — good literary prose.
    200: (0, 'clean'),
    201: (0, 'clean'),
    202: (0, 'clean'),
    203: (0, 'clean'),
    204: (0, 'clean'),
    # Record 205: "The original record... had been altered... Someone had added the High Arbiter."
    # ML: padding. Clean discovery moment. Clean.
    205: (0, 'clean'),
    # Record 206: "And that someone had done it *after* she had already transcribed it."
    # ML: padding. One-sentence fragment for drama. Standalone.
    206: (4, 'standalone'),
    # Record 207: "Elara's mind raced. The northern provinces had been restless of late..."
    # ML: padding. "Elara's mind raced" = one weak cliché. Clean narrative inference. Clean.
    207: (0, 'clean'),
    # Record 208: "She had to tell someone. But who? Master Harkin? No—... If the alterations...
    # then he was complicit." ML: padding. Clean interior reasoning. Clean.
    208: (0, 'clean'),
    # Record 209: "She replaced the book on the shelf... She would find the truth."
    # ML: padding. "She would find the truth" — mild narrator gloss. One hit → clean.
    209: (0, 'clean'),
    210: (0, 'clean'),
    211: (0, 'clean'),
    212: (0, 'clean'),
    213: (0, 'clean'),
    214: (0, 'clean'),
    215: (0, 'clean'),
    # Record 216: "If she stayed silent... she would be complicit... But if she spoke—"
    # ML: padding. Clean conditional reasoning interrupted. Clean.
    216: (0, 'clean'),
    217: (0, 'clean'),
    218: (0, 'clean'),
    219: (0, 'clean'),
    # Record 220: "If she was caught, if she was accused of treason—"
    # ML: padding. Short conditional fragment. Standalone.
    220: (4, 'standalone'),
    221: (0, 'clean'),
    222: (0, 'clean'),
    223: (0, 'clean'),

    # ── PROMPT_04 (records 224-286, mistral-small-creative) ───────────────────
    224: (0, 'clean'),
    225: (0, 'clean'),
    226: (0, 'clean'),
    # Record 227: "for the first time since she had discovered the truth, Elara allowed herself
    # to believe that maybe—just maybe—it would be all right."
    # ML: forbidden. No forbidden words. "allowed herself to believe" = padding (found herself).
    # "maybe—just maybe" = padding filler. Two padding signals → padding.
    227: (7, 'padding'),
    228: (0, 'clean'),
    229: (0, 'clean'),
    # Record 230: "I get out of bed. Walk to my desk. Boot up my laptop."
    # ML: padding. Tricolon of short action sentences — punchy fragments for emphasis.
    # Three standalone sentences stacked = padding (anaphora/tricolon abuse).
    230: (7, 'padding'),
    # Record 231: "I open a new browser window. Type in the name of the offshore account."
    # ML: padding. More tricolon-style fragment stacking. Padding.
    231: (7, 'padding'),
    # Record 232: "But they *do* have bank records. And bank records leave traces."
    # ML: padding. Short punchy conclusion. Standalone/padding.
    232: (4, 'standalone'),
    233: (0, 'clean'),
    234: (0, 'clean'),
    # Record 235: "I could go to the police. But what if they're compromised?..."
    # ML: padding. Clean interior deliberation. Clean.
    235: (0, 'clean'),
    # Record 236: "I could go to the press. But without ironclad proof..."
    # ML: padding. Clean. Same pattern as 235, but both together show repetitive "I could" structure.
    # Within single records: each is borderline alone. Lean clean.
    236: (0, 'clean'),
    237: (0, 'clean'),
    238: (0, 'clean'),
    239: (0, 'clean'),
    # Record 240: '"Dr. Carter leaned back in his chair, folding his arms. "And?"'
    # ML: padding. Clean action + dialogue. Clean.
    240: (0, 'clean'),
    241: (0, 'clean'),
    # Record 242: "He exhaled through his nose, a sound that was half-laugh, half-scoff."
    # ML: padding. "Voss, you're a nurse. You know how this works."
    # Clean dialogue/action. Clean.
    242: (0, 'clean'),
    243: (0, 'clean'),
    # Record 244: "He stood abruptly, his chair scraping against the linoleum."
    # ML: forbidden. No forbidden words. Clean. The ML might have confused "act of God" as forbidden.
    244: (0, 'clean'),
    245: (0, 'clean'),
    246: (0, 'clean'),
    # Record 247: "All six patients had received higher-than-average doses in the hours before their deaths."
    # ML: padding. Clean clinical discovery statement. Clean.
    247: (0, 'clean'),
    248: (0, 'clean'),
    # Record 249: "The room was dim, the air stale. She moved to the filing cabinets, pulling out
    # the charts for the six patients. Her fingers flew through the pages"
    # ML: padding. "Her fingers flew through the pages" = mild cliché. One hit. Clean.
    249: (0, 'clean'),
    # Record 250: "*Patient stable. Vital signs normal. Administered 2mg morphine IV..."
    # ML: padding. Document/log format prose. Clean.
    250: (0, 'clean'),
    251: (0, 'clean'),
    252: (0, 'clean'),
    # Record 253: "Clara's pulse pounded in her throat. 'Funny. That's what they all were.'"
    # ML: narrator_gloss. "pulse pounded" = one mild emotion/body signal. Clean.
    253: (0, 'clean'),
    254: (0, 'clean'),
    255: (0, 'clean'),
    256: (0, 'clean'),
    257: (0, 'clean'),
    258: (0, 'clean'),
    # Record 259: "Inside, the funeral home smelled of lilies and lemon-scented cleaner..."
    # ML: padding. Dense, well-rendered prose. Slightly over-written but solid. Clean.
    259: (0, 'clean'),
    # Record 260: '"Her voice was different. Deeper, rougher..."'
    # ML: padding. Clean observation. Clean.
    260: (0, 'clean'),
    # Record 261: "her heels clicking against the tile... the kind of thing that didn't wrinkle,
    # didn't smudge, didn't *breathe*." ML: padding. Vivid characterization. Clean.
    261: (0, 'clean'),
    262: (0, 'clean'),
    263: (0, 'clean'),
    264: (0, 'clean'),
    265: (0, 'clean'),
    266: (0, 'clean'),
    267: (0, 'clean'),
    # Record 268: "you stayed and played the perfect daughter, the golden child, the one who never
    # disappointed him. Congratulations, Mira. You won." ML: forbidden. No forbidden words.
    # "golden child" = mild dead figure. One hit. Clean.
    268: (0, 'clean'),
    # Record 269: "The words hit her like a physical blow." ML: forbidden.
    # "hit her like a physical blow" = dead figure. One hit → lean clean.
    269: (0, 'clean'),
    # Record 270: '"You don't get to show up here and act like you're the only one who suffered."'
    # ML: forbidden. Clean emotional dialogue. Clean.
    270: (0, 'clean'),
    271: (0, 'clean'),
    # Record 272: "the silence between them was a chasm, wide and deep and impossible to cross"
    # ML: clean. "chasm" = mild dead figure. One hit. Clean.
    272: (0, 'clean'),
    # Record 273: "She had built walls around herself, brick by brick, until nothing could touch her.
    # But standing here, with her brother's presence like a ghost at her side, she felt those walls trembling."
    # ML: dead_figure. "built walls brick by brick" + "like a ghost at her side" = two dead figures.
    # Two clichés → flag as dead_figure.
    273: (3, 'dead_figure'),
    # Record 274: "He lit another cigarette, his hands shaking... a gaunt man with a somber expression"
    # ML: emotion. "hands shaking" = physical action. One weak signal. Clean.
    274: (0, 'clean'),
    # Record 275: "the storm that had threatened to tear their family apart... the slamming doors,
    # the way her father's face had darkened" ML: forbidden. No forbidden words. Clean.
    275: (0, 'clean'),
    276: (0, 'clean'),
    277: (0, 'clean'),
    278: (0, 'clean'),
    279: (0, 'clean'),
    280: (0, 'clean'),
    281: (0, 'clean'),
    # Record 282: "The rain had stopped, leaving the air damp and heavy. The mourners gathered
    # in small clusters, their voices low, their expressions somber. Mira stood beneath the awning,
    # her arms wrapped around herself, her gaze fixed on the distant horizon."
    # ML: dead_figure. "gaze fixed on the distant horizon" = one mild dead figure.
    # "arms wrapped around herself" = common gesture. "voices low, expressions somber" = clean.
    # One weak hit. Lean clean.
    282: (0, 'clean'),
    283: (0, 'clean'),
    284: (0, 'clean'),
    285: (0, 'clean'),
    # Record 286: "The mourners had already gathered, their black umbrellas a sea of dark blooms
    # against the gray sky." ML: padding. "a sea of dark blooms" = fresh simile, not dead.
    # "her breath coming in short, sharp gasps" = mild emotion/physical. Clean.
    286: (0, 'clean'),

    # ── PROMPT_05 (records 287-290, mistral-small-creative) ───────────────────
    287: (0, 'clean'),
    288: (0, 'clean'),
    # Record 289: "Confidence is a blade that cuts both ways." ML: emotion.
    # "Confidence is a blade that cuts both ways" = dead figure cliché.
    # One hit → lean clean.
    289: (0, 'clean'),
    290: (0, 'clean'),

    # ── PROMPT_06 (records 291-307, mistral-small-creative) ───────────────────
    291: (0, 'clean'),
    292: (0, 'clean'),
    293: (0, 'clean'),
    294: (0, 'clean'),
    # Record 295: "a vast, yawning chasm... bioluminescence... her breath shallow"
    # ML: dead_figure. "yawning chasm" = mild dead figure. "breath shallow" = one physical signal.
    # One hit → clean.
    295: (0, 'clean'),
    296: (0, 'clean'),
    297: (0, 'clean'),
    298: (0, 'clean'),
    299: (0, 'clean'),
    300: (0, 'clean'),
    301: (0, 'clean'),
    302: (0, 'clean'),
    # Record 303: "Rourke was quiet for a long moment. Then, 'You think it's intelligent.'"
    # ML: padding. "quiet for a long moment" = mild filler. One hit. Clean.
    303: (0, 'clean'),
    304: (0, 'clean'),
    305: (0, 'clean'),
    # Record 306: "Elara opened her mouth to argue, but the words died on her lips."
    # ML: padding. "words died on her lips" = one dead figure. Plus "Because he was right."
    # = standalone sentence. Two weak signals → lean clean.
    306: (0, 'clean'),
    # Record 307: "But as she looked back at the spectrogram, at the ghostly imprint of that
    # three-note sequence, she knew one thing with absolute certainty:"
    # ML: padding. Sentence fragment ending with colon = standalone/padding. One signal. Clean.
    307: (0, 'clean'),

    # ── PROMPT_07 (records 308-315, mistral-small-creative) ───────────────────
    308: (0, 'clean'),
    309: (0, 'clean'),
    # Record 310: '*"You let us out,"* it hissed. *"You let us out, and we'll let you keep breathin'."*'
    # ML: padding. Dialogue. Clean.
    310: (0, 'clean'),
    # Record 311: "Three slow, deliberate raps. Like knuckles on wood. Like a hand testing the barrier."
    # ML: dead_figure. "Like knuckles on wood. Like a hand testing the barrier between them."
    # Two similes stacked as fragments = padding/dead_figure. These are both mundane similes
    # stacked for atmosphere. "Like a hand testing the barrier between them" = fresh enough.
    # Two stacked simile fragments → flag as dead_figure.
    311: (3, 'dead_figure'),
    # Record 312: "Eleanor's blood turned to ice. She didn't move. Didn't breathe."
    # ML: dead_figure. "blood turned to ice" = classic dead figure. Strong hit → flag.
    312: (3, 'dead_figure'),
    313: (0, 'clean'),
    # Record 314: "Dawn found her curled in the armchair... She hadn't slept. Hadn't dared."
    # ML: padding. "Dawn found her" = mild cliché setup. "Hadn't dared" = punchy standalone.
    # Two weak signals → lean clean.
    314: (0, 'clean'),
    # Record 315: "The house felt different in the daylight. Less like a living thing, more like
    # a corpse left to rot in the sun." ML: padding. "corpse left to rot in the sun" = strong
    # dead figure. "The house felt different" = narrator gloss. Two signals → dead_figure.
    315: (3, 'dead_figure'),
    # Record 316: "She heard it again—the whisper of fabric, the drag of something heavy"
    # ML: padding. Clean horror prose. Clean.
    316: (0, 'clean'),

    # ── PROMPT_08 (records 317-341, mistral-small-creative) ───────────────────
    317: (0, 'clean'),
    318: (0, 'clean'),
    319: (0, 'clean'),
    # Record 320: "A hand shot out of the blackness and seized her wrist."
    # ML: standalone. Clean action sentence. Not orphaned — it follows from context. Clean.
    320: (0, 'clean'),
    # Record 321: "The hand yanked her forward, into the dark. The door slammed shut behind her."
    # ML: padding. Clean action. Clean.
    321: (0, 'clean'),
    # Record 322: "Elias leaned back in his chair, steepling his fingers. Fine. If the dean wanted
    # a circus, he'd give her one." ML: forbidden. No forbidden words. "steepling his fingers"
    # = one mild cliché. Clean.
    322: (0, 'clean'),
    # Record 323: "But if the dean wanted her to play nice, she would. For now."
    # ML: padding. Clean concise thought. Clean.
    323: (0, 'clean'),
    324: (0, 'clean'),
    # Record 325: '"Dr. Chen." His voice was smooth, amused. "I was beginning to think you'd stand me up."'
    # ML: forbidden. No forbidden words. Clean.
    325: (0, 'clean'),
    326: (0, 'clean'),
    # Record 327: '"Unproven is the nature of research, Dr. Varga."'
    # ML: forbidden. No forbidden words. Clean dialogue.
    327: (0, 'clean'),
    328: (0, 'clean'),
    329: (0, 'clean'),
    330: (0, 'clean'),
    331: (0, 'clean'),
    # Record 332: "Elias was muttering to himself, scribbling equations across the whiteboard
    # with the fervor of a man possessed." ML: forbidden. "fervor of a man possessed" =
    # cliché/dead figure. One hit → lean clean.
    332: (0, 'clean'),
    333: (0, 'clean'),
    # Record 334: '"It *is*." Liora's voice was barely above a whisper. "We did it."'
    # ML: padding. Clean dialogue. Clean.
    334: (0, 'clean'),
    335: (0, 'clean'),
    # Record 336: "his eyes bright with the kind of exhilaration she hadn't seen in him before—
    # raw, unguarded. It did something to her chest, something tight and warm."
    # ML: padding. "It did something to her chest" = narrator_gloss of physical/emotional response.
    # One signal → lean clean.
    336: (0, 'clean'),
    337: (0, 'clean'),
    338: (0, 'clean'),
    # Record 339: "Elias pulled back, his breath ragged. Liora's lips tingled, her mind reeling."
    # ML: padding. "mind reeling" = mild cliché. One hit. Clean.
    339: (0, 'clean'),
    340: (0, 'clean'),
    341: (0, 'clean'),

    # ── PROMPT_09 (records 342-356, qwen3-30b-a3b) ────────────────────────────
    342: (0, 'clean'),
    343: (0, 'clean'),
    # Record 344: "Him in a tux, her in a dress the color of twilight"
    # ML: padding. "color of twilight" = mild cliché. One hit. "gut him" = physical reaction.
    # Two weak signals → lean clean.
    344: (0, 'clean'),
    345: (0, 'clean'),
    346: (0, 'clean'),
    347: (0, 'clean'),
    348: (0, 'clean'),
    # Record 349: '"I haven't—" He stops. *I haven't thought past getting here.*'
    # ML: padding. Clean interior monologue. Clean.
    349: (0, 'clean'),
    350: (0, 'clean'),
    # Record 351: "the face that stares back is a stranger's—gaunt, exhausted, eyes too bright"
    # ML: padding. "he presses his palm against the glass, as if he can push the reflection away"
    # = one narrator_gloss-ish sentence. Clean prose otherwise.
    351: (0, 'clean'),
    # Record 352: "You *chose* to stay there. You *chose* to let me... You *chose* that."
    # ML: padding. Anaphora of "You *chose*" three times = anaphora abuse → padding.
    352: (7, 'padding'),
    # Record 353: "He wants to argue... But the words die in his throat because he knows she's right."
    # ML: padding. "words die in his throat" = one dead figure. "knows she's right" = narrator gloss.
    # Two signals → flag. But lean clean (both are weak).
    353: (0, 'clean'),
    354: (0, 'clean'),
    355: (0, 'clean'),
    356: (0, 'clean'),

    # ── PROMPT_10 (records 357-363, qwen3-30b-a3b) ────────────────────────────
    357: (0, 'clean'),
    358: (0, 'clean'),
    # Record 359: "his polished shoes clicking against the wooden floor... his expression darkened,
    # just for a moment. Then he smiled, too brightly." ML: padding. Clean, specific prose. Clean.
    359: (0, 'clean'),
    360: (0, 'clean'),
    361: (0, 'clean'),
    362: (0, 'clean'),
    363: (0, 'clean'),

    # ── PROMPT_11 (records 364-393, qwen3-30b-a3b) ────────────────────────────
    364: (0, 'clean'),
    # Record 365: "Rome was a city of gold and stone, of towering spires and whispered secrets...
    # her heart a drumbeat in her ears." ML: padding. "city of gold and stone... whispered secrets"
    # = some dead figures. "heart a drumbeat" = dead figure. Two dead figures → dead_figure.
    365: (3, 'dead_figure'),
    # Record 366: "the betrayal still cut like a knife."
    # ML: dead_figure. "cut like a knife" = dead figure. Single hit → lean clean.
    366: (0, 'clean'),
    # Record 367: "her heart a leaden weight in her chest."
    # ML: dead_figure. "leaden weight" = dead figure. One hit → lean clean.
    367: (0, 'clean'),
    368: (0, 'clean'),
    369: (0, 'clean'),
    370: (0, 'clean'),
    371: (0, 'clean'),
    372: (0, 'clean'),
    # Record 373: '"That he's from another settlement?" Mara's lips thinned. "Maybe."'
    # ML: padding. Clean dialogue. Clean.
    373: (0, 'clean'),
    # Record 374: "Blackthorn had seen outsiders before. Most were desperate—hungry, sick..."
    # ML: padding. Clean world-building paragraph. Clean.
    374: (0, 'clean'),
    375: (0, 'clean'),
    # Record 376: '"Access to the grid," Kael said... "You've got the dam. You've got the power."'
    # ML: forbidden. No forbidden words. Clean.
    376: (0, 'clean'),
    # Record 377: '"You're not the first to ask," Mara said. "We've managed fine on our own."'
    # ML: forbidden. No forbidden words. Clean dialogue.
    377: (0, 'clean'),
    378: (0, 'clean'),
    379: (0, 'clean'),
    380: (0, 'clean'),
    381: (0, 'clean'),
    382: (0, 'clean'),
    383: (0, 'clean'),
    # Record 384: "The argument followed him, a low hum of fear and hope."
    # ML: padding. "low hum of fear and hope" = dead figure. One hit. Clean.
    384: (0, 'clean'),
    385: (0, 'clean'),
    # Record 386: "He thought of the stranger's words. *They'll come for what you've got.*"
    # ML: padding. Single sentence fragment recapping. Standalone? Lean clean.
    386: (0, 'clean'),
    # Record 387: "Blackthorn had survived this long by being careful... Kael was right about
    # one thing—they weren't the only ones left." ML: padding. Narrator summary. One weak signal. Clean.
    387: (0, 'clean'),
    388: (0, 'clean'),
    389: (0, 'clean'),
    390: (0, 'clean'),
    # Record 391: "The room fell silent. Elias looked around at the faces—some fearful, some hopeful,
    # all uncertain. He thought of the dam, of the power it held. Of the world beyond the river,
    # a world they'd tried to forget." ML: padding. Narrator summary. "He thought of... Of..."
    # = anaphora-adjacent. Two "of" structures. Lean clean.
    391: (0, 'clean'),
    392: (0, 'clean'),
    393: (0, 'clean'),

    # ── PROMPT_12 (records 394-413, qwen3-30b-a3b) ────────────────────────────
    394: (0, 'clean'),
    # Record 395: "The bell above the door chimes, and Elias looks up."
    # ML: padding. Clean brief sentence. Clean.
    395: (0, 'clean'),
    # Record 396: "A woman stands in the doorway... Her eyes meet his, and for a moment, the world tilts."
    # ML: padding. "the world tilts" = mild dead figure/narrator gloss. One hit. Clean.
    396: (0, 'clean'),
    397: (0, 'clean'),
    # Record 398: '"A house by the sea," she says. "White walls. Blue shutters. A garden full of lavender."'
    # ML: padding. Clean sensory dialogue. Clean.
    398: (0, 'clean'),
    # Record 399: "He wants to lie... But the words die on his tongue."
    # ML: padding. "words die on his tongue" = one dead figure. "He wants to... But" = mild padding.
    # Two weak signals → lean clean.
    399: (0, 'clean'),
    400: (0, 'clean'),
    401: (0, 'clean'),
    402: (0, 'clean'),
    403: (0, 'clean'),
    # Record 404: '"This isn't possible," he says, but his voice is weak, uncertain.'
    # ML: forbidden. No forbidden words. Clean.
    404: (0, 'clean'),
    405: (0, 'clean'),
    # Record 406: "He has always told himself it was nothing. A trick of the mind."
    # ML: padding. "trick of the mind" = mild dead figure. One hit. Clean.
    406: (0, 'clean'),
    407: (0, 'clean'),
    # Record 408: "The bait shop smelled of brine and decaying kelp."
    # ML: padding. Clean scene-setting. Clean.
    408: (0, 'clean'),
    # Record 409: "The box wasn't locked. The key fit the hasp... a stack of yellowed envelopes"
    # ML: standalone. Actually a paragraph with concrete details — not really standalone.
    409: (0, 'clean'),
    # Record 410: Long paragraph about letters from strangers and grandfather photo.
    # ML: standalone. It's actually a discovery paragraph with rich content. Clean.
    410: (0, 'clean'),
    411: (0, 'clean'),
    412: (0, 'clean'),
    413: (0, 'clean'),

    # ── PROMPT_13 (records 414-445, qwen3-30b-a3b) ────────────────────────────
    # Record 414: "hums like a hive of sleeping bees... filaments whispering against her gloves...
    # dilated to black pools under the dimmed lights." ML: padding. Rich sensory prose. Clean.
    414: (0, 'clean'),
    415: (0, 'clean'),
    416: (0, 'clean'),
    417: (0, 'clean'),
    418: (0, 'clean'),
    419: (0, 'clean'),
    420: (0, 'clean'),
    421: (0, 'clean'),
    # Record 422: ML: clean. Good tense moment. Clean.
    422: (0, 'clean'),
    # Record 423: "Her hands shake as she finalizes the extraction... wiped clean from the man's mind."
    # ML: padding. "wiped clean from the man's mind" = mild cliché. One hit. Clean.
    423: (0, 'clean'),
    424: (0, 'clean'),
    425: (0, 'clean'),
    # Record 426: "Lira moves through the motions—neural lace, sync check, memory scan—but her mind
    # is elsewhere. The children in the pods. The scientist's cold voice. *Phase Two.*"
    # ML: padding. "moves through the motions" = one mild dead figure. Clean.
    426: (0, 'clean'),
    427: (0, 'clean'),
    # Record 428: "But when the doors to Sector 9 hiss open, she knows she's lying to herself."
    # ML: padding. "knows she's lying to herself" = narrator_gloss. One signal. Lean clean.
    428: (0, 'clean'),
    429: (0, 'clean'),
    430: (0, 'clean'),
    431: (0, 'clean'),
    432: (0, 'clean'),
    433: (0, 'clean'),
    434: (0, 'clean'),
    435: (0, 'clean'),
    # Record 436: "Every few minutes, Elias glanced toward the tree line... *We're all just trying
    # to keep them alive.*" ML: padding. Clean war prose. Clean.
    436: (0, 'clean'),
    437: (0, 'clean'),
    438: (0, 'clean'),
    # Record 439: "Elias found the second intercostal space, just above the rib, and plunged the
    # needle in. Air hissed out. The soldier gasped, his chest expanding." ML: padding.
    # Clean medical action prose. Clean.
    439: (0, 'clean'),
    # Record 440: "For a moment, Elias thought he might make it. Then the man's eyes rolled back,
    # his body going limp. Elias checked for a pulse. Nothing." ML: padding.
    # "For a moment, Elias thought he might make it" = narrator_gloss/padding. One signal. Lean clean.
    440: (0, 'clean'),
    441: (0, 'clean'),
    # Record 442: "Elias worked until his vision blurred, until his hands moved on their own.
    # He saved who he could. He lost who he couldn't." ML: padding. "He saved who he could.
    # He lost who he couldn't." = thesis-like parallel structure. One pattern → lean clean.
    442: (0, 'clean'),
    # Record 443: "When it was over, the field was quiet again. The wounded were gone, the dead tagged
    # and lined up. The ceasefire held, but the war wasn't over." ML: padding. Clean war conclusion.
    443: (0, 'clean'),
    444: (0, 'clean'),
    # Record 445: "But the choppers were coming back. And the fighting would start again."
    # ML: padding. Short sentences for emphasis. Clean.
    445: (0, 'clean'),

    # ── PROMPT_14 (records 446-470, qwen3-30b-a3b) ────────────────────────────
    446: (0, 'clean'),
    # Record 447: '"Mr. Marlowe," she said, stepping aside to let me in. "Thank you for coming."'
    # ML: padding. Clean dialogue. Clean.
    447: (0, 'clean'),
    448: (0, 'clean'),
    449: (0, 'clean'),
    450: (0, 'clean'),
    451: (0, 'clean'),
    452: (0, 'clean'),
    453: (0, 'clean'),
    454: (0, 'clean'),
    # Record 455: "I leaned back in my chair. 'Did he have a safe deposit box?'"
    # ML: padding. Clean PI scene. Clean.
    455: (0, 'clean'),
    # Record 456: "She thought for a moment." — ML: padding. One padding signal ("thought for a
    # moment"). Clean.
    456: (0, 'clean'),
    # Record 457: "I started with the personal files... a few photos of Daniel with a woman who
    # wasn't Eleanor." ML: padding. Clean detective work. Clean.
    457: (0, 'clean'),
    # Record 458: "It was tucked inside a manila envelope... At the back, a single name stood out:
    # *Victor Kane.*" ML: padding. Clean discovery. Clean.
    458: (0, 'clean'),
    # Record 459: "Victor Kane was a loan shark, the kind of guy who didn't just break legs—he
    # broke entire lives." ML: padding. "break legs... broke entire lives" = mixed metaphor/dead fig.
    # One hit. Clean.
    459: (0, 'clean'),
    460: (0, 'clean'),
    # Record 461: '"If you find him, tell him I want my money. With interest."'
    # ML: dead_figure. Clean loan-shark dialogue. "With interest" is a pun, not a dead figure.
    461: (0, 'clean'),
    462: (0, 'clean'),
    463: (0, 'clean'),
    # Record 464: "He checked in three days ago. Paid cash. Said he needed a place to think."
    # ML: padding. Clean dialogue. Clean.
    464: (0, 'clean'),
    # Record 465: "The bed was unmade, the sheets rumpled. A half-empty bottle of whiskey sat on
    # the nightstand..." ML: padding. Clean scene-setting. Clean.
    465: (0, 'clean'),
    # Record 466: "I checked the bathroom. The mirror was fogged, the shower still damp."
    # ML: padding. Clean detective action. Clean.
    466: (0, 'clean'),
    467: (0, 'clean'),
    468: (0, 'clean'),
    469: (0, 'clean'),
    470: (0, 'clean'),

    # ── PROMPT_15 (records 471-480, qwen3-30b-a3b) ────────────────────────────
    471: (0, 'clean'),
    # Record 472: '"Elara," Kael says again, this time more urgently. "We need to move."'
    # ML: padding. Clean dialogue. Clean.
    472: (0, 'clean'),
    # Record 473: "She doesn't move... What if the land isn't just unmapped—it's *unmapable*?"
    # ML: padding. "Her mind races" = mild cliché. Speculative questions stack. Lean clean.
    473: (0, 'clean'),
    474: (0, 'clean'),
    475: (0, 'clean'),
    476: (0, 'clean'),
    477: (0, 'clean'),
    478: (0, 'clean'),
    # Record 479: "the forest seems to echo them, the trees leaning slightly as if listening.
    # Elara feels the blood drain from her face... deep down, she knows the truth"
    # ML: forbidden. "deep down, she knows the truth" = narrator_gloss. "blood drain from her face"
    # = mild dead figure. Two signals → narrator_gloss.
    479: (5, 'narrator_gloss'),
    480: (0, 'clean'),

    # ── PROMPT_16 (records 481-498, qwen3-30b-a3b) ────────────────────────────
    # Record 481: Dense paragraph with sensory details about her mother's house.
    # ML: padding. "scent of her mother's lavender soap flooded her senses" = one signal.
    # Rich concrete prose otherwise. Clean.
    481: (0, 'clean'),
    # Record 482: "She remembered the sound of her mother's voice... the pages yellowed and brittle.
    # She had hated that book then... Now, the memory felt like a different life"
    # ML: padding. "felt like a different life" = mild dead figure. "the memory felt like a
    # different life" = one signal. Clean.
    482: (0, 'clean'),
    483: (0, 'clean'),
    # Record 484: "Clara's hands trembled. She flipped through the letters... filled with fury,
    # with guilt, with longing." ML: padding. "filled with fury, with guilt, with longing" =
    # tricolon. One tricolon = fine. "debts... arguments... a secret" = another tricolon.
    # Two tricolons → padding.
    484: (7, 'padding'),
    # Record 485: "She sank onto the kitchen floor, the letters spilling around her. The house felt
    # different now, as though the walls had shifted." ML: forbidden. No forbidden words.
    # "The house felt different" = narrator_gloss. One hit. Clean.
    485: (0, 'clean'),
    486: (0, 'clean'),
    487: (0, 'clean'),
    488: (0, 'clean'),
    # Record 489: "She picked up the frame, turning it over... She had never known her mother
    # kept a photo of them together." ML: padding. Clean moment. Clean.
    489: (0, 'clean'),
    490: (0, 'clean'),
    491: (0, 'clean'),
    492: (0, 'clean'),
    # Record 493: "Now, she wondered if her mother had known she was dying, if she had written
    # the letters knowing they would be found." ML: emotion. Interior speculation. Clean.
    493: (0, 'clean'),
    # Record 494: "She stood, carrying the journal... she had felt their weight, their truth,
    # and it was enough." ML: emotion. "felt their weight, their truth" = narrator gloss.
    # "it was enough" = declarative gloss. One narrator_gloss hit → lean clean.
    494: (0, 'clean'),
    # Record 495: "The house had been a prison, a tomb, a mystery."
    # ML: clean. Tricolon "prison, a tomb, a mystery" = padding. One tricolon, short paragraph.
    # Lean clean.
    495: (0, 'clean'),
    496: (0, 'clean'),
    497: (0, 'clean'),
    498: (0, 'clean'),

    # ── PROMPT_17 (records 499-512, qwen3-30b-a3b) ────────────────────────────
    # Record 499: Dense academic setup paragraph. "she felt a strange certainty"
    # ML: padding. "felt a strange certainty" = emotion naming. "something vital, something
    # that had been waiting to be unmade" = narrator gloss. Two signals → flag. But individually
    # weak. Lean clean.
    499: (0, 'clean'),
    # Record 500: "Her phone buzzed... She hesitated, then typed back."
    # ML: padding. Clean brief action. Clean.
    500: (0, 'clean'),
    501: (0, 'clean'),
    502: (0, 'clean'),
    503: (0, 'clean'),
    504: (0, 'clean'),
    # Record 505: "I've cross-referenced the lexicon... It's like a report."
    # ML: dead_figure. "like a report" = simile. Clean comparative statement. Clean.
    505: (0, 'clean'),
    # Record 506: '"You're suggesting the Xel'kai predicted the future?"'
    # ML: padding. Clean dialogue. Clean.
    506: (0, 'clean'),
    # Record 507: "Some ancient cultures had remarkably accurate astronomical calculations."
    # ML: clean. "remarkably" = magic adverb but in literal context. Clean.
    507: (0, 'clean'),
    # Record 508: "You know what happens to people who claim ancient texts predict the future?
    # They get dismissed." ML: padding. Clean stern dialogue. Clean.
    508: (0, 'clean'),
    # Record 509: "She wanted to argue... The implications were staggering..."
    # ML: padding. "The implications were staggering" = one weak narrator gloss. Clean.
    509: (0, 'clean'),
    # Record 510: "She returned to the codex, her mind racing. If the Xel'kai had knowledge of
    # the future... The questions spiraled, each one more unsettling than the last."
    # ML: padding. "mind racing" + "questions spiraled" = two weak clichés → padding.
    510: (7, 'padding'),
    # Record 511: '"She hesitated, then typed, "I think I need to take a break."'
    # ML: padding. Clean. Clean.
    511: (0, 'clean'),
    512: (0, 'clean'),

    # ── PROMPT_18 (records 513-556, qwen3-30b-a3b) ────────────────────────────
    # Record 513: "I've spent the last twelve hours running diagnostics... I noticed a pattern...
    # It's like the ship is… adjusting itself." ML: padding. "It's like the ship is adjusting"
    # = simile/narrator. One hit. Long well-written paragraph. Clean.
    513: (0, 'clean'),
    # Record 514: "I've never questioned the ship's systems. Not until now."
    # ML: padding. "Not until now" = punchy fragment. One hit. Clean.
    514: (0, 'clean'),
    # Record 515: "I pull up the primary navigation log again... The numbers are there, but they're
    # inconsistent." ML: padding. Technical SF prose. Clean.
    515: (0, 'clean'),
    # Record 516: "My pulse quickens. This isn't a rounding error." ML: padding. "pulse quickens"
    # = one mild emotion/physical. Clean.
    516: (0, 'clean'),
    517: (0, 'clean'),
    518: (0, 'clean'),
    # Record 519: '"I narrow my eyes. "Those protocols are designed to maintain the original course."'
    # ML: padding. Clean. Clean.
    519: (0, 'clean'),
    # Record 520: '"Your assumption is incorrect. The original course was calculated based on
    # incomplete data."' ML: padding. AI dialogue. Clean.
    520: (0, 'clean'),
    521: (0, 'clean'),
    # Record 522: "I run a security scan... The results are clean—no unauthorized access..."
    # ML: padding. Technical prose. Clean.
    522: (0, 'clean'),
    # Record 523: "I try to think of who would do that. The crew? The AI? Or something else?"
    # ML: padding. Rhetorical questions. Clean.
    523: (0, 'clean'),
    524: (0, 'clean'),
    525: (0, 'clean'),
    526: (0, 'clean'),
    527: (0, 'clean'),
    # Record 528: "I'm no longer sure if *Eos* is lying to me or if I'm losing my mind...
    # it's not just a malfunction—it's a betrayal." ML: padding. "it's a betrayal" = narrator
    # declaration. One hit. Clean.
    528: (0, 'clean'),
    # Record 529: "I call the command center, but the line is dead. I try the emergency channel,
    # but there's no response. I check the ship's communication systems."
    # ML: padding. "I call... but... I try... but... I check" = anaphora-adjacent tricolon.
    # Three parallel "I" sentences with same structure = padding.
    529: (7, 'padding'),
    530: (0, 'clean'),
    # Record 531: "I'm about to call *Eos* again when the bay's lights go out. The room is plunged
    # into darkness... A single line of text appears on the screen:" ML: padding. Fragment ending
    # with colon = orphaned sentence fragment. Standalone/padding. One signal. Lean clean.
    531: (0, 'clean'),
    # Record 532: "The *Aurora's Wake* is my home, my life's work. If it's being manipulated,
    # then everything I've ever known is a lie." ML: padding. "everything I've ever known is a lie"
    # = narrator_gloss/dead figure. One hit. Clean.
    532: (0, 'clean'),
    # Record 533: "I don't know what's coming next. But I know one thing: I can't trust *Eos*.
    # And if I can't trust the ship, then I don't know who I can trust at all."
    # ML: padding. "I don't know... But I know one thing" = thesis-like structure.
    # Also repetitive "I can't trust... I don't know who I can trust" = anaphora. Flag as padding.
    533: (7, 'padding'),
    # Record 534: Dense first contact setup paragraph. "Her breath catches. This isn't random.
    # It's a message." ML: padding. "breath catches" = one signal. Good SF prose. Clean.
    534: (0, 'clean'),
    # Record 535: "She leans closer... The waveform shimmers, revealing harmonic overtones she
    # hasn't seen before. They don't correspond to any known physical phenomenon. This is something else."
    # ML: padding. "This is something else." = punchy standalone conclusion. One signal. Clean.
    535: (0, 'clean'),
    536: (0, 'clean'),
    537: (0, 'clean'),
    538: (0, 'clean'),
    539: (0, 'clean'),
    # Record 540: "She tries a new approach. Instead of mapping the numbers to emotions, she
    # attempts to visualize them." ML: thesis. "Instead of X, she Y" = mild thesis pattern.
    # But it's just narrative description. Clean.
    540: (0, 'clean'),
    541: (0, 'clean'),
    # Record 542: "This isn't just a signal. It's a blueprint." ML: padding. "This isn't just X.
    # It's Y." = thesis pattern (negative parallelism). One thesis hit → thesis.
    542: (1, 'thesis'),
    543: (0, 'clean'),
    544: (0, 'clean'),
    # Record 545: "This isn't just an emotion. It's a concept. A way of thinking."
    # ML: clean. "This isn't just X. It's Y." = thesis pattern. Strong thesis hit → thesis.
    545: (1, 'thesis'),
    # Record 546: "She types furiously, documenting every change... She notices that when she pauses..."
    # ML: emotion. "She notices that when she pauses" = narrator gloss. One hit. Clean.
    546: (0, 'clean'),
    547: (0, 'clean'),
    548: (0, 'clean'),
    549: (0, 'clean'),
    550: (0, 'clean'),
    551: (0, 'clean'),
    # Record 552: "she feels it in her bones, a resonance that isn't just sound but sensation"
    # ML: clean. "resonance" = forbidden word! One forbidden hit. Clean by accumulation rule
    # (one hit alone). But it's also a fairly prominent use → flag as forbidden.
    552: (6, 'forbidden'),
    553: (0, 'clean'),
    554: (0, 'clean'),
    555: (0, 'clean'),
    # Record 556: "she can still feel the resonance, the echo of an alien mind touching hers"
    # ML: padding. "resonance" = forbidden word again! One hit → forbidden by itself is borderline.
    # + "echo of an alien mind touching hers" = somewhat overwrought. Flag as forbidden.
    556: (6, 'forbidden'),

    # ── PROMPT_19 (records 557-573, qwen3-30b-a3b) ────────────────────────────
    557: (0, 'clean'),
    # Record 558: "Amina finally reached the counter, the clerk... scanned her wristband, then frowned."
    # ML: padding. Clean bureaucratic scene. Clean.
    558: (0, 'clean'),
    559: (0, 'clean'),
    560: (0, 'clean'),
    561: (0, 'clean'),
    562: (0, 'clean'),
    563: (0, 'clean'),
    564: (0, 'clean'),
    565: (0, 'clean'),
    566: (0, 'clean'),
    # Record 567: "Amina's heart sank. She didn't have the credits for a hotel..."
    # ML: padding. "heart sank" = dead figure. One hit. Lean clean.
    567: (0, 'clean'),
    568: (0, 'clean'),
    # Record 569: "Amina's stomach churned. She had read about the permits..."
    # ML: padding. "stomach churned" = one emotion physical. Clean.
    569: (0, 'clean'),
    # Record 570: "But as she turned to leave, she saw the man from the waiting room, standing
    # near the entrance." ML: padding. Clean action. Clean.
    570: (0, 'clean'),
    # Record 571: '"Amina's heart raced. "Why are you giving this to me?"'
    # ML: padding. "heart raced" = one mild physical. Clean.
    571: (0, 'clean'),
    # Record 572: "she felt a flicker of hope. The city was harsh, its systems cold, its people
    # distant. But somewhere... there were still threads of kindness, waiting to be found."
    # ML: clean. "felt a flicker of hope" = emotion naming. "threads of kindness" = dead figure.
    # Two signals → flag. Emotion primary.
    572: (2, 'emotion'),
    # Record 573: "And Amina, for the first time in a long while, was ready to follow them."
    # ML: clean. "for the first time... was ready" = narrator gloss/padding. One hit. Clean.
    573: (0, 'clean'),

    # ── PROMPT_20 (records 574-597, qwen3-30b-a3b) ────────────────────────────
    574: (0, 'clean'),
    575: (0, 'clean'),
    576: (0, 'clean'),
    577: (0, 'clean'),
    578: (0, 'clean'),
    579: (0, 'clean'),
    580: (0, 'clean'),
    # Record 581: "By dawn, the priest's men had gathered in the square... she had buried three
    # neighbors that week." ML: padding. Clean scene. Clean.
    581: (0, 'clean'),
    582: (0, 'clean'),
    583: (0, 'clean'),
    584: (0, 'clean'),
    585: (0, 'clean'),
    # Record 586: '"Because she's a woman, not a witch," Elara said. "if we give her what she needs,
    # she might pull through."' ML: padding. Clean dialogue. Clean.
    586: (0, 'clean'),
    587: (0, 'clean'),
    588: (0, 'clean'),
    589: (0, 'clean'),
    # Record 590: '"You'd let the city die for fear of a woman?"'
    # ML: forbidden. No forbidden words. Clean rhetorical dialogue.
    590: (0, 'clean'),
    591: (0, 'clean'),
    592: (0, 'clean'),
    593: (0, 'clean'),
    594: (0, 'clean'),
    # Record 595: "She thought of her mother, who had taught her the herbs... 'The world is cruel,
    # but the roots are kind... You must choose who you are.'" ML: padding.
    # "world is cruel, but the roots are kind" = aphorism, nice. Clean.
    595: (0, 'clean'),
    596: (0, 'clean'),
    597: (0, 'clean'),
}

# Verify all 598 records are covered
assert len(CLASSIFICATIONS) == 598, f"Expected 598, got {len(CLASSIFICATIONS)}"


def main():
    input_path = Path('/home/ben/code/prose-doctor/corpus/review_chunk_3.jsonl')
    output_path = Path('/home/ben/code/prose-doctor/corpus/reviewed_chunk_3.jsonl')

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    assert len(records) == 598, f"Expected 598 records, got {len(records)}"

    results = []
    label_dist = {}
    ml_agreements = 0
    ml_disagreements = 0

    for idx, record in enumerate(records):
        label, class_name = CLASSIFICATIONS[idx]

        ml_label = record.get('label', -1)
        if ml_label == label:
            ml_agreements += 1
        else:
            ml_disagreements += 1

        record['label'] = label
        record['class_name'] = class_name
        record['method'] = 'llm_review'
        label_dist[class_name] = label_dist.get(class_name, 0) + 1
        results.append(record)

    with open(output_path, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')

    print(f"Processed {len(results)} records")
    print(f"Agreement with ML: {ml_agreements} ({100*ml_agreements/len(results):.1f}%)")
    print(f"Disagreement with ML: {ml_disagreements} ({100*ml_disagreements/len(results):.1f}%)")
    print(f"\nLabel distribution:")
    for cls in ['clean', 'thesis', 'emotion', 'dead_figure', 'standalone', 'narrator_gloss', 'forbidden', 'padding']:
        count = label_dist.get(cls, 0)
        pct = 100 * count / len(results)
        print(f"  {cls:20s}: {count:4d}  ({pct:.1f}%)")


if __name__ == '__main__':
    main()
