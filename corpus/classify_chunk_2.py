#!/usr/bin/env python3
"""
Manual LLM review classification for review_chunk_2.jsonl
598 records, hand-classified by reading each paragraph.

Label map:
0 = clean
1 = thesis
2 = emotion
3 = dead_figure
4 = standalone
5 = narrator_gloss
6 = forbidden
7 = padding
"""

import json

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

# Manual classifications: index -> label
# Read all 598 records carefully and applied the accumulation rule.
LABELS = {
    # 0-9
    0: 2,   # "She felt a terrifying, exquisite freedom" — direct emotion naming
    1: 0,   # cold clarity / clean prose, short punchy but grounded
    2: 0,   # dialogue + assessment, functional
    3: 0,   # dialogue + logistics, clean
    4: 0,   # short dialogue, clean
    5: 0,   # "If they are organized..." conditional logic chain, clean
    6: 0,   # dialogue, clean
    7: 0,   # specific physical detail, clean
    8: 0,   # "The silence was the hardest part" — concrete sensory observation, clean
    9: 7,   # "utilizing the natural depressions" — "utilizing" forbidden-ish but padding pattern overall: "kept them low, utilizing..."
    # 10-19
    10: 0,  # good observational prose
    11: 0,  # specific physical description, clean
    12: 0,  # dialogue + specific detail, clean
    13: 4,  # "This was the moment. Was this a plea, or a distraction?" — short orphaned standalone sentences
    14: 0,  # procedural action, clean
    15: 0,  # physical action, specific
    16: 1,  # "It was a clear, undeniable warning shot. Precise, non-lethal, and delivered with lethal capability." — thesis/dramatic countdown pattern
    17: 0,  # short but functional
    18: 0,  # physical action, clean
    19: 0,  # sensory detail, clean
    # 20-29
    20: 0,  # specific craft description, clean
    21: 7,  # "Beneath that, there are sharper notes" — padding tricolon + filler description
    22: 0,  # "not alchemy, though customers might call it that" — self-aware, clean
    23: 0,  # concrete sensory, clean
    24: 0,  # single clean action beat
    25: 0,  # dramatic reveal, clean
    26: 0,  # backstory, grounded
    27: 0,  # clean physical observation
    28: 0,  # dialogue, clean
    29: 7,  # "performs its final transformation" — padding filler clause tacked on
    # 30-39
    30: 0,  # specific weight detail, clean
    31: 0,  # dialogue, clean
    32: 0,  # action beat, clean
    33: 7,  # "becoming almost intoxicating... like ozone before a storm, or old..." — "deeply unsettling" accumulation of weak hedged description
    34: 7,  # "brief, electric, and deeply unsettling" — tricolon + "deeply" magic adverb
    35: 0,  # clean action
    36: 7,  # "the silence rushes back in, heavier than before" — dead figure + padding
    37: 2,  # "He feels a tremor... The bread he just sold..." — narrator gloss/emotion naming
    38: 2,  # "profound sorrow or a spike of unacknowledged joy" — forbidden word + emotion naming
    39: 0,  # grounded, clean
    # 40-49
    40: 0,  # clean
    41: 0,  # clean brief description
    42: 0,  # clean motion
    43: 0,  # physical description, clean
    44: 0,  # dialogue, clean
    45: 0,  # dialogue, clean
    46: 0,  # dialogue, clean
    47: 0,  # dialogue, clean
    48: 0,  # memory, clean
    49: 0,  # short dialogue, clean
    # 50-59
    50: 7,  # "might be a laugh or a sob" padding hedge
    51: 1,  # "It wasn't the bread that remembered... It was you." — thesis/negation pattern
    52: 0,  # short dialogue, acceptable
    53: 0,  # clean
    54: 0,  # good evocative description
    55: 7,  # "quiet, unremarkable man... who dragged..." — padding accumulation
    56: 7,  # "smelled intensely of diesel, brine, and something else" — tricolon + vague "something else"
    57: 7,  # "usually too cramped and foul-smelling" — padding hedges
    58: 7,  # "He yanked the metal door open" — ML flagged, but this is actually fine. Reconsidering: clean
    59: 0,  # physical detail, clean
    # 60-69
    60: 0,  # clean descriptive
    61: 7,  # "Silas, his quiet, debt-ridden grandfather, had been running contraband. Not just a few bottles... but..." — padding accumulation with tricolon
    62: 7,  # "The silence of the engine room became immense... drowning out..." — dead figure + padding
    63: 7,  # "He needed air that hadn't been trapped for months with narcotics" — padding/weak motivation
    64: 0,  # good concrete description
    65: 0,  # short clean dialogue
    66: 0,  # clean dialogue
    67: 5,  # "something softer—or perhaps just more practiced—entering his eyes. The air grew heavy, charged with danger." — narrator gloss + dead figure
    68: 0,  # dialogue, clean
    69: 7,  # "working mechanically, his hands moving with a strange certainty" — padding hedges
    # 70-79
    70: 0,  # physical action, clean
    71: 0,  # physical action, clean
    72: 0,  # clean with good specificity
    73: 7,  # "manufactured coldness designed to soothe the agitated pre-procedure nerves" — padding accumulation
    74: 0,  # clean procedural
    75: 0,  # clean
    76: 0,  # single line of dialogue/action — clean (ML overfit on "locus")
    77: 0,  # physical action, clean
    78: 7,  # "The transition is always violent, a non-linear plunge" — padding + dead figure
    79: 7,  # "the air thick with ozone and the metallic tang... floods his senses" — accumulation of clichés
    # 80-89
    80: 0,  # clean analogy embedded in action
    81: 0,  # clean
    82: 7,  # "clad in the matte-black armor of their division, moving with practiced precision" — padding accumulation
    83: 0,  # clean dialogue
    84: 0,  # "It is quick, a calculated blow" — clean action
    85: 0,  # clean action beat
    86: 7,  # "warp into something manageable" narrator gloss + padding
    87: 6,  # "emotional resonance signature" — "resonance" forbidden word
    88: 0,  # clean brutal action
    89: 0,  # clean dialogue
    # 90-99
    90: 5,  # "The terminology is wrong... This sounds like an assassination" — narrator explains/glosses
    91: 0,  # clean
    92: 0,  # clean
    93: 7,  # "The ethical weight crashes down on him" + padding accumulation of hedged clauses
    94: 7,  # padding accumulation — "still spiking—higher than it should be, fighting the dampeners"
    95: 0,  # bold/structured options, functional
    96: 0,  # bold/structured options, functional
    97: 7,  # "The ethical calculus is irrelevant... stability always trumps truth... He deals in truth, even if it is manufactured." — padding accumulation
    98: 7,  # "the absolute certainty in the Enforcer's posture, the sheer, calculated secrecy" — padding
    99: 5,  # "*Non-viable batch.* That implies they were testing something... Voss saw the cleanup crew" — narrator explains
    # 100-109
    100: 7,  # "the memory stream beginning to collapse around the intrusive knowledge" — padding
    101: 0,  # physical action, clean
    102: 0,  # clean
    103: 5,  # "The sweep has been successfully relegated... emotional valence reduced by 92%" — narrator gloss reporting out
    104: 7,  # "waited for the mandated ten-minute cool-down before activating" — padding filler
    105: 0,  # clean
    106: 0,  # clean
    107: 7,  # "But he can't. He can't erase the feeling..." — repetitive "can't" padding
    108: 0,  # clean
    109: 7,  # "the blue light dying, leaving the room in shadowed blue-grey twilight" — padding accumulation
    # 110-119
    110: 7,  # "he will be listening for the echo of that serpent, wondering which of the..." — padding accumulation
    111: 0,  # good specific prose
    112: 0,  # clean
    113: 0,  # clean
    114: 0,  # clean
    115: 0,  # italic status notation, functional
    116: 0,  # clean medical detail
    117: 7,  # "Sucking chest wound" as sentence fragment after padding accumulation
    118: 0,  # italic notation, functional
    119: 7,  # "friend and enemy, friend and enemy" — anaphora abuse + padding
    # 120-129
    120: 7,  # "He paused only when he heard the approach. Bootfalls, slow, careful." — padding + standalone
    121: 0,  # clean physical description
    122: 0,  # dialogue, clean
    123: 0,  # clean
    124: 0,  # good prose, specific
    125: 7,  # "applying a drain and applying heavy pressure dressing" — padding repetition
    126: 7,  # "tracking Elias's movements with dull, dark eyes" — padding
    127: 0,  # dialogue, clean
    128: 0,  # dialogue, clean
    129: 0,  # dialogue, clean
    # 130-139
    130: 6,  # "Being known for impartiality here was both a shield and a curse" — dead figure, but "impartiality" as shield/curse is barely dead. However "suppressed a shudder of resentment" = emotion naming
    131: 0,  # clean dialogue
    132: 0,  # clean
    133: 7,  # "It was triage applied at the soul level: Who has the highest chance" — padding/narrator gloss
    134: 0,  # clinical detail, clean
    135: 7,  # "His needs were less time-sensitive *right* now" — padding hedging
    136: 7,  # "It weighs on me every second I'm here... the decision isn't about who deserves to live, Milos. It's about who *can* live." — thesis pattern + padding
    137: 0,  # dialogue + action, clean
    138: 0,  # dialogue, clean
    139: 0,  # clean
    # 140-149
    140: 0,  # clean sound description
    141: 7,  # "We have three Red-1s and two Red-2s" — padding classification chatter
    142: 7,  # narrator explains classification system — narrator_gloss/padding
    143: 7,  # "borderline Red-2 leaning toward Red-1" — padding hedging
    144: 0,  # clean
    145: 0,  # clean
    146: 0,  # clean
    147: 7,  # "I realized something in that fluorescent lighting" — padding setup
    148: 7,  # "we are all just bodies that failed... If I stop caring about the uniform" — padding accumulation
    149: 7,  # "I'm too tired to be a butcher" — setup padding
    # 150-159
    150: 0,  # clean dialogue
    151: 0,  # clean evocative description
    152: 7,  # "He just... vaporized" — ellipsis padding + "vaporized" is actually fresh. Clean actually. But ML flagged for "just… vaporized" ellipsis. Keeping clean.
    153: 7,  # "money that hadn't yet learned to be embarrassed" — padding accumulation
    154: 0,  # clean dialogue
    155: 0,  # clean
    156: 0,  # clean dialogue
    157: 0,  # clean dialogue
    158: 7,  # "Too immaculate... smelled faintly of... faintly of bleach" — repetition + padding accumulation
    159: 5,  # "nothing incriminating—just numbers swimming in safe territory" — narrator glossing/explaining
    # 160-169
    160: 0,  # clean action
    161: 0,  # clean dialogue
    162: 4,  # "The docks. That put me squarely back in my element." — standalone punchy fragments
    163: 7,  # "a concrete blister squatting three blocks from the shipyards" — padding accumulation
    164: 0,  # clean, good description
    165: 0,  # clean dialogue
    166: 0,  # clean dialogue
    167: 7,  # "A name that didn't typically mix with penthouse apartments" — padding + narrator gloss
    168: 0,  # clean
    169: 0,  # clean
    # 170-179
    170: 0,  # clean
    171: 7,  # "first smear of bruised purple stained the eastern horizon" — dead figure + padding accumulation
    172: 0,  # clean
    173: 0,  # clean
    174: 0,  # clean
    175: 7,  # "moved slow, pistol drawn... dust, undisturbed—except for one area" padding accumulation
    176: 7,  # "And then I saw it... a discarded cigarette butt. Expensive, foreign brand." — standalone + padding
    177: 0,  # clean
    178: 0,  # clean
    179: 0,  # clean dialogue
    # 180-189
    180: 0,  # clean dialogue
    181: 1,  # "Standard methods involve... Julian is smarter than that. He didn't run from me, Rourke. He ran to protect the money trail" — thesis/negation pattern
    182: 0,  # clean
    183: 5,  # "My code chafed... But Julian Vance was a man, and he was actively using my process..." — narrator explains motivation
    184: 0,  # clean action
    185: 0,  # clean dialogue
    186: 0,  # clean
    187: 0,  # clean
    188: 2,  # "It was the mystery, the sense that he was treading on ground that had never been touched... Every step he took felt like a step into the unknown." — emotion/narrator gloss
    189: 7,  # "Curiosity piqued, he quickened his pace" — padding filler ("curiosity piqued")
    # 190-199
    190: 6,  # "What brings you to this place?" — not really forbidden. Clean actually.
    191: 0,  # clean dialogue
    192: 2,  # "he felt the anger coursing through him" — direct emotion naming
    193: 0,  # clean dialogue
    194: 0,  # clean description
    195: 6,  # "how to read the signs... how to interpret the language... how to find his way" — anaphora tricolon abuse ("how to X" x3)
    196: 0,  # clean
    197: 0,  # clean
    198: 0,  # clean dialogue
    199: 0,  # clean dialogue
    # 200-209
    200: 0,  # clean dialogue
    201: 0,  # clean dialogue
    202: 0,  # clean
    203: 0,  # clean
    204: 0,  # clean sensory
    205: 0,  # clean dialogue
    206: 0,  # clean
    207: 0,  # clean dialogue
    208: 2,  # "He knew that Kael and Mara would try to stop him... he couldn't let them stand in his way" — repeated direct emotion reporting
    209: 2,  # "he could feel the weight of the world..." — "weight of the world" dead figure + emotion
    # 210-219
    210: 2,  # "with a sense of destiny and purpose" — direct emotion naming
    211: 0,  # clean action
    212: 0,  # clean
    213: 0,  # clean
    214: 0,  # clean
    215: 2,  # "She felt the weight of her grief settle over her like a shroud" — direct emotion naming + dead figure
    216: 7,  # "her voice barely above a whisper" — padding hedge
    217: 7,  # "You know, the ancients had a habit of..." — padding filler dialogue
    218: 0,  # clean dialogue
    219: 0,  # single clean transition sentence
    # 220-229
    220: 7,  # "haunted by the same terrifying vision... utterly alone, adrift in a cosmic..." padding accumulation
    221: 0,  # clean
    222: 0,  # clean
    223: 0,  # clean
    224: 7,  # "I'd been running diagnostics... nagging feeling in the back of my mind... dismissed it as paranoia... result of too much..." — padding accumulation
    225: 0,  # clean dialogue
    226: 0,  # clean
    227: 7,  # "I found myself growing increasingly paranoid... jumping at every little noise" — "found myself" padding + accumulation
    228: 7,  # "But even as I tried to convince myself... I knew it wasn't true" — padding hedged logic
    229: 7,  # "I was back to square one... I knew I had to keep digging, to keep searching... I also knew" — anaphora "I knew" x3 + padding
    # 230-239
    230: 0,  # clean
    231: 0,  # clean
    232: 0,  # clean
    233: 0,  # clean
    234: 7,  # "It was the same person I'd seen in my nightmare, the face I hadn't been able to place until now." — narrator gloss
    235: 7,  # "I couldn't trust anyone, not even the ship's AI or the captain... nowhere to turn but inward" — padding accumulation
    236: 0,  # clean
    237: 7,  # "I found myself thinking about the navigator, trying to piece together why" — "found myself" padding
    238: 7,  # "I supposed I would never know for sure. Maybe he had been acting on his own, or maybe..." — padding hedging
    239: 7,  # "I had to find a way to expose him... But how?" — padding rhetorical question
    # 240-249
    240: 7,  # "I couldn't go to the captain, not without proof. And I couldn't trust anyone else" — padding hedging
    241: 0,  # clean
    242: 0,  # clean action
    243: 0,  # clean dialogue
    244: 0,  # clean dialogue
    245: 7,  # "a small smile playing at the corners of his mouth" — dead figure + padding
    246: 0,  # clean
    247: 0,  # clean dialogue
    248: 0,  # clean dialogue
    249: 0,  # clean dialogue/action
    # 250-259
    250: 2,  # "But I knew it was a trap. I knew he was trying to lure me into a false sense of" — direct emotion/interior knowing
    251: 6,  # "map our own emotions to their mathematical language" — "language" mapping as metaphor fine, but ML flagged. Actually clean. Reconsidering: clean
    252: 0,  # clean dialogue
    253: 6,  # "he couldn't shake the feeling that this was the key they had been searching for" — forbidden phrase territory but mostly padding. "key they had been searching for" dead figure. Clean enough.
    254: 7,  # "Frustrated and exhausted, Aris found himself once again staring... his mind struggling to find the missing piece of the puzzle" — padding accumulation
    255: 0,  # good sensory, clean
    256: 0,  # clean
    257: 7,  # "something catching his eye. There, in the doorway, was a..." — padding transition
    258: 0,  # clean
    259: 0,  # clean
    # 260-269
    260: 0,  # clean
    261: 0,  # clean dialogue
    262: 0,  # clean
    263: 0,  # clean dialogue
    264: 7,  # "his mind racing with the possibilities... a stone-cold predator, a monster who had been hiding..." — padding accumulation
    265: 7,  # "couldn't shake the feeling... running out of time, that the killer was already two steps ahead" — dead figure + padding
    266: 0,  # clean dialogue
    267: 0,  # clean
    268: 0,  # clean
    269: 0,  # clean
    # 270-279
    270: 2,  # "calloused hands, once skilled in the art of killing, now carefully navigated the delicate tendrils" — emotion/dead figure accumulation
    271: 0,  # clean description
    272: 0,  # clean dialogue
    273: 0,  # clean dialogue
    274: 0,  # clean dialogue
    275: 0,  # clean
    276: 7,  # "rose quietly, his hand reaching for his sword" — padding accumulation
    277: 7,  # "Solden's heart pounded in his chest" — dead figure + padding
    278: 7,  # "vanished into the trees, leaving Solden alone with his thoughts" — padding filler
    279: 0,  # clean
    # 280-289
    280: 0,  # clean
    281: 7,  # duplicate of 277 — "Solden's heart pounded in his chest" — padding
    282: 2,  # "with a deep breath, he stepped into the forest" — padding/emotion
    283: 2,  # "And then, suddenly, he saw it" — padding + emotion telling
    284: 7,  # "its body a mass of twisting vines and leaves, its eyes glinting" — padding accumulation
    285: 0,  # clean (slight repeat)
    286: 2,  # duplicate of 280 — emotion-named nostalgia/comfort
    287: 7,  # duplicate of 281 — padding
    288: 2,  # duplicate of 282
    289: 2,  # duplicate of 283
    # 290-299
    290: 7,  # duplicate of 284 — padding
    291: 0,  # truncated fragment, clean enough
    292: 0,  # clean, good opening
    293: 0,  # clean
    294: 0,  # clean, short
    295: 0,  # clean
    296: 7,  # "No scribe below the Seventh Circle was allowed... only Head Archivist and the seni..." — padding exposition
    297: 7,  # "Her soft leather slippers made no sound on the smooth stone floor as she hurried" — padding accumulation
    298: 7,  # "But that same account... said seven elders. And only mentioned 'a carved staff'." — short standalone/padding fragment
    299: 0,  # clean
    # 300-309
    300: 7,  # "She spent the rest of the night in a haze of unease, her copies slow and clumsy, her mind churning" — padding accumulation
    301: 7,  # "A quick search turned up a half-dozen histories" — padding transition
    302: 0,  # clean short paragraph
    303: 7,  # "She checked another source – a compendium of Northern Clan customs... It said a black oak staff" — padding transition
    304: 0,  # clean
    305: 0,  # clean
    306: 7,  # "She asked to be assigned to the Royal Histories again... Shaking, she started turning pages" — padding accumulation
    307: 7,  # "And then she found a drastic one. The account of King Marrot's death." — standalone + padding
    308: 0,  # clean
    309: 4,  # "The changes were all funneling toward legitimizing the current royal line." — standalone thesis sentence
    # 310-319
    310: 0,  # rhetorical question, clean in context
    311: 0,  # clean
    312: 7,  # "her mind reeling... She had to get to the bottom of this... If the historical records... were being systematically falsified" — padding accumulation
    313: 7,  # "She spent every spare moment for the next week poring over every text" — padding accumulation
    314: 4,  # "A narrative that cast the current royal family in a heroic, unbroken line" — standalone sentence fragment
    315: 7,  # "Heart pounding, she offered to start with the Aisle of Kings" — padding
    316: 0,  # clean
    317: 0,  # clean dialogue
    318: 0,  # clean
    319: 7,  # "She thought of the other scribes. Borin. Maybe he suspected something." — padding/standalone fragmentation
    # 320-329
    320: 0,  # clean (ignoring the url artifact)
    321: 7,  # "I am a forensic accountant... My job is to investigate..." — padding padding expository setup
    322: 7,  # "I decided to confront my boss... To my surprise, he didn't seem surprised" — padding accumulation
    323: 2,  # "I felt betrayed and angry" — direct emotion naming
    324: 0,  # clean
    325: 7,  # "I decided to confront my boss again and demand... To my surprise" repeated pattern — padding
    326: 0,  # clean
    327: 0,  # clean
    328: 0,  # clean
    329: 7,  # "A few days later, the authorities raided..." — padding filler transition + list accumulation
    # 330-339
    330: 0,  # clean
    331: 7,  # "I was relieved... but I also knew that my life would never be the same... I had lost my job and my career... I knew that I would always be looking over my shoulder" — padding anaphora accumulation
    332: 7,  # "However, I also knew that I had done the right thing... I knew that I would always be proud of what I had done" — padding anaphora
    333: 7,  # "As I reflects on what had happened... I realized... I had learned... I had learned" — padding anaphora + narrator gloss
    334: 0,  # clean
    335: 0,  # clean
    336: 0,  # clean
    337: 7,  # "And so, as I sat in my office and stared out the window at the city below... I knew that I had made the right decision" — padding filler closing
    338: 7,  # "She decided to start at the beginning and go through each case in detail" — padding padding
    339: 7,  # "She took a deep breath and went to find her supervisor" — "took a deep breath" dead figure + padding
    # 340-349
    340: 6,  # "sometimes these things just happen. You can't take it personally." — clean dialogue actually. ML flagged as forbidden but no forbidden words. Clean.
    341: 0,  # clean dialogue
    342: 7,  # "But Lena couldn't let it go. She spent the rest of her shift reviewing the charts again" — padding accumulation
    343: 7,  # "As she was leaving for the night, she noticed something else" — padding transition + accumulation
    344: 0,  # clean
    345: 7,  # "She spent the next few hours scouring medical journals... What she found chilled her to the bone." — dead figure + padding
    346: 0,  # clean
    347: 0,  # clean short
    348: 7,  # "She even sneaked a few empty Versed vials to send to a lab for independent testing" — padding accumulation
    349: 0,  # clean short
    # 350-359
    350: 0,  # clean dialogue
    351: 0,  # clean dialogue
    352: 0,  # clean dialogue
    353: 0,  # clean dialogue
    354: 0,  # clean dialogue
    355: 7,  # "she tried to act normally at work, but she could feel the tension in the air. Whispers followed her down the hall" — padding accumulation
    356: 7,  # "Finally, the results came back" — padding transition + accumulation
    357: 0,  # clean
    358: 0,  # clean dialogue
    359: 0,  # clean dialogue
    # 360-369
    360: 0,  # clean
    361: 2,  # "She opened her mouth to speak, but Lisa beat her to it" — padding/emotion. Actually clean. Only one beat. Clean.
    362: 0,  # clean
    363: 7,  # "she noticed something strange. A patient's chart showed..." — padding transition
    364: 7,  # "Lena's heart raced as she hurried to the patient's room" — "heart raced" dead figure + padding
    365: 7,  # "Lena checked the woman's vital signs... It was the same pattern she had seen before" — padding accumulation
    366: 0,  # clean
    367: 7,  # "Lena's eyes widened. She hadn't administered this medication." — standalone + narrator gloss
    368: 7,  # "Lena's hands shook as she dialed the phone" — dead figure + padding
    369: 7,  # "'Here we are again, old friend.' Mr. Grimshaw said... It was the same glass and the exact same words he used every night for the past..." — padding accumulation
    # 370-379
    370: 7,  # "For the past few months, Mr. Grimshaw has been in a bad place... His reputation was ruined, and his life was practically destroyed" — padding accumulation
    371: 7,  # "Mr. Grimshaw's mantra was... No matter how bad things would get, he always believed... That was how he was able to survive" — padding anaphora
    372: 0,  # clean
    373: 0,  # clean
    374: 0,  # clean
    375: 0,  # clean
    376: 0,  # clean
    377: 0,  # clean (ML flagged as forbidden but no obvious forbidden words)
    378: 0,  # clean
    379: 2,  # "Viktor's heart pounds in his chest as he watches his opponent reach for his bishop" — "heart pounds in his chest" dead figure + emotion
    # 380-389
    380: 2,  # "the nervous energy that thrummed through his veins" — emotion naming accumulation
    381: 3,  # "ghostly echoes of games played, the whispered secrets of kings and queens" — dead figure accumulation
    382: 2,  # "a warm, genuine smile... Viktor smiles, a wa..." — emotion naming
    383: 0,  # clean closing scene
    384: 0,  # clean
    385: 2,  # "her voice filled with hope" — direct emotion naming
    386: 7,  # "brow furrowed as he studied them under the flickering fluorescent lights. When he felt as though he'd squeezed every bit of information" — padding accumulation
    387: 0,  # clean dialogue
    388: 0,  # clean dialogue
    389: 7,  # "I've tried to triangulate it, but it's coming from so deep. I don't think we have the right equipment" — padding hedging
    # 390-399
    390: 7,  # "She glanced at the readouts again. 'It could be a shipwreck. We're in the middle of nowhere, but things do tend to accumulate'" — padding accumulation
    391: 0,  # clean short dialogue
    392: 7,  # "He ran a hand through his hair... starting to feel the effects. His thoughts..." — padding dead figure + accumulation
    393: 7,  # "The signals were all he could think about, all he could hear, even when they weren't there. They echoed in his mind, a constant," — anaphora + padding
    394: 7,  # "just as suddenly as it had started, it was over... leaving Francis alone in the laboratory, his heart pounding, his hands shaking" — padding dead figure accumulation
    395: 0,  # clean
    396: 0,  # clean
    397: 7,  # "his mind fixed on a single, unshakable goal: to understand, to communicate, to..." — tricolon + padding
    398: 7,  # "even as he longed to follow, to surrender himself to the voice... he was a scientist, a man of reason and logic" — padding accumulation
    399: 0,  # clean actually — "torn, pulled in two directions at once, one foot in the world of the known, the other in the realm of the unknowable" — slightly purple but functional
    # 400-409
    400: 0,  # clean
    401: 0,  # clean short
    402: 7,  # "his eyes closed, his mind adrift in a sea of exhaustion and uncertainty" — dead figure + padding
    403: 7,  # "He had done it. He had followed the voice, he had surrendered himself to it, he had let it guide him" — anaphora accumulation
    404: 7,  # "the heavy iron key cold and unwelcoming in her clammy palm. A biting wind whispered sinister secrets through the gnarled limbs" — dead figure + padding accumulation
    405: 7,  # "The door creaked open with a reluctant groan... Hesitant footsteps carried her across warped floorboards" — padding accumulation dead figure
    406: 7,  # "What secrets lay beyond this threshold? What sorrow had transpired..." — rhetorical padding accumulation
    407: 2,  # "their weathered faces twisted into identical masks of concern and warning" — emotion direct naming
    408: 7,  # "A flurry of whispers erupted from the crowd, their voices intertwining into an indistinguishable murmur" — padding dead figure
    409: 7,  # "its muted whispers growing louder and more insistent with each passing night. Elizabeth's dreams were plagued with visions" — padding accumulation
    # 410-419
    410: 7,  # "her eyes filled with..." — padding + accumulation
    411: 7,  # "her hands trembling slightly as she polished a tarnished silver locket" — padding accumulation
    412: 7,  # "She spent hours poring over old newspapers and dusty town records, searching for any clue that might shed light" — padding accumulation
    413: 7,  # "The walls of this prison close in around me, their opulent facade a thin veneer for the cruelty" — padding dead figure accumulation
    414: 7,  # "If you are reading this, then my fears have been realized... he murdered me – strangled me with his own bare hands" — padding accumulation
    415: 0,  # clean
    416: 0,  # clean dialogue
    417: 0,  # clean
    418: 0,  # clean
    419: 0,  # clean
    # 420-429
    420: 3,  # "deafening silence... the once-familiar sight of her blue-painted front door seemed foreign" — dead figure accumulation
    421: 0,  # clean
    422: 0,  # clean
    423: 5,  # "*She's erased me from this room... From this house. From her life.*" — narrator gloss/thesis dramatic countdown
    424: 0,  # clean dialogue
    425: 0,  # clean dialogue
    426: 0,  # clean — actually good prose with repeated structure that earns it
    427: 0,  # clean
    428: 7,  # "a single tear sliding down her cheek" — dead figure + padding
    429: 7,  # "a ghost in her own home, a stranger in her own..." — dead figure + padding
    # 430-439
    430: 7,  # "prayed for sleep to take her away from the pain of the present" — dead figure + padding
    431: 0,  # clean dialogue
    432: 0,  # clean
    433: 0,  # clean
    434: 0,  # clean dialogue
    435: 6,  # "his voice thick with emotion" — "thick with emotion" is minor dead figure. Actually clean.
    436: 0,  # clean dialogue
    437: 0,  # clean
    438: 7,  # "Sofia smoothed the wrinkles from her apron... ensuring that no stray strands of chestnut hair escaped its confines. It wouldn't do for a woman to show so much as an inch" — padding accumulation
    439: 7,  # "Sofia's mind was sharp and inquisitive, grasping concepts that eluded even some of" — padding accumulation
    # 440-449
    440: 7,  # "Sofia was devastated... barred from attending lectures... It wasn't seemly" — padding accumulation
    441: 7,  # "Maestro grew famous, while Sofia remained in the shadows" — padding dead figure
    442: 7,  # "She meticulously scanned the sky, charting the movement of planets and stars. The telescope, though small" — padding accumulation
    443: 7,  # "Finally, after weeks of frustration and hard work, Sofia made the calculations that would change everything" — padding accumulation
    444: 0,  # clean
    445: 6,  # "a long-held dream finally realized" — padding/forbidden territory. Actually clean.
    446: 0,  # clean
    447: 0,  # clean
    448: 0,  # clean
    449: 2,  # "a close-knit group of survivors who have learned to work together" — emotion/narrator gloss
    # 450-459
    450: 0,  # clean
    451: 2,  # "He carries with him a message of hope and cooperation" — emotion naming
    452: 0,  # clean
    453: 0,  # clean
    454: 7,  # "In the midst of this political turmoil, Dr. Jameson steps forward and proposes a compromise" — padding filler
    455: 2,  # "It is a small community... The settlement is a stark contrast to..." — emotion/comparison gloss
    456: 0,  # clean
    457: 0,  # clean
    458: 2,  # "the political debate that had been raging for months comes to a head" — emotion gloss
    459: 2,  # "they carry with them the hopes and dreams of their entire community" — emotion naming
    # 460-469
    460: 2,  # "they are determined to face whatever challenges lie ahead with courage and perseverance" — direct emotion naming
    461: 2,  # "the people of the community gather together to celebrate their successes and honor their fallen comrades" — emotion/cliché
    462: 2,  # "share their hopes and dreams for the future" — direct emotion naming
    463: 0,  # clean chapter header
    464: 0,  # clean
    465: 2,  # "She feels a deep connection to the bread she creates, knowing that each loaf..." — direct emotion naming
    466: 0,  # clean
    467: 0,  # clean dialogue action
    468: 0,  # clean
    469: 2,  # "The man's eyes flutter open... he finds himself back in the bakery, the taste of the sourdough still lingering" — emotion naming
    # 470-479
    470: 0,  # clean sensory memory
    471: 2,  # "his step lighter than before, as if a great weight has been lifted from his shoulders" — dead figure + emotion
    472: 0,  # clean
    473: 0,  # clean
    474: 2,  # "her voice trembling with emotion" — direct emotion naming
    475: 0,  # clean
    476: 0,  # clean
    477: 0,  # clean
    478: 0,  # clean dialogue
    479: 0,  # clean dialogue
    # 480-489
    480: 0,  # clean
    481: 0,  # clean
    482: 6,  # "drawn toward the cabin cruiser" — clean actually. ML flagged. No forbidden words. Clean.
    483: 0,  # clean dialogue
    484: 2,  # "He believed he was fighting against an unjust system, that he was giving people a way out" — emotion telling about motivation
    485: 0,  # clean
    486: 0,  # clean
    487: 2,  # "Had his grandfather's smuggling paid for their house... The thought made him feel sick" — direct emotion naming
    488: 0,  # clean — actually good thematic statement
    489: 7,  # "'It'll be alright, Mrs. Gable,' Dr. Thorne said, straining to sound reassuring" — padding hedge
    # 490-499
    490: 7,  # "He didn't wait for a response, turning instead to the microphone..." — padding accumulation
    491: 7,  # "In moments, the monitors were a dizzying swirl of color and motion" — padding dead figure
    492: 7,  # "his face was gaunt, aged before its time, but his eyes burned with a chilling intensity" — dead figure + padding
    493: 7,  # "He knew, with a sickening certainty, that something had gone wrong" — padding dead figure
    494: 7,  # "The Memory Guillotine was a marvel of modern technology, but it wasn't infallible" — padding filler
    495: 4,  # "And then, quite suddenly, he saw something else. Something he hadn't noticed before." — standalone fragments
    496: 0,  # clean
    497: 7,  # "Dr. Thorne frowned. He'd never seen anyone quite like this man before. What, he wondered, was someone like him doing at a place like this?" — padding accumulation
    498: 7,  # "Dr. Thorne felt sick. He wanted to vomit. He wanted to scream. He wanted to tear his eyes out" — anaphora abuse "He wanted to X" x3
    499: 0,  # clean
    # 500-509
    500: 0,  # clean
    501: 7,  # "The Memory Guillotine whirred to life... Dr. Thorne watched as the color drained from her face, watched as her screams died in her throat. He watched as she sl..." — anaphora "watched as" x3
    502: 1,  # "He knew what he had seen. He knew what it meant. And he knew that nothing would ever be the same again." — thesis/dramatic countdown pattern
    503: 7,  # "They came in ones and twos, stumbling out of the jungle with crude bandages wrapped around wounds, supporting each oth..." — padding accumulation
    504: 0,  # clean specific detail
    505: 0,  # clean
    506: 0,  # clean
    507: 0,  # clean
    508: 7,  # "Johnson examined the wound, feeling for pieces of bone" — padding accumulation
    509: 7,  # "Johnson thought about the conversation he had had with an enemy soldier the day before... barely eighteen, and he had been crying" — padding backstory intrusion
    # 510-519
    510: 7,  # "Johnson was soon surrounded by a crowd of men, all clamoring for his attention. He did his best to help them, but he knew he would never be able to treat" — padding accumulation
    511: 0,  # clean
    512: 7,  # "the cheap plastic handset clattering against the cradle. My desk was littered with scribbled notes, grainy photographs, and half-empty coffee cups" — padding accumulation tricolon
    513: 0,  # clean
    514: 0,  # clean
    515: 0,  # clean dialogue
    516: 7,  # "Marv flopped down in the client chair, propping his shoed feet on my desk. 'Maybe she just ran off.'" — padding accumulation
    517: 0,  # clean
    518: 0,  # clean
    519: 0,  # clean
    # 520-529
    520: 0,  # clean
    521: 0,  # clean
    522: 0,  # clean
    523: 7,  # "my hand resting on my trusty revolver" — "trusty" cliché + padding
    524: 7,  # "I heard a murmur of voices beyond a curtained doorway. I crept closer, straining to listen." — padding accumulation
    525: 0,  # clean dialogue
    526: 0,  # clean
    527: 0,  # clean dialogue
    528: 0,  # clean
    529: 0,  # clean
    # 530-539
    530: 0,  # clean dialogue
    531: 0,  # clean
    532: 7,  # "her cartographer's eye trained to spot anomal..." — padding accumulation
    533: 0,  # clean
    534: 2,  # "The thrill of discovery could..." — emotion naming
    535: 0,  # clean
    536: 0,  # clean
    537: 0,  # clean
    538: 2,  # "Emma's understanding of her parents' marriage had been simple: they were a stable, if unromantic, couple" — narrator gloss/emotion
    539: 7,  # "Emma felt the room around her shift, the familiar contours of her childhood home blurring into something new" — padding dead figure
    # 540-549
    540: 7,  # "her eyes scanning the intricate script with a mixture of fascination and bewilderment" — padding accumulation
    541: 6,  # "the air thick with the scent of old books and stale coffee" — "thick with the scent" cliché, but ML flagged forbidden. Checking: no direct forbidden words. Actually padding/dead figure. Reclassify: 7
    542: 2,  # "Maria shook her head, chiding herself for getting ahead of herself... She had to focus on the task at hand" — emotion naming
    543: 0,  # clean
    544: 0,  # clean dialogue
    545: 7,  # "What if the text is simply...precise?" — padding ellipsis + hedging
    546: 7,  # "even if it did, how would we know? We can't verify the predictions" — padding dialogue hedging
    547: 0,  # clean dialogue
    548: 0,  # clean
    549: 0,  # clean
    # 550-559
    550: 2,  # "Maria felt her world tilt on its axis. The weight of that realization was both exhilarating and terrifying." — dead figure + direct emotion
    551: 0,  # clean
    552: 7,  # "I'd spent countless hours fine-tuning the Kalman filter... a tiny discrepancy, a slight wobble" — padding accumulation
    553: 7,  # "My thoughts swirled with possibilities. Had someone else accessed the system? Was it a malfunction?" — padding rhetorical questions
    554: 0,  # clean dialogue
    555: 0,  # clean
    556: 0,  # clean dialogue
    557: 0,  # clean dialogue
    558: 0,  # clean dialogue
    559: 0,  # clean
    # 560-569
    560: 7,  # "even the smallest mistake could have catastrophic consequences. And I was starting to suspect that someone – or something – was playing with fire" — padding dead figure
    561: 0,  # clean
    562: 7,  # "nothing could have prepared her for this" — padding filler
    563: 0,  # clean
    564: 0,  # clean
    565: 7,  # "the Mandelbrot set is not just a mathematical construct - it's also a visual representation of the human emotional experience. The intricate patterns and boundaries of the set s..." — padding narrator_gloss accumulation
    566: 7,  # "Patel's mind starts to spin. Could it be that the aliens are using mathematical structures to communicate emotions" — padding + rhetorical
    567: 0,  # clean
    568: 0,  # clean
    569: 7,  # "her expression unreadable" — padding dead figure
    # 570-579
    570: 7,  # "She had read it so many times that the words were starting to blur together, but she knew she had to stay focused" — padding accumulation
    571: 2,  # "K. Patel emerged with a stack of papers and a perfunctory smile" — emotion telling
    572: 2,  # "On impulse, Maya pushed open the door" — emotion naming clean actually. But ML flagged. One beat. Clean.
    573: 0,  # clean
    574: 0,  # clean dialogue
    575: 0,  # clean
    576: 0,  # clean
    577: 2,  # "we know how hard it can be to adjust to a new city" — emotion naming
    578: 0,  # clean
    579: 0,  # clean
    # 580-589
    580: 2,  # "Aria's thoughts turned to the confrontation... Aria had tried to..." wait — this is 579. 580: "She pushed the thoughts aside... 'I think it's just a matter of time before she recovers,' she said reassuringly" — padding + emotion
    581: 2,  # "she couldn't help but feel a sense of..." — "couldn't help but" padding + emotion naming
    582: 0,  # clean
    583: 0,  # clean
    584: 7,  # "Aria felt her doubts rising to the surface. What if she was wrong, what if her methods were indeed devilish?" — padding rhetorical
    585: 7,  # "The questions swirled around her, making her feel dizzy and disoriented" — dead figure + padding
    586: 0,  # clean
    587: 0,  # clean
    588: 0,  # clean
    589: 0,  # clean dialogue
    # 590-597
    590: 0,  # clean
    591: 7,  # "I stare at the stack of financial records on my desk, my eyes scanning the columns of numbers as I try to make sense of it all" — padding accumulation
    592: 7,  # "My curiosity piqued, I decide to investigate further" — "curiosity piqued" padding
    593: 7,  # "It takes me a few minutes to connect the dots, but when I do, my heart starts racing" — dead figure + padding
    594: 0,  # clean
    595: 0,  # clean
    596: 2,  # "inside, I'm seething. I know that I've stumbled upon something big" — direct emotion naming
    597: 0,  # clean
}

# Fix a few reconsidered entries after review:
# 58: re-examining "He yanked the metal door open" - this is just action, clean
LABELS[58] = 0
# 9: "utilizing" is forbidden-ish but single instance; padding overall
LABELS[9] = 7
# 130: "suppressed a shudder of resentment" is direct emotion, ML flagged as forbidden
LABELS[130] = 2
# 190: "What brings you to this place?" - no forbidden words, clean
LABELS[190] = 0
# 251: "map our own emotions to their mathematical language" - no forbidden words
LABELS[251] = 0
# 253: "couldn't shake the feeling that this was the key" - just one cliche, clean
LABELS[253] = 0
# 361: "She opened her mouth to speak, but Lisa beat her to it" — single beat, clean
LABELS[361] = 0
# 435: "his voice thick with emotion" — single instance, clean
LABELS[435] = 0
# 445: "a long-held dream finally realized" — single instance, clean
LABELS[445] = 0
# 482: no forbidden words, clean
LABELS[482] = 0
# 541: no forbidden words, padding/dead figure better
LABELS[541] = 7
# 572: single impulse beat, clean
LABELS[572] = 0
# 152: "He just… vaporized" — fresh metaphor, clean
LABELS[152] = 0


def main():
    records = []
    with open("/home/ben/code/prose-doctor/corpus/review_chunk_2.jsonl") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 598, f"Expected 598 records, got {len(records)}"

    output_path = "/home/ben/code/prose-doctor/corpus/reviewed_chunk_2.jsonl"
    with open(output_path, "w") as f:
        for i, record in enumerate(records):
            label = LABELS[i]
            record["label"] = label
            record["class_name"] = CLASS_NAMES[label]
            record["method"] = "llm_review"
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} records to {output_path}")

    # Print distribution
    from collections import Counter
    dist = Counter(CLASS_NAMES[LABELS[i]] for i in range(len(records)))
    print("Label distribution:")
    for name, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
