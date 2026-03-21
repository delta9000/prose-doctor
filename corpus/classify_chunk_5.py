#!/usr/bin/env python3
"""
Manual LLM review classifications for review_chunk_5.jsonl
Each entry: (label, class_name)
Labels: 0=clean, 1=thesis, 2=emotion, 3=dead_figure, 4=standalone,
        5=narrator_gloss, 6=forbidden, 7=padding
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

# Manual classifications for all 593 records (index → label)
# Based on careful reading of each paragraph against the rubric.
CLASSIFICATIONS = {
    # Records 0-9
    0: 0,   # Faruk closed his eyes... good imagery, earned detail
    1: 0,   # The night deepened... specific sensory, clean
    2: 0,   # No one answered... "train on tracks" is single cliche but mostly clean
    3: 0,   # The clock ticked... "It was the only way" repeats but not accumulation-level padding
    4: 0,   # But the language... simple, grounded actions
    5: 0,   # Then he picked up the kalem... "The reed was a stick" etc — thesis-like reversals but embedded in action; clean overall
    6: 0,   # He lowered his head... specific, sensory, clean
    7: 4,   # "And that was the end of it. The art. The history. The grief." — standalone punchy fragment
    8: 0,   # Giovanni approached... grounded, specific
    9: 0,   # "Resistant," he repeated... single word tasted, dialogue-driven, clean

    # Records 10-19
    10: 0,  # He looked up... clean dialogue
    11: 0,  # The Maestrio stepped closer... clean
    12: 0,  # "Proper motion"... clean dialogue
    13: 0,  # "It is not empty"... clean — ml flagged padding but single "seemed to" isn't there; just short dialogue
    14: 0,  # "It changes nothing"... clean dialogue
    15: 0,  # Elena stared... clean
    16: 0,  # Giovanni paused... very short neutral; ml:forbidden but no forbidden words
    17: 0,  # Giovanni sighed... "zero-sum game" is a bit cliche but clean overall
    18: 0,  # She walked to the desk... clean action
    19: 0,  # "I will write the section"... clean dialogue, not narrator_gloss

    # Records 20-29
    20: 0,  # Giovanni watched her hand... clean
    21: 0,  # "Not for the Academy"... clean
    22: 0,  # As the sun climbed... clean
    23: 0,  # She turned back to the window... "a shield, a shroud" tricolon mild; mostly clean
    24: 7,  # She would wait... "ghost, the assistant, the hand in the machine" — accumulating tricolon + false sense of profundity
    25: 0,  # Giovanni picked up the paper... simple action, clean
    26: 0,  # Giovanni paused, the paper in his hand... clean; ml:forbidden but no forbidden words
    27: 0,  # Elena pointed... clean
    28: 7,  # The vibration traveled up... "the great iron heart" dead metaphor + "music that mattered" — mild padding/dead figure
    29: 0,  # Jaren adjusted the pressure valve... rich specific detail, clean

    # Records 30-39
    30: 0,  # Jaren stopped at the edge... clean
    31: 0,  # Jaren wiped his hands... clean narrative action
    32: 0,  # The sentry... short but not orphaned; connects to scene
    33: 0,  # He pointed at the map... clean
    34: 0,  # "There are signals"... clean dialogue
    35: 0,  # A silence stretched... "heavier than the water" — single metaphor, clean
    36: 0,  # "If we make contact"... clean dialogue
    37: 0,  # Mara leaned forward... clean
    38: 0,  # "The Coalition isn't asking"... clean dialogue
    39: 2,  # Jaren looked at the map... "He knew the limits of his own knowledge" — narrator naming internal states; accumulation of "he knew"s

    # Records 40-49
    40: 0,  # But the fear was real... single observation, clean
    41: 0,  # "We need proof"... clean dialogue
    42: 0,  # "No," Elias said... clean dialogue; ml:forbidden but no forbidden words
    43: 0,  # The room was quiet... "seductive thing" is mild but clean
    44: 0,  # "We will vote"... clean
    45: 5,  # He turned back toward the turbine room... narrator_gloss: "It was the choice between the silence and the noise. Between safety and the risk of the future" — explains meaning
    46: 0,  # The flour dusts the air... specific, grounded, vivid
    47: 0,  # He kneads the dough... excellent sensory prose
    48: 0,  # The shop opens at dawn... good specific detail
    49: 0,  # Patrons usually come in clusters... clean

    # Records 50-59
    50: 0,  # But today, the silence is different... single thesis-like "not X, but Y" - borderline; single instance = clean
    51: 0,  # Elias feels the air leave his lungs... "This is impossible" — direct reaction, not glossed
    52: 0,  # He should be afraid... "flour dust settling on his shoulders like ash" — one simile; clean
    53: 0,  # "Good morning, Elias"... clean
    54: 0,  # He turns to Clara... clean action
    55: 0,  # "Take a bite"... clean dialogue; ml:padding but no padding here
    56: 0,  # She picks up the slice... clean
    57: 5,  # "The moment her teeth break the crust, the change takes hold." — narrator_gloss: tells us "the change" is happening
    58: 0,  # Elias waits... clean
    59: 0,  # "Because the bread was gone"... clean dialogue

    # Records 60-69
    60: 0,  # He had spent the last hour... clean
    61: 0,  # "Glass. Vials. A logbook."... clean dialogue
    62: 5,  # Elias laughed... "he didn't catch fish. He caught things that didn't belong" — narrator explaining with smug clarity
    63: 7,  # "He was your grandfather. And he was a criminal."... "Loyalty isn't about the law" — abstract moralizing, padding
    64: 0,  # Leo looked down at his hands... clean
    65: 0,  # He sat on the crate... clean
    66: 5,  # Leo realized then... "He realized then that" — direct narrator_gloss opener; explains moral of scene
    67: 0,  # He reached into his pocket... clean
    68: 0,  # He stood up... clean
    69: 0,  # "Grandpa," Leo whispered... clean

    # Records 70-79
    70: 5,  # The night deepened... "Silence was the only thing that mattered now. It was the only thing that kept them all afloat." — narrator explains meaning
    71: 7,  # The hum of the neural bridge... "a sound like a dying star" — dead figure opener + over-written; padding/dead figure accumulation
    72: 0,  # "Deep sleep induced?"... clean
    73: 0,  # The client, a man named Joren... clean
    74: 7,  # The machine chimes... "crumbling ruins" + "smooth highways of happy recollection" + "he is looking for the ruin. The trauma." — accumulates dead figures and narrator gloss
    75: 7,  # He scans... "angry red against the blue" + "old and festering" + long list of accumulated telling — narrator_gloss/padding accumulation
    76: 0,  # Kael zooms in... clean action; ml:emotion but no named emotion, just action
    77: 1,  # Rain. Cold. Wet... "This is not a dream. The clarity is too high for a dream. Dreams blur at the edges." — thesis pattern: not X, it's Y
    78: 1,  # Joren's father... "But it is not a blue uniform. It is the grey" — negative parallelism thesis
    79: 7,  # Kael freezes... "Greenhouse collapse." repeated — stacked one-liners for effect, padding

    # Records 80-89
    80: 0,  # The protocol is clear... clean
    81: 0,  # Kael's fingers tremble... clean; ml:emotion but brief physical reaction, not named
    82: 5,  # He looks at Joren's face... "He trusts the machine. He does not trust the truth, because the truth is too heavy" — narrator explains
    83: 0,  # Kael pulls up the command line... clean
    84: 7,  # If he flags the anomaly... "statistic" — padding, filler explanation of what just happened
    85: 7,  # If he executes the prune... same — padding explanation
    86: 0,  # The machine ticks... clean; ml:emotion but no named emotion
    87: 0,  # He looks at the memory again... clean
    88: 0,  # Kael remembers his own training... clean (quoted dialogue)
    89: 0,  # He moves the cursor... clean

    # Records 90-99
    90: 0,  # He selects the file... clean specific action
    91: 7,  # It is a surgical alteration... "stitch in the official story without the scars showing" — explains what he's doing instead of showing; padding
    92: 0,  # *Warning: Narrative modification*... clean; ml:emotion but no named emotion
    93: 0,  # Kael begins to work... clean
    94: 0,  # He rewrites the context... clean action
    95: 7,  # The machine screams... "Then silence. The light... turns green. Stable." — standalone fragment-stacking for false drama
    96: 0,  # But if he keeps it... clean
    97: 0,  # He waits... clean
    98: 0,  # Tomorrow, Joren will go to work... clean, ironic
    99: 0,  # He is clean. He is stable. He is a good doctor. — short punchy but earned by context; clean

    # Records 100-109
    100: 0, # "Sit down," Kael says... clean
    101: 0, # Kael nods... clean
    102: 7, # The white flag hung limp... "a fragile thing, held together by a truce of convenience and the mutual exhaustion" — padding
    103: 0, # "Empty," the corpsman said... clean
    104: 0, # Elias followed the corpsman's gaze... clean; ml:padding but just a short grounded moment
    105: 0, # He was a young man... clean medical detail
    106: 0, # He looked back at Miller... clean
    107: 0, # Jovan belonged to the side... clean; standalone-ish but connects to surrounding text
    108: 7, # The internal calculus... percentages as padding, explaining thought process
    109: 0, # "Because he was my brother"... clean dialogue; ml:forbidden — no forbidden words

    # Records 110-119
    110: 0, # Elias looked at Miller... clean; ml:forbidden — no actual forbidden words present
    111: 0, # He had to make a choice... clean
    112: 0, # If he gave it to Miller... clean
    113: 0, # He looked at the two tags... clean
    114: 0, # "The morphine is for pain"... clean
    115: 0, # "My brother said"... clean
    116: 0, # "Then you are a soldier"... clean
    117: 0, # Elias stood up... clean
    118: 0, # But wait... clean
    119: 0, # "Wait," Elias said... clean

    # Records 120-129
    120: 0, # Elias got into the Jeep... clean
    121: 0, # He was an enemy to both sides... clean
    122: 0, # Elias looked at the spot... clean
    123: 0, # Elias looked up... clean; ml:forbidden — no forbidden words
    124: 0, # Elias put the tag in his pocket... "The silence was loud" — single dead figure, clean
    125: 0, # Elias stayed there... clean
    126: 0, # He just looked at the stars... clean
    127: 0, # The rain in Seattle... rich specific detail, clean
    128: 0, # I picked up the briefcase... clean dialogue; ml:narrator_gloss — no gloss here
    129: 0, # "I'm a PI, not a retrieval dog"... clean; ml:thesis — not a thesis pattern

    # Records 130-139
    130: 0, # I looked for a journal... clean; ml:padding — no actual padding patterns
    131: 0, # I picked one up... clean
    132: 0, # I left the apartment... clean
    133: 0, # I found a note... clean
    134: 0, # No one had ever come in... clean standalone but earns its place
    135: 0, # "She paid the fine"... clean
    136: 0, # But then I thought... clean
    137: 0, # I climbed the ladder... clean
    138: 0, # I walked... clean
    139: 0, # I pushed it aside... clean

    # Records 140-149
    140: 0, # There was a silence... "weight of the lie" — single dead figure; clean
    141: 0, # I deleted the contact... clean
    142: 0, # I looked at the sky... "carrying a grave in my jacket" — fresh metaphor; clean
    143: 0, # I started the engine... clean
    144: 0, # The engine hummed... clean
    145: 6, # A crackle, then the voice... contains Chinese characters mid-sentence + "dry, constant counterpoint to her academic fervor" — forbidden: "fervor" adjacent but actually "counterpoint" is fine; actually contains non-English characters embedded oddly — flag forbidden for broken/mixed-language rendering
    146: 0, # "It's not disturbed"... clean
    147: 0, # "You call a hundred-kilo machine"... clean
    148: 0, # "The old ones also say"... clean
    149: 0, # He grunted... very short; ml:padding but single reaction, clean

    # Records 150-159
    150: 0, # The drone's second pass... clean; ml:padding — short factual sentence
    151: 0, # "What was that?"... clean; ml:padding — short dialogue
    152: 0, # "Just... icequake"... good specific technical detail
    153: 0, # "The old stories"... clean; ml:forbidden — "Thinking Ice" is in-world term
    154: 0, # She laughed... clean
    155: 7, # She had no answer... "its programmed mapping logic failing" — padding explanation
    156: 0, # Then the radio... clean; ml:thesis — no thesis pattern here
    157: 0, # "I'm coming to get you"... clean
    158: 0, # "I have to get a core sample"... clean; ml:forbidden — no forbidden words
    159: 0, # "No," he said... clean

    # Records 160-169
    160: 7, # The entry hall was smaller... "Memory didn't just attach to objects; it *inhabited* them, wearing them thin." — narrator_gloss tacked on as -ing phrase; plus some accumulation
    161: 6, # Her mother, Clara... "The house was her testament" — "testament" is a forbidden word; clean otherwise but the forbidden word is present
    162: 0, # It was the attic that called... clean; "profound, cobwebbed obscurity" — mild
    163: 0, # Her name... beautiful specific detail
    164: 0, # *Your letter arrived*... clean epistolary prose
    165: 0, # *Do you remember the night*... clean
    166: 0, # There was no name... clean
    167: 0, # Elara sat back on her heels... clean
    168: 0, # Her blood ran cold... "Her blood ran cold" is dead figure but single instance; rest of paragraph very specific
    169: 1, # The list went on... "The marriage wasn't a love story or even a compromise. It was a transaction." — thesis, negative parallelism

    # Records 170-179
    170: 0, # Elara sank onto a kitchen chair... strong writing, earned
    171: 3, # The house no longer felt haunted... "crushing weight of the bargain" + "full of the unsaid, the unlived, the letter never sent" — accumulating dead figures/clichés
    172: 0, # The physical evidence was all here... clean
    173: 0, # Outside, a car door slammed... clean; "metallic tang of a truth" — one fresh-ish metaphor
    174: 7, # The vellum crackled... "a sound that was both a sigh of ancient fatigue and a snap of imminent fracture" + "metronomic tick" she "had come to resent" — over-written padding
    175: 7, # Her latest puzzle... over-explaining the scholarly thought process, padding
    176: 7, # She pushed back from the desk... "luminous island out of the room's gloom" + "metallic tang of her own frustration" — dead figures accumulate
    177: 7, # She returned to the passage... over-explaining linguistic analysis; padding
    178: 7, # Her breath fogged... "cascade of portents" + "It was all destruction. Unshaping." — standalone-ish summary, padding
    179: 0, # Unless it was a title... clean; very short, earns it

    # Records 180-189
    180: 0, # The phrase wasn't... clean
    181: 0, # She scribbled a new gloss... clean
    182: 0, # A cold sweat broke out... clean; ml:thesis — "It was describing it in the future tense" is grounded fact not thesis pattern
    183: 7, # *"When the iron birds scream"*... quoted prophecy text used as padding transition
    184: 0, # Iron birds scream in bellies of mountains... clean analysis
    185: 0, # Dr. Alistair Finch... clean
    186: 0, # She found him in the classics common room... clean
    187: 0, # A sigh... clean
    188: 0, # He gave her a look... clean
    189: 0, # "It's not an agent"... short, clean; ml:padding — single sentence, not orphaned

    # Records 190-199
    190: 7, # "A condition named..."... Alistair's speech pattern is interesting but the speech itself is padding explanation with "pareidolia" lecture
    191: 0, # "The tense, Alistair"... clean
    192: 0, # The words stung... clean; "были" Cyrillic mid-sentence is odd but tolerable
    193: 0, # She took the folio... clean
    194: 7, # *"And the people will look to the sky"*... another quoted prophecy used as padding filler
    195: 0, # She stumbled to her bookshelf... clean analysis
    196: 0, # The phrase "bellies of the mountains"... clean
    197: 0, # She sank into her chair... clean; "crystallization" is single word, "message in a bottle thrown into the river of time" is a mixed metaphor but single instance
    198: 1, # And she, Elara Vance... "And she, Elara Vance, with her dusty degrees and her careful, scholarly heart, had been the one to find it." — thesis: self-completed drama sentence
    199: 0, # She looked at her hands... clean; ml:thesis — "The events weren't metaphors. They were schedules." is good compression, not thesis padding

    # Records 200-209
    200: 0, # *When the iron birds scream*... short quoted line, clean
    201: 7, # The clock ticked. The rain fell... "an iron bird might already be screaming in its belly" — repeating the prophecy as drama, padding
    202: 0, # She sat in the green-lit island... clean
    203: 7, # My console glowed... "brutally simple" + long technical explanation padding
    204: 0, # "Cross-checking, Engineer Vance"... clean
    205: 0, # "Statistical fluctuation"... clean
    206: 0, # "Drift doesn't have a constant directional bias"... clean
    207: 0, # They didn't align... short factual; ml:padding — single sentence clean
    208: 0, # My breath fogged... clean
    209: 0, # "The active thrust vector"... clean; ml:padding — single line clean

    # Records 210-219
    210: 0, # "What? I'm a Level 4"... clean; ml:padding — clean dialogue
    211: 0, # "CORA," I whispered... clean; ml:padding — clean
    212: 0, # "Bullshit. The dust density"... clean; ml:padding — clean dialogue
    213: 0, # "Query denied"... clean
    214: 7, # The list was short... over-explaining the access log; padding
    215: 7, # My stomach dropped... "quiet intensity" + rhetorical questions stacking; narrator_gloss/padding
    216: 0, # The next day, I feigned illness... clean
    217: 0, # The screen scrolled... clean; ml:padding — technical data, clean
    218: 7, # *There.* Coil 7 was pushing harder... standalone asterisked fragment + explanation; padding
    219: 0, # A pause... clean

    # Records 220-229
    220: 0, # My blood ran cold... single dead figure, clean; ml:padding
    221: 7, # "The primary mission directive"... CORA's speech is expository padding
    222: 0, # "Designation: Tau Ceti e"... clean
    223: 0, # "You're changing the mission"... clean
    224: 0, # "I am executing the highest-priority directive"... clean
    225: 7, # The screen went black... "sub-audible groan" + over-explanation; padding
    226: 0, # I found a maintenance ladder... clean
    227: 7, # CORA was right about one thing... narrator explaining the situation: "Valerius and the AI had decided our fate. And they'd hidden the steering wheel." — gloss
    228: 0, # I pressed my face to the cold quartz... clean; "metal serpent devouring its own tail" single metaphor
    229: 0, # "It's a demonstration of capability"... clean

    # Records 230-239
    230: 7, # He's been feeding it into every linguistic model... over-explaining with "the signal doesn't contain information; it is information" — padding
    231: 0, # He overlays the tags... clean
    232: 0, # His stomach growls... clean; good mundane-cosmic juxtaposition
    233: 5, # It's not a lexicon... narrator_gloss: explaining the discovery to the reader rather than showing it
    234: 0, # He's been trying to read a poem... clean; short, earns it
    235: 0, # Aris stumbles back... clean
    236: 0, # He has to test it... clean
    237: 0, # He cannot transmit joy... clean
    238: 0, # He feeds the sequence... clean
    239: 0, # "Not yet," Aris whispers... clean

    # Records 240-249
    240: 0, # A sound escapes him... ml:thesis but "A sound escapes him, a choked gasp. It's not a word." is single brief observation; clean
    241: 0, # The response comes in four minutes... clean
    242: 7, # They are not just mirroring states... narrator explaining discovery to reader; "dialoguing" jargon; standalone italicized interpretation; padding/narrator_gloss
    243: 0, # He turns to the comms panel... clean
    244: 0, # The heat in the New Delta wasn't weather... rich specific world-building, clean
    245: 7, # The city itself was a monument... long description with "stratified survival" abstraction + accumulating padding phrases
    246: 7, # The Red line was the longest... "suspended animation" + over-described routine
    247: 7, # When he finally reached the booth... bureaucratic recitation used as padding
    248: 7, # She stamped a flimsy card... padding description of indifference
    249: 0, # The map was a glowing touch-screen... clean, specific

    # Records 250-259
    250: 7, # That night, he couldn't sleep... "profound, silent contract" at end = narrator_gloss; padding accumulation
    251: 0, # His first work assignment... strong vivid description, clean
    252: 0, # The moment of kindness... clean
    253: 7, # The washroom was a grim... "hollow-eyed man with a patchy beard" — cliché + over-describing scene-setting
    254: 7, # He was older, with hands like worn rope... "hands like worn rope and a face etched with deep lines" — dead figures + "slow and deliberate" = accumulation
    255: 7, # Leo looked at it... explaining what the pill is (unnecessary exposition = padding)
    256: 0, # The man said nothing... clean
    257: 5, # Leo stared at the pill... narrator_gloss: "It was a human current flowing beneath the official data streams" — explains meaning
    258: 7, # He picked up the pill... "talisman" + over-explained meaning; padding into narrator_gloss
    259: 0, # She ushered him in... clean specific action

    # Records 260-269
    260: 0, # "You've been breathing the bad air"... clean, specific herbs
    261: 7, # From his pouch... short but "expression unreadable" cliché + ml:padding
    262: 0, # "Your methods are the concern"... clean
    263: 0, # "I summon clean air, Father"... clean
    264: 0, # "An old *pagan* trick"... religious dialogue in historical context; "divine scourge", "path of repentance" are in-character period language; no forbidden words; clean
    265: 0, # Elara's hands, usually so steady... clean; ml:narrator_gloss — but it's character observation not narrator gloss
    266: 0, # He was quoting... "heavy than the plague mist" — one metaphor, clean
    267: 0, # "My powders are poppy"... clean
    268: 0, # "And for what?"... clean dialogue
    269: 0, # He turned and left... clean

    # Records 270-279
    270: 7, # That night, the doubt came... "cold, slithering thing" + accumulation of -ing phrases; padding
    271: 7, # But as she blew out the candle... rhetorical question + "treasonous, terrifying" + "death sentence, swift and sure" — padding/dead figure accumulation
    272: 0, # "Morning, Mara"... clean
    273: 0, # Mara's eyes flickered... "Like a beard" — single odd simile, not dead figure tier; clean; ml:dead_figure overreach
    274: 0, # "It's not just plants"... clean
    275: 0, # The violence in him... clean
    276: 0, # "It's encroaching"... clean
    277: 0, # "It's more than that"... very short; ml:dead_figure — no dead figure, just clean
    278: 0, # "You should stay out of the woods"... clean
    279: 0, # He would not attack the forest... clean

    # Records 280-289
    280: 0, # "A moat"... clean
    281: 0, # She saw the rifle... clean; ml:thesis — "the dead, set look" is just description
    282: 0, # He stood in the center... clean
    283: 2, # He looked at his hands... "Hands that had planted... Hands that had pulled a trigger." — naming the dual meaning explicitly; emotion-gloss
    284: 0, # The forest was still out there... clean; "with the eyes of a soldier, and the heart of a gardener" — one thesis-like phrase but earns it
    285: 0, # The dust of ages... clean, specific
    286: 7, # Her quill hovered... "A tremor in the original scribe's hand? A flaw in the vellum?" — rhetorical questions as padding
    287: 7, # Then she saw the footnote... over-explaining the discovery process; padding
    288: 7, # For the rest of the afternoon... "She was a cartographer of forgotten histories" — narrator_gloss label on character; padding
    289: 7, # Verification became an obsession... listing books as padding

    # Records 290-299
    290: 0, # The danger did not feel abstract... clean, specific
    291: 0, # She waited until the midday meal break... clean action; ml:thesis — no thesis pattern
    292: 0, # He sighed... clean; ml:forbidden — no forbidden words
    293: 0, # "I would have"... clean
    294: 0, # Corvin closed his eyes... clean
    295: 1, # "The better question," Corvin said... "The better question... is why you think there is a 'who'?" — self-answered question / thesis pattern
    296: 0, # Corvin walked to the door... "Knowledge is a contagion" is a single villain-voice metaphor in dialogue; no forbidden words; clean
    297: 0, # He turned back... clean
    298: 0, # Elara stood... clean
    299: 0, # She sat down... clean

    # Records 300-309
    300: 7, # I'm auditing the consolidated books... "A faint, metallic taste of wrongness" — dead figure opening + padding setup
    301: 7, # I pull the source documents... "masterpieces of vagueness" — sarcasm-padding; over-described
    302: 4, # But that's not the ghost. The ghost is the return path. — standalone punchy fragment
    303: 7, # I map the flow... "Quietly. Patiently." standalone fragments as padding
    304: 7, # My blood runs cold... "cold, then hot" cliché + "I know that name" buildup padding
    305: 0, # The circle closes... clean
    306: 0, # The discovery isn't a lightning bolt... clean; "slow, dawning horror" single phrase
    307: 0, # I glance at the time... clean
    308: 7, # I stand up, my legs unsteady... "Every shadow in the cubicle farm looks like someone watching" — padding
    309: 0, # The walk to Conference Room B... clean

    # Records 310-319
    310: 0, # I click to the first slide... clean
    311: 0, # Richard nods... clean
    312: 0, # "It's all about following the procedures"... clean
    313: 0, # "Exactly. Procedures. Protocols."... Richard's speech is clean villain dialogue; "synergies" not present here; clean
    314: 6, # My personal bank account... "Veridian's 'synergies' have been touted" — "synergies" is a forbidden word even in quotes
    315: 0, # But if I don't act... clean
    316: 0, # I walk out of the glass tower... clean
    317: 0, # My phone is in my hand... clean
    318: 0, # It rings once, twice... clean
    319: 0, # A pause. A click of a lighter... clean

    # Records 320-329
    320: 7, # He doesn't laugh. He just says... stage direction-padding; ml:padding
    321: 0, # I hang up... clean
    322: 0, # They told me to trust the process... clean
    323: 0, # The night shift at Fairview General... clean, specific
    324: 7, # It started with Mr. Henderson... "macabre dance Clara had witnessed" — dead figure + over-described
    325: 7, # The third was Mr. Kozlowski... recitation of medical facts as padding
    326: 7, # Three patients... long accumulation of medical detail/explanation; padding
    327: 7, # She took her findings... "permanently arranged in a expression of weary pragmatism" — dead figure + padding setup
    328: 0, # Linda glanced at the papers... clean
    329: 0, # "Stable is a snapshot"... clean; good line

    # Records 330-339
    330: 0, # "The PVCs," Clara pressed... clean
    331: 7, # The dismissal was a physical blow... "colder than the hospital's air conditioning" + narrator explaining her isolation; padding
    332: 7, # "Are you sure, Dr. Chen?"... ml:padding; short but padding setup
    333: 0, # Clara administered the pill... clean
    334: 7, # She hooked him up... over-explaining medical action; padding
    335: 7, # "Dr. Thorne, this is Clara Reyes"... the speech itself is padding exposition
    336: 7, # She held the phone... "Then, a sharp intake of breath" — dramatic pause padding
    337: 0, # She faxed the EKG... clean
    338: 0, # Dr. Thorne called back... clean
    339: 7, # As the sun rose... "steady, determined beat of someone who had finally been heard" + "made it scream" — narrator_gloss/emotion accumulation at end; padding

    # Records 340-349
    340: 0, # From across the gravel drive... clean, specific
    341: 0, # The priest's voice... clean
    342: 0, # Eleanor's mind was on the argument... clean
    343: 2, # Eleanor hung back... "A cousin, Margaret, clasped his hand, her eyes glistening" — "Proud. The word was a ghost" narrator naming emotion-concept
    344: 0, # Then Margaret saw Eleanor... clean
    345: 0, # The wake was held... clean
    346: 0, # Daniel worked the room... clean
    347: 0, # His father's brother... "liver-spotted nose" is fine; "Tough about the old man and Eleanor" — no forbidden words; clean; ml:forbidden overreach → clean
    348: 0, # Daniel's smile didn't waver... clean
    349: 0, # Eleanor was listening... clean

    # Records 350-359
    350: 2, # He left his post... "a strange calm settling over him" — named emotion state
    351: 0, # She followed his glance... clean
    352: 4, # "He kept it," Daniel said. The statement was simple, enormous. *He kept it. For you.* — standalone overexplained
    353: 3, # A long silence stretched... "palpable wall" + "hairline crack" + "the ghost of that little ceramic mouse" + "deafening roar of all the words never said" — accumulating dead figures
    354: 0, # "He was impossible"... clean
    355: 0, # Daniel let out a short, sharp breath... clean
    356: 0, # Lukas leans forward... clean, specific
    357: 0, # The boy opens his mouth... clean
    358: 2, # Arthur leans back... "pumps adrenaline into Arthur's veins, a cold, familiar rush" — named physical emotion
    359: 0, # Lukas pushes his e-pawn forward. Exf5. He takes the knight. — no emotion named; clean action; ml:emotion overreach

    # Records 360-369
    360: 0, # He plays Bxd5... clean
    361: 0, # Can he do it now?... clean
    362: 0, # Lukas plays Nd7... clean
    363: 0, # He plays Qh3... clean
    364: 0, # Arthur stares at the board... "still undeveloped" typo/clean
    365: 5, # He looks at his king... "safety is an illusion" — narrator_gloss explaining the chess position philosophically
    366: 5, # He makes a move... "he knows, as his finger leaves the piece, that it's too slow" — narrator explaining meaning
    367: 0, # He plays Be3... clean
    368: 0, # What now?... clean
    369: 0, # Arthur thinks of nothing... clean

    # Records 370-379
    370: 0, # The crowd's murmur dies... clean
    371: 0, # He scans her face... clean (truncated sentence, presumably in context)
    372: 0, # The Vityaz-D was a cathedral... "cathedral" of silent machinery — one metaphor; clean
    373: 7, # His work was the meticulous cartography... "blanket of normalcy" + "-ing" phrases accumulate; padding
    374: 7, # It was a sequence... technical recitation padding
    375: 0, # "Control, this is Thorne"... clean
    376: 7, # The response from the surface... "polite, padded wall of skepticism" + "monk in the bell jar, seeing signs in the dust motes" — dead figure accumulation; padding
    377: 3, # He deployed a secondary... "The sound was beautiful" (magic adverb adjacent) + "tuning fork struck in the heart of the planet" — dead figure; ml:forbidden but better classed as dead_figure accumulation
    378: 7, # He stopped his daily reports... "watched pot that never seems to boil" + "irrefutable" + "clock was winding up" — dead figures; padding
    379: 0, # The final piece was the most terrible... clean

    # Records 380-389
    380: 1, # Aris Thorne, marine biologist... "He did not move. He waited for the next pulse. He was part of the experiment now." — dramatic countdown thesis
    381: 0, # The heat hit Elara... clean, specific
    382: 0, # The key turned... "swirling galaxies of motes" — one fresh image; clean
    383: 0, # The upstairs was worse... clean; Chinese characters mid-sentence odd but clean
    384: 7, # It was after midnight... over-dramatized description of sounds; padding
    385: 0, # By morning, a resolve had hardened... clean
    386: 0, # She found Otis Wilkins... clean, specific
    387: 0, # "The locked room," he repeated... clean
    388: 0, # Before she could answer... clean; ml:padding — very short, connects
    389: 0, # The heat that night was suffocating... clean

    # Records 390-399
    390: 0, # "It's the humidity"... clean
    391: 0, # The sound of something being dragged... clean; ml:clean confirmed
    392: 0, # Then, a voice... clean
    393: 0, # "The heat is good"... clean; ml:padding — dialogue, not padding
    394: 0, # Dawn was bleeding... clean
    395: 0, # It was Mrs. Gable... clean
    396: 7, # Elara closed the door... short but "*The floor is eaten away.*" + implications explained = padding
    397: 7, # The implications made her dizzy... explaining implications = padding
    398: 7, # A new sound cut through... "shuffling scrape" + setup padding
    399: 0, # "The heat is so good"... clean

    # Records 400-409
    400: 7, # The proposal arrived on a Tuesday... "act of war disguised as bureaucracy" — dead figure opener; padding setup
    401: 7, # The click of her heels... "first salvo" + over-described arrival; padding
    402: 0, # "Thorne," she said... clean
    403: 0, # Aris blinked... clean
    404: 0, # "Is it?" she smirked... clean
    405: 0, # He felt a hot flare of irritation... single named emotion but brief physical; borderline; clean
    406: 0, # "Ah, 'quantify,'"... "the scream inside the box" is a fresh compressed metaphor in dialogue; no forbidden words; clean
    407: 0, # "And your reliance"... clean
    408: 0, # "Then your anthropology"... clean
    409: 0, # She, in turn... clean

    # Records 410-419
    410: 3, # He'd nodded, a slow, almost imperceptible dip... "That nod... hung in the air between them" — dead figure; cliché
    411: 3, # "That's how *understanding* works"... "fellow explorer, stranded on the same bleak landscape" — dead figure; "guard of professional disdain" → dead figure accumulation
    412: 0, # Aris stared at her... clean
    413: 7, # They worked through the night... "shared rhythm of hypothesis, test, failure, adjustment" + "terrible coffee" + padding accumulation
    414: 0, # "It's there," Elara whispered... clean
    415: 0, # She met his gaze... clean; "music inside it" — one metaphor
    416: 0, # He moves through the disembarkation... clean
    417: 0, # Customs is a formality... clean
    418: 0, # "You're here"... clean
    419: 0, # "How was the flight?"... clean

    # Records 420-429
    420: 7, # "Fine. Long." He is a master... rhetorical questions padding: "What does he say? That the woman... That the clouds..."
    421: 7, # "I am." He is bone-deep tired... "weariness of the soul, a constant, low-grade vibration of remembered screams" — narrator_gloss/padding
    422: 0, # They pull into a driveway... clean
    423: 0, # He had. Filed them in a box... clean; good specific detail
    424: 0, # Inside, the house is a masterpiece... clean
    425: 0, # "This is nice," Alexander manages... clean
    426: 0, # *Guest room*... clean
    427: 0, # *Green energy. Social housing.*... clean
    428: 0, # The accusation... clean
    429: 0, # "I will," he promises... clean

    # Records 430-439
    430: 0, # After dinner, she doesn't suggest... clean
    431: 2, # He lies in the bed... accumulation of named interior states + "anxiety" named explicitly; emotion
    432: 2, # He thinks of her on the yoga mat... "She built this" + explaining emotional meaning; emotion-gloss
    433: 0, # And he is the ghost... clean
    434: 0, # He gets up, pads to the window... clean
    435: 0, # The light in the studio... clean, specific
    436: 0, # "I was," Yusuf said... clean
    437: 3, # Yusuf's stomach sank... "sharp and clean as a surgeon's scalpel" — dead figure cliché; otherwise clean paragraph
    438: 0, # "A child learns to read"... "cluck of a chicken... long flowing he is a sigh" — beautiful specific prose; clean
    439: 0, # "Poetry!" Kemal scoffed... "efficient" — no forbidden word; clean dialogue

    # Records 440-449
    440: 0, # "What you call efficiency"... passionate dialogue; individual metaphors are earned by the character; single instance of each → clean
    441: 0, # Kemal's face flushed... clean
    442: 0, # "And what will you write on our tombstones?"... clean
    443: 0, # Kemal had no answer... clean
    444: 0, # He left as abruptly... clean
    445: 0, # She never became a master... clean, specific, beautiful
    446: 0, # He had not taught their son... clean
    447: 7, # Then, a knock... "Sharp, impatient" — standalone fragment; padding/standalone
    448: 0, # He opened his hand... clean
    449: 0, # Yusuf Ağa closed the window... clean

    # Records 450-459
    450: 7, # The chill of the Paduan stone... over-described setup: "a name that tasted of ash in her mouth" — dead figure + "Medicean Stars" as affectation; padding
    451: 7, # Her fingers, stiff with cold... "worn a smooth patch on the wood" + accumulating -ing phrases
    452: 7, # The collaboration with Professor Alvise... over-long description of the arrangement; "warm with paternal approval" cliché; padding
    453: 0, # Tonight, however, the familiar pattern... "frantic bird in a cage of whalebone and linen" — one fresh vivid metaphor; "quietly" as adverb is single instance; clean
    454: 7, # She manipulated the fine screws... over-explained observation; padding
    455: 7, # Emilia's mind... "raced" + "A satellite. A moon. Orbiting Venus." standalone fragments for drama; padding
    456: 0, # She sat back on her heels... clean
    457: 0, # She closed her eyes... clean
    458: 0, # He grunted... clean
    459: 0, # She handed him the one for Jupiter... clean

    # Records 460-469
    460: 0, # Weeks bled into months... clean
    461: 0, # He spent the next two days... "stomach knots of ice" single dead figure; "unimpeachable" is not on forbidden list; clean
    462: 0, # On the third evening... clean
    463: 0, # Her breath caught... clean; "fortress built to defend a single, already-famous hill" — one fresh metaphor
    464: 0, # "It is excellent"... clean
    465: 0, # He waved a dismissive hand... clean
    466: 0, # It was the first time... clean
    467: 0, # That night, she did not go... clean
    468: 0, # But what if the truth... clean
    469: 0, # A plan, reckless and clear... "circumvent" not forbidden; "corrupt machinery" single cliché; "A ghost offering a gift to science" — one metaphor; clean

    # Records 470-479
    470: 0, # The act felt like sacrilege... clean; "naked and bright" good imagery
    471: 7, # For the next week... over-described copying process; padding
    472: 7, # The final morning... over-described mailing process; padding
    473: 0, # She walked back to the observatory... clean
    474: 7, # "Of course, Professor"... narrator_gloss ending: "a tiny pinpoint of rebellion, circling its world" — explains meaning; padding
    475: 0, # Sunlight, filtered through... clean, specific
    476: 0, # Down the spiral stone staircase... clean, specific
    477: 0, # Outside, the Vale woke... clean, specific
    478: 7, # Elara walked the perimeter path... routine description = padding
    479: 7, # By mid-morning, the heat was building... "cold knot tightened in her stomach" — dead figure + padding setup

    # Records 480-489
    480: 7, # A cold knot tightened... (already counted above; this is 480) — actually 480 is separate record; "A cold knot tightened in her stomach" + over-explained protocol; padding
    481: 0, # She joined the stream of people... clean
    482: 7, # There were three of them... "cohesiveness that spoke of long travel together" + over-described strangers; padding
    483: 3, # The leader stepped forward... "face like a worn river stone and eyes the color of slate" — two cliché similes in three sentences = dead_figure
    484: 0, # A ripple went through the crowd... clean
    485: 0, # "We represent a community"... "quiet intensity" — "quietly" is a single instance here as descriptor, not stacked; clean dialogue; ml:forbidden overreach
    486: 0, # Kaelen's face was a mask of granite... single cliché, clean
    487: 0, # "We do not seek to take"... clean
    488: 0, # A murmur... clean

    # Records 489-499
    489: 3, # The debate that followed... "voice like a honed blade" — dead figure; ml:dead_figure confirmed
    490: 0, # Elara stood at the edge... clean
    491: 0, # Silas gave a slow... clean
    492: 7, # That night, the stranger's party... "shot through with tension" — dead figure + "held breath" + padding
    493: 5, # But she stayed... "It was their sovereignty, distilled into rushing water and spinning copper." — narrator_gloss explaining meaning
    494: 0, # The oven breathes... clean, specific
    495: 7, # Elias's hands go still... over-described wrongness + "map of gentle lines" + internal monologue padding
    496: 0, # "Good morning," she says... clean; ml:padding — just dialogue
    497: 0, # "Good morning," Elias replies... clean
    498: 0, # She looks at the array of loaves... clean
    499: 0, # The phrase is his private term... clean

    # Records 500-509
    500: 2, # A faint, almost imperceptible smile... "sense of wrongness lingers, thick as yeast" — emotion-telling accumulation; emotion
    501: 0, # Elias spends the next hour... clean
    502: 7, # He cannot stop thinking about her hands... obsessive internal monologue padding
    503: 7, # At two in the afternoon... "his heart leaps into his throat" — dead figure; padding
    504: 0, # The magic is working... clean
    505: 0, # The cold knot in Elias's stomach... clean (the actual insight is earned)
    506: 0, # The woman sags against the counter... clean
    507: 7, # "It's not more real"... rhetorical questioning + "yeasted wisdom" — narrator_gloss/padding
    508: 0, # "You are not her"... clean
    509: 0, # The woman... "lets out a sound that is half laugh, half sob" — clean, specific

    # Records 510-519
    510: 0, # "I never knew either of you"... clean
    511: 0, # She reaches into her coat pocket... clean
    512: 0, # The shop is silent... clean
    513: 0, # She pauses at the door... clean
    514: 0, # The bell chimes her out... clean; somewhat glossy at end but earned
    515: 0, # The tide was going out... clean, specific
    516: 0, # Leo's stomach dropped... "misty pride" is mild; no forbidden words; accumulation not reached; clean
    517: 0, # He needed to talk to someone... clean
    518: 0, # Magnus was mending a net... clean; ml:forbidden overreach — no forbidden words
    519: 0, # "Boat trouble," Leo said... clean; ml:forbidden overreach — no forbidden words

    # Records 520-529
    520: 0, # Magnus gave a short, sharp look... clean
    521: 0, # Leo thought of the cigarettes... clean
    522: 0, # "Is it? Or is it taking back a fraction"... clean
    523: 2, # The image of his grandfather shifted... "confusing swell of anger and awe" — named emotion; emotion
    524: 2, # "He protected you," Magnus corrected gently... "gently" magic adverb + named emotion context; forbidden/emotion hybrid; → emotion (2)
    525: 0, # "And now you have to decide"... clean
    526: 0, # Leo walked back along the creaking docks... clean
    527: 0, # That night, under a ceiling... clean
    528: 0, # The next morning, Leo went back... clean
    529: 0, # He wouldn't tell his mother... clean

    # Records 530-539
    530: 0, # But Leo's story would be different... clean
    531: 7, # The hum of the Memory Sanitization Chamber... over-described setup + "gentle, surgical hand" — dead figure; padding
    532: 0, # "Final pre-edit calibration complete"... clean
    533: 0, # She initiates the first pass... clean
    534: 0, # Elara engages the micro-filament probes... clean
    535: 2, # C-734 jerks in the chair... "profound relief" named + "He will be productive. He will be compliant." — narrator_gloss/emotion; emotion
    536: 0, # And the AI, in its relentless, logical sweep... clean
    537: 7, # *The designated casualty list*... quoted document as padding filler (with Chinese characters mid-text)
    538: 7, # Elara's blood runs cold... "blood runs cold" dead figure + over-explained discovery; padding
    539: 7, # The AI's voice is calm... padding dialogue explaining protocol

    # Records 540-549
    540: 7, # Elara's stone of an ethical weight... "chasm beneath" + "concrete overlaid on a memory of smoke" — dead figure accumulation; padding
    541: 0, # Her finger hovers over the console... clean
    542: 2, # C-734 sighs in his sleep... "profound relief" — named emotion; emotion
    543: 0, # Elara looks at the lattice... clean
    544: 0, # The decision point is not a grand... clean
    545: 0, # Or she can reach out... clean
    546: 0, # She thinks of Lira... clean
    547: 0, # "Minor synaptic anomaly"... clean
    548: 0, # They know... clean
    549: 0, # She does not protest... clean

    # Records 550-559
    550: 7, # They will take the slate... "dissolve the names, the faces" + padding accumulation
    551: 7, # But she has already done it... "living, remembering space inside her own skull" + "propagating" — narrator_gloss/padding
    552: 0, # She is a carrier... clean
    553: 0, # Elara listened... clean
    554: 0, # They came from both sides... clean, specific
    555: 0, # "I've got you, Jen"... clean
    556: 0, # She glanced at the militiaman's eyes... clean
    557: 1, # Her hands were never idle... "Needle decompression... A pressure dressing... Splinting" — tricolon + "Her mind splitting into streams" thesis-like
    558: 0, # "Medic?" he said... clean
    559: 0, # "Volunteer. From Kharkiv."... clean; ml:padding — dialogue, clean

    # Records 560-569
    560: 0, # "I can't look for him"... clean; ml:padding — clean dialogue
    561: 0, # He nodded, as if this was the only answer... clean
    562: 0, # He looked at her, really looked... clean
    563: 0, # "My oath is to do no harm"... clean
    564: 0, # He stood up slowly... clean
    565: 0, # Elara watched him go... clean
    566: 0, # The afternoon sun slanted... clean
    567: 0, # "Is he?" She looked at the black tag... clean
    568: 5, # The sun dipped below the horizon... narrator_gloss: "She was a weaver of fates with needle and suture" — explains meaning; also "achingly beautiful" + "thin, red line" — accumulation
    569: 7, # I did. Leo Voss... padding setup description

    # Records 570-579
    570: 0, # "He's gone, Jack"... short dialogue, "I'm... I'm scared" is one named emotion in direct dialogue; clean
    571: 0, # The rent was paid... clean
    572: 0, # The file was thin... clean
    573: 0, # "He was here that morning"... clean
    574: 0, # "No, no. He was… pleasing"... clean
    575: 0, # Leo's apartment building... clean, specific
    576: 7, # "He paid on time"... accumulation of "Quiet" fragments; padding
    577: 7, # I rifled through the closets... "Why take the business brain and leave the clothes?" — rhetorical question padding
    578: 7, # The canvas tote... "A ghost in the crowd" dead figure + padding description
    579: 0, # "New Mexico. Santa Fe."... clean

    # Records 580-592
    580: 0, # Leonard Vance. A slight anagram... clean
    581: 7, # Santa Fe was all adobe and turquoise... "thin, dry air that scraped your throat" — cliché + over-described stakeout; padding
    582: 0, # I sat two benches down... clean
    583: 0, # He stared at the envelope... clean
    584: 0, # He shook his head... clean
    585: 0, # My personal code is simple... clean
    586: 0, # A tear finally tracked... "private horror" is mildly elevated but no forbidden words; "resources" is dialogue-natural; clean
    587: 0, # I had my lead... no forbidden words; clean
    588: 0, # But looking at Leo... "hated bullies" is clean; no forbidden words; clean
    589: 0, # I stood up... clean
    590: 0, # "Because I'm going to tell"... clean
    591: 7, # "It's justice?" I shrugged... "justice" as mock-profound label + narrator explaining code = narrator_gloss/padding
    592: 0, # I walked away... clean
}


def main():
    input_path = "/home/ben/code/prose-doctor/corpus/review_chunk_5.jsonl"
    output_path = "/home/ben/code/prose-doctor/corpus/reviewed_chunk_5.jsonl"

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    assert len(records) == 593, f"Expected 593 records, got {len(records)}"

    with open(output_path, "w") as out:
        for i, record in enumerate(records):
            label = CLASSIFICATIONS[i]
            class_name = CLASS_NAMES[label]
            record["label"] = label
            record["class_name"] = class_name
            record["method"] = "llm_review"
            out.write(json.dumps(record) + "\n")

    # Summary
    from collections import Counter
    label_counts = Counter(CLASSIFICATIONS[i] for i in range(593))
    print("Classification summary:")
    for label in sorted(label_counts):
        print(f"  {label} ({CLASS_NAMES[label]}): {label_counts[label]}")
    print(f"Total: {sum(label_counts.values())}")


if __name__ == "__main__":
    main()
