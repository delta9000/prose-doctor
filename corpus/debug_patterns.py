#!/usr/bin/env python3
"""Debug pattern matching on known slop examples."""
import re

tests = [
    # Should be EMOTION (2)
    "He felt a profound, terrifying certainty.",
    "He felt a profound, almost debilitating exhaustion settle over him",
    "Aris feels a sudden, icy clarity. He's been trying to decode words.",
    "He felt a strange, dissociative calm settle over him.",
    "She felt a surge of anger.",
    "A sense of dread washed over him.",
    "Fear gripped her.",
    # Should be FORBIDDEN (6)
    "His work was a testament to the resilience of the human spirit.",
    "It created a profound resonance in the room.",
    "The resonant thrumming was the soundtrack to his life.",
    "a heavy, comforting perfume — profound frustration — resonant sounds",
    # Should be THESIS (1)
    "It was not about the money. It was about something else entirely.",
    "It wasn't a sudden flash of insight. It was a slow, cold seep.",
    # Should be PADDING (7)
    "She began to understand what he meant.",
    "He found himself thinking about the old days.",
    "She couldn't help but smile at the irony.",
]

for t in tests:
    wc = len(t.split())

    # Emotion patterns
    e1 = bool(re.search(r'\bfelt (?:a )?(?:surge|wave|rush|pang|twinge|flush|stab|bolt|jolt|knot) of \w', t, re.IGNORECASE))
    e2 = bool(re.search(r'\ba (?:sense|wave|rush|surge|flood|tide|weight|wall|knot) of (?:dread|grief|shame|guilt|terror|despair|longing|relief|joy|anger|fear|anxiety|panic|sadness|sorrow|loss|helplessness|hopelessness|foreboding|unease|regret|resentment)', t, re.IGNORECASE))
    e3 = bool(re.search(r'(?:dread|grief|shame|guilt|terror|despair|longing|relief|joy|anger|fear|anxiety|panic|sadness|sorrow|loss) (?:gripped|seized|washed over|flooded|overwhelmed|settled over|enveloped|consumed)', t, re.IGNORECASE))
    e4 = bool(re.search(r'\bfelt (?:a )?(?:profound|sudden|deep|overwhelming|acute|sharp|unexpected) (?:sense|feeling|rush|wave|certainty|clarity|sadness|grief|fear|terror|dread|longing|peace|calm|exhaustion|weight|chill)', t, re.IGNORECASE))
    e5 = bool(re.search(r'\bfelt profoundly\b', t, re.IGNORECASE))
    emotion = e1 or e2 or e3 or e4 or e5

    # Forbidden
    f1 = bool(re.search(r'\bprofound(?:ly|ity)?\b', t, re.IGNORECASE))
    f2 = bool(re.search(r'\bresonan(?:ce|t|ted|tes|ting)\b', t, re.IGNORECASE))
    f3 = bool(re.search(r'\btestament\b', t, re.IGNORECASE))
    forbidden = f1 or f2 or f3

    # Thesis
    t1 = bool(re.search(r'It was not .{3,80}\. It was ', t, re.IGNORECASE))
    t2 = bool(re.search(r"It wasn'?t .{3,80}\. It was ", t, re.IGNORECASE))
    t3 = bool(re.search(r"It wasn'?t .{3,80}\. It'?s ", t, re.IGNORECASE))
    thesis = t1 or t2 or t3

    # Padding
    p1 = bool(re.search(r'\bbegan to\b', t, re.IGNORECASE))
    p2 = bool(re.search(r'\bfound (?:herself|himself|themselves|myself)\b', t, re.IGNORECASE))
    p3 = bool(re.search(r"\bcouldn'?t help but\b", t, re.IGNORECASE))
    padding = p1 or p2 or p3

    print(f"TEXT: {t[:70]}")
    print(f"  wc={wc} emotion={emotion}({e1},{e2},{e3},{e4},{e5}) forbidden={forbidden}({f1},{f2},{f3}) thesis={thesis} padding={padding}")
    print()
