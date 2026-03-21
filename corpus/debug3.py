#!/usr/bin/env python3
"""Check specific records I identified as slop during reading."""
import json
import re

records = []
with open('/home/ben/code/prose-doctor/corpus/review_chunk_1.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Records I noted as potentially slop during manual reading:
# Record 8: "He felt a profound, terrifying certainty." -> EMOTION or FORBIDDEN (profound)
# Record 9: (context_after) The realization was not a sudden flash...  -> check
# Record 56: "Kepler-186d. The gas giant. A dead end." -> STANDALONE?
# Record 82: "Aris feels a sudden, icy clarity." -> EMOTION
# Record 101: resonance/profound/framework -> FORBIDDEN
# Record 102: "the exhaustion, the stress, the four-day caffeine haze—it all falls away, replaced by a profound, unexpected sense of recognition." -> FORBIDDEN + EMOTION
# Record 350: "And there it was. Not about the money..." -> THESIS
# Record 370: "He feels a profound, almost debilitating exhaustion" -> EMOTION
# Record 371: (emotion) -> check
# Record 454: resonant/synergy -> FORBIDDEN

check_indices = [8, 9, 56, 82, 101, 102, 350, 354, 358, 370, 371, 387, 388, 454]

def get_signals(text):
    wc = len(text.split())

    # Forbidden
    forbidden_words = [
        (r'\btapestry\b', 'tapestry'), (r'\bsymphony\b', 'symphony'),
        (r'\bcrucible\b', 'crucible'), (r'\btestament\b', 'testament'),
        (r'\bvisceral\b', 'visceral'), (r'\bpalimpsest\b', 'palimpsest'),
        (r'\bgossamer\b', 'gossamer'), (r'\bethereal\b', 'ethereal'),
        (r'\bineffable\b', 'ineffable'), (r'\btableau\b', 'tableau'),
        (r'\bliminal\b', 'liminal'), (r'\balchemy\b', 'alchemy'),
        (r'\bresonan(?:ce|t|ted|tes|ting)\b', 'resonan*'),
        (r'\bprofound(?:ly|ity)?\b', 'profound'),
        (r'\bparadigm\b', 'paradigm'), (r'\bsynergy\b', 'synergy'),
        (r'\becosystem\b', 'ecosystem'), (r'\bframework\b', 'framework'),
        (r'\bdelve[ds]?\b', 'delve'), (r'\butiliz(?:e|es|ed|ing)\b', 'utilize'),
        (r'\bleverage[ds]?\b', 'leverage'), (r'\brobust\b', 'robust'),
        (r'\bharness(?:es|ed|ing)?\b', 'harness'),
        (r'\bstreamlin(?:e|es|ed|ing)\b', 'streamline'),
    ]
    fs = 0; fhits = []
    for pat, name in forbidden_words:
        if re.search(pat, text, re.IGNORECASE):
            fs += 1; fhits.append(name)

    # Emotion
    emotion_nouns = r'(?:dread|grief|shame|guilt|terror|despair|longing|relief|joy|anger|fear|anxiety|panic|sadness|sorrow|loss|helplessness|hopelessness|foreboding|unease|regret|resentment)'
    e1 = bool(re.search(r'\b(?:felt|feels) (?:a )?(?:surge|wave|rush|pang|twinge|flush|stab|bolt|jolt|knot) of \w', text, re.IGNORECASE))
    e2 = bool(re.search(r'\ba (?:sense|wave|rush|surge|flood|tide|weight|wall|knot) of ' + emotion_nouns, text, re.IGNORECASE))
    e3 = bool(re.search(emotion_nouns + r' (?:gripped|seized|washed over|flooded|overwhelmed|settled over|enveloped|consumed|clutched) (?:him|her|them|me)', text, re.IGNORECASE))
    e4 = bool(re.search(r'\b(?:felt|feels) (?:a )?(?:profound|sudden|deep|overwhelming|acute|sharp|unexpected)[,\s]', text, re.IGNORECASE))
    e5 = bool(re.search(r'\b(?:felt|feels) profoundly\b', text, re.IGNORECASE))
    e6 = bool(re.search(r'\ba (?:cold|sudden|icy|chilling) (?:clarity|certainty|realization|recognition)\b', text, re.IGNORECASE))
    es = sum([e1, e2, e3, e4, e5, e6])

    # Thesis
    t1 = bool(re.search(r'It was not .{3,80}\. It was ', text, re.IGNORECASE))
    t2 = bool(re.search(r"It wasn'?t .{3,80}\. It was ", text, re.IGNORECASE))
    t3 = bool(re.search(r"It wasn'?t .{3,80}\. It'?s ", text, re.IGNORECASE))
    ts = sum([t1, t2, t3])

    # Padding
    ps = 0
    if re.search(r'\bbegan to\b', text, re.IGNORECASE): ps += 1
    if re.search(r'\bseemed to\b', text, re.IGNORECASE): ps += 1
    if re.search(r'\bfound (?:herself|himself|themselves|myself)\b', text, re.IGNORECASE): ps += 1
    if re.search(r"\bcouldn'?t help but\b", text, re.IGNORECASE): ps += 1

    return wc, fs, fhits, es, ts, ps

for idx in check_indices:
    r = records[idx]
    wc, fs, fhits, es, ts, ps = get_signals(r['text'])
    ml = r.get('ml_class_name', '?')
    print(f"Record {idx} (ml={ml}): wc={wc} fs={fs}{fhits} es={es} ts={ts} ps={ps}")
    print(f"  TEXT: {r['text'][:140]}")
    print()

# Also check "Not about the money" record 350
print("--- checking text around 'not about the money' ---")
for i, r in enumerate(records):
    if 'not about the money' in r['text'].lower() or 'And there it was.' in r['text']:
        wc, fs, fhits, es, ts, ps = get_signals(r['text'])
        print(f"Record {i} (ml={r.get('ml_class_name')}): wc={wc} ts={ts}")
        print(f"  TEXT: {r['text'][:140]}")
        print()
