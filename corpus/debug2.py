#!/usr/bin/env python3
"""Check padding pattern hits across corpus."""
import json
import re

def padding_score(text):
    score = 0.0
    hits = []
    if re.search(r'\bbegan to\b', text, re.IGNORECASE):
        score += 1; hits.append('began_to')
    if re.search(r'\bseemed to (?!be (?:a|an|the|more|less)\b)', text, re.IGNORECASE):
        score += 1; hits.append('seemed_to')
    if re.search(r'\bfound (?:herself|himself|themselves|myself)\b', text, re.IGNORECASE):
        score += 1; hits.append('found_herself')
    if re.search(r"\bcouldn'?t help but\b", text, re.IGNORECASE):
        score += 1; hits.append('couldnt_help_but')
    if re.search(r'\bserves? as\b', text, re.IGNORECASE):
        score += 1; hits.append('serves_as')
    if re.search(r'\bstands? as\b', text, re.IGNORECASE):
        score += 1; hits.append('stands_as')
    if re.search(r'\b(?:it\'?s|it is) worth (?:noting|mentioning)\b', text, re.IGNORECASE):
        score += 1; hits.append('worth_noting')
    if re.search(r'\bimportantly,?\s', text, re.IGNORECASE):
        score += 0.5; hits.append('importantly')
    if re.search(r'\binterestingly,?\s', text, re.IGNORECASE):
        score += 1; hits.append('interestingly')
    if re.search(r'\bnotably,?\s', text, re.IGNORECASE):
        score += 0.5; hits.append('notably')
    return score, hits

def forbidden_score(text):
    score = 0
    hits = []
    words_patterns = [
        (r'\btapestry\b', 'tapestry'),
        (r'\bsymphony\b', 'symphony'),
        (r'\bcrucible\b', 'crucible'),
        (r'\btestament\b', 'testament'),
        (r'\bvisceral\b', 'visceral'),
        (r'\bpalimpsest\b', 'palimpsest'),
        (r'\bgossamer\b', 'gossamer'),
        (r'\bethereal\b', 'ethereal'),
        (r'\bineffable\b', 'ineffable'),
        (r'\btableau\b', 'tableau'),
        (r'\bliminal\b', 'liminal'),
        (r'\balchemy\b', 'alchemy'),
        (r'\bresonan(?:ce|t|ted|tes|ting)\b', 'resonan*'),
        (r'\bprofound(?:ly|ity)?\b', 'profound'),
        (r'\bparadigm\b', 'paradigm'),
        (r'\bsynergy\b', 'synergy'),
        (r'\becosystem\b', 'ecosystem'),
        (r'\bframework\b', 'framework'),
        (r'\bdelve[ds]?\b', 'delve'),
        (r'\butiliz(?:e|es|ed|ing)\b', 'utilize'),
        (r'\bleverage[ds]?\b', 'leverage'),
        (r'\brobust\b', 'robust'),
        (r'\bharness(?:es|ed|ing)?\b', 'harness'),
        (r'\bstreamlin(?:e|es|ed|ing)\b', 'streamline'),
    ]
    for pat, name in words_patterns:
        if re.search(pat, text, re.IGNORECASE):
            score += 1
            hits.append(name)
    return score, hits

records = []
with open('/home/ben/code/prose-doctor/corpus/review_chunk_1.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print("=== Records with padding score >= 2 ===")
pad_count = 0
for i, r in enumerate(records):
    ps, hits = padding_score(r['text'])
    if ps >= 2:
        pad_count += 1
        print(f"Record {i}: ps={ps} hits={hits}")
        print(f"  TEXT: {r['text'][:120]}")
        print()
print(f"Total: {pad_count}")

print()
print("=== Records with forbidden score >= 2 ===")
forb_count = 0
for i, r in enumerate(records):
    fs, hits = forbidden_score(r['text'])
    if fs >= 2:
        forb_count += 1
        print(f"Record {i}: fs={fs} hits={hits}")
        print(f"  TEXT: {r['text'][:120]}")
        print()
print(f"Total: {forb_count}")

print()
print("=== Records with emotion score >= 1 ===")
em_count = 0
for i, r in enumerate(records):
    text = r['text']
    emotion_nouns = r'(?:dread|grief|shame|guilt|terror|despair|longing|relief|joy|anger|fear|anxiety|panic|sadness|sorrow|loss|helplessness|hopelessness|foreboding|unease|regret|resentment)'
    e1 = bool(re.search(r'\b(?:felt|feels) (?:a )?(?:surge|wave|rush|pang|twinge|flush|stab|bolt|jolt|knot) of \w', text, re.IGNORECASE))
    e2 = bool(re.search(r'\ba (?:sense|wave|rush|surge|flood|tide|weight|wall|knot) of ' + emotion_nouns, text, re.IGNORECASE))
    e3 = bool(re.search(emotion_nouns + r' (?:gripped|seized|washed over|flooded|overwhelmed|settled over|enveloped|consumed|clutched) (?:him|her|them|me)', text, re.IGNORECASE))
    e4 = bool(re.search(r'\b(?:felt|feels) (?:a )?(?:profound|sudden|deep|overwhelming|acute|sharp|unexpected)[,\s]', text, re.IGNORECASE))
    e5 = bool(re.search(r'\b(?:felt|feels) profoundly\b', text, re.IGNORECASE))
    e6 = bool(re.search(r'\ba (?:cold|sudden|icy|chilling) (?:clarity|certainty|realization|recognition)\b', text, re.IGNORECASE))
    score = sum([e1, e2, e3, e4, e5, e6])
    if score >= 1:
        em_count += 1
        if em_count <= 20:
            print(f"Record {i}: es={score} ({e1},{e2},{e3},{e4},{e5},{e6})")
            print(f"  TEXT: {r['text'][:120]}")
            print()
print(f"Total: {em_count}")
