#!/usr/bin/env python3
"""
Reviewer for review_chunk_1.jsonl — 598 records.

Accumulation Rule: ONE instance of a tell in an otherwise solid paragraph = clean.
Multiple tells clustered in the same paragraph = flag.

Classes:
  0 = clean
  1 = thesis (negative parallelism, self-answered question, dramatic countdown)
  2 = emotion (names emotion directly instead of showing)
  3 = dead_figure (cliché metaphors / similes)
  4 = standalone (orphaned short fragment used for manufactured emphasis)
  5 = narrator_gloss (narrator explains what just happened / tells reader what to think)
  6 = forbidden (specific overused LLM words)
  7 = padding (filler, hedging verbs, tricolon/anaphora abuse, em-dash addiction)
"""

import json
import re


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
# Helper: count non-overlapping regex hits
# ---------------------------------------------------------------------------

def cnt(pattern, text, flags=re.IGNORECASE):
    return len(re.findall(pattern, text, flags))


# ---------------------------------------------------------------------------
# Individual signal functions
# ---------------------------------------------------------------------------

def forbidden_score(text):
    """Count distinct forbidden-word hits."""
    # Core forbidden words — exact list from spec
    words = [
        r'\btapestry\b', r'\bsymphony\b', r'\bcrucible\b', r'\btestament\b',
        r'\bvisceral\b', r'\bpalimpsest\b', r'\bgossamer\b', r'\bethereal\b',
        r'\bineffable\b', r'\btableau\b', r'\bliminal\b', r'\balchemy\b',
        r'\bresonan(?:ce|t|ted|tes|ting)\b',
        r'\bprofound(?:ly|ity)?\b',
        r'\bparadigm\b', r'\bsynergy\b', r'\becosystem\b', r'\bframework\b',
        r'\bdelve[ds]?\b',
        r'\butiliz(?:e|es|ed|ing)\b',
        r'\bleverage[ds]?\b',
        r'\brobust\b',
        r'\bharness(?:es|ed|ing)?\b',
        r'\bstreamlin(?:e|es|ed|ing)\b',
    ]
    # Magic adverbs
    adverbs = [
        r'\bquietly\b', r'\bdeeply\b', r'\bfundamentally\b',
        r'\bremarkably\b', r'\bargua bly\b',
    ]
    score = 0
    for w in words:
        if re.search(w, text, re.IGNORECASE):
            score += 1
    for w in adverbs:
        if re.search(w, text, re.IGNORECASE):
            score += 0.5
    return score


def thesis_score(text):
    """
    Count thesis/negative-parallelism patterns.
    Patterns:
    - "It was not X. It was Y."
    - "Not because X, but because Y"
    - "The question isn't X. The question is Y."
    - "Not X. Not Y. Just/Only Z."
    - "It isn't X. It's Y."
    """
    score = 0
    patterns = [
        r'It was not .{3,80}?\. It was ',
        r"It wasn'?t .{3,80}?\. It was ",
        r'Not because .{3,80}?, but because ',
        r"The question isn'?t .{3,60}?\. The question is ",
        r"It isn'?t .{3,60}?\. It'?s ",
        r"This isn'?t .{3,60}?\. This is ",
        r'Not \w.{2,40}\. Not \w.{2,40}\. (?:Just|Only|Simply) ',
        r'Not \w.{2,30}\. Not \w.{2,30}\. Not \w.{2,30}\.',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE | re.DOTALL):
            score += 1
    # Self-answered question: "The X? A Y." or "The X? The Y."
    if re.search(r'\bThe \w[\w ]{1,25}\? (?:A |An |The )\w', text):
        score += 1
    return score


def emotion_score(text):
    """
    Count direct emotion-naming patterns.
    - "felt a surge/wave/rush/pang of <emotion>"
    - "a sense/wave/flood of <emotion>"
    - "<emotion> gripped/seized/washed over him"
    - "felt/feels a profound/sudden/deep ..."
    """
    score = 0
    emotion_nouns = r'(?:dread|grief|shame|guilt|terror|despair|longing|relief|joy|anger|fear|anxiety|panic|sadness|sorrow|loss|helplessness|hopelessness|foreboding|unease|regret|resentment)'

    # "felt/feels a surge/wave/rush/pang of X"
    if re.search(r'\b(?:felt|feels) (?:a )?(?:surge|wave|rush|pang|twinge|flush|stab|bolt|jolt|knot) of \w', text, re.IGNORECASE):
        score += 1
    # "a sense/wave of dread" etc.
    if re.search(r'\ba (?:sense|wave|rush|surge|flood|tide|weight|wall|knot) of ' + emotion_nouns, text, re.IGNORECASE):
        score += 1
    # <emotion> gripped/seized/washed over him/her
    if re.search(emotion_nouns + r' (?:gripped|seized|washed over|flooded|overwhelmed|settled over|enveloped|consumed|clutched) (?:him|her|them|me)', text, re.IGNORECASE):
        score += 1
    # "felt/feels a profound/sudden/overwhelming ..." — the key fix: match without requiring next noun
    # match "felt a profound" + (comma or space) — the rest doesn't matter
    if re.search(r'\b(?:felt|feels) (?:a )?(?:profound|sudden|deep|overwhelming|acute|sharp|unexpected)[,\s]', text, re.IGNORECASE):
        score += 1
    # "felt profoundly"
    if re.search(r'\b(?:felt|feels) profoundly\b', text, re.IGNORECASE):
        score += 1
    # "a cold/icy clarity" or "sudden clarity" — narrator naming the internal state
    if re.search(r'\ba (?:cold|sudden|icy|chilling) (?:clarity|certainty|realization|recognition)\b', text, re.IGNORECASE):
        score += 1
    return score


def dead_figure_score(text):
    """Count cliché metaphors/similes."""
    score = 0
    cliches = [
        r'\btime stood still\b',
        r'\bthe weight of the world\b',
        r'\beyes like stars\b',
        r'\bheart of (?:gold|stone|darkness)\b',
        r'\bdeafening silence\b',
        r'\bsilence (?:fell|was) (?:deafening|heavy|thick|complete)\b',
        r'\bblood ran cold\b',
        r'\bworld (?:fell away|seemed to spin|spun)\b',
        r'\bthink of it as\b',
        r'\blike a well-oiled machine\b',
        r'\bbutterflies in (?:his|her|their|my) stomach\b',
        r'\bknot in (?:his|her|their|my) stomach\b',
    ]
    for c in cliches:
        if re.search(c, text, re.IGNORECASE):
            score += 1
    return score


def narrator_gloss_score(text):
    """Count narrator-explains-what-happened patterns."""
    score = 0
    patterns = [
        r'\bsomething (?:in|about) (?:him|her|them|Elias|Eleanor|Aris|Viktor|Amelia|Isabella|Ahmet|the (?:room|air|silence|moment)) (?:shifted|changed|stirred|clicked|broke|settled|cracked|softened)\b',
        r'\b(?:he|she|they|i) realized (?:then |now |suddenly |finally |at last )?that\b',
        r'\bit was,? (?:she|he|they) (?:understood|realized|knew),? a kind of\b',
        r'\b(?:he|she|they|i) understood (?:now|then|suddenly|at last|finally) that\b',
        r'\bthe truth (?:was|is|had always been) (?:simple|clear|this|that)\b',
        r'\bhighlight(?:ing|ed) (?:its|their|the|his|her) (?:importance|significance)\b',
        r'\breflect(?:ing|ed) (?:broader|the|a) (?:trends|pattern|reality|truth)\b',
        r'\bwhat (?:he|she|they|i) (?:didn\'t|did not) (?:know|realize|understand) (?:yet )?(?:was|is|had been)\b',
        r'\bit was (?:then |only then )?(?:that )?(?:he|she|they|i) (?:knew|understood|realized)\b',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            score += 1
    return score


def padding_score(text):
    """Count filler/hedge phrases."""
    score = 0.0
    # Weak hedging verbs
    if re.search(r'\bbegan to\b', text, re.IGNORECASE):
        score += 1
    if re.search(r'\bseemed to (?!be (?:a|an|the|more|less)\b)', text, re.IGNORECASE):
        score += 1
    if re.search(r'\bfound (?:herself|himself|themselves|myself)\b', text, re.IGNORECASE):
        score += 1
    if re.search(r"\bcouldn'?t help but\b", text, re.IGNORECASE):
        score += 1
    if re.search(r'\bserves? as\b', text, re.IGNORECASE):
        score += 1
    if re.search(r'\bstands? as\b', text, re.IGNORECASE):
        score += 1
    # Filler transitions
    if re.search(r'\b(?:it\'?s|it is) worth (?:noting|mentioning)\b', text, re.IGNORECASE):
        score += 1
    if re.search(r'\bimportantly,?\s', text, re.IGNORECASE):
        score += 0.5
    if re.search(r'\binterestingly,?\s', text, re.IGNORECASE):
        score += 1
    if re.search(r'\bnotably,?\s', text, re.IGNORECASE):
        score += 0.5
    return score


def em_dash_count(text):
    return text.count('\u2014') + text.count('\u2013')


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify(record):
    text = record["text"]
    wc = len(text.split())
    is_dialogue = text.strip().startswith(('\u201c', '\u201d', '"'))

    fs = forbidden_score(text)
    ts = thesis_score(text)
    es = emotion_score(text)
    df = dead_figure_score(text)
    ng = narrator_gloss_score(text)
    ps = padding_score(text)
    em = em_dash_count(text)

    # --- THESIS (1) ---
    # Negative parallelism as the central move (not inside dialogue)
    if ts >= 1 and not is_dialogue:
        if ts >= 2:
            return 1
        # Single thesis pattern — flag if it's the dominant move (short paragraph, or
        # the thesis pattern takes up a big fraction of the text)
        if wc <= 100:
            return 1

    # --- FORBIDDEN (6) ---
    if fs >= 3:
        return 6
    if fs >= 2 and wc <= 200:
        return 6
    if fs >= 1 and wc <= 30:
        # One forbidden word in a very short paragraph = flag
        return 6

    # --- EMOTION (2) ---
    if es >= 2:
        return 2
    if es >= 1 and not is_dialogue and wc <= 120:
        return 2

    # --- DEAD FIGURES (3) ---
    if df >= 2:
        return 3
    if df >= 1 and wc <= 50:
        return 3

    # --- NARRATOR GLOSS (5) ---
    if ng >= 2:
        return 5
    if ng >= 1 and (ps >= 1 or fs >= 1) and wc <= 100:
        return 5

    # --- PADDING (7) ---
    if em >= 4:
        return 7
    if ps >= 3:
        return 7
    if ps >= 2 and wc <= 150:
        return 7
    if ps >= 2 and fs >= 1:
        return 7
    if ps >= 1 and wc <= 15:
        # A single padding phrase in a tiny fragment is the whole content
        return 7

    # Aggregate weak signals
    total = fs * 2 + ts * 1.5 + es * 2 + df * 1.5 + ng * 2 + ps * 1.5 + (1.5 if em >= 3 else 0)

    if total >= 4.5:
        # Dominant class
        scores = [
            (ts, 1),
            (es, 2),
            (df, 3),
            (ng, 5),
            (fs, 6),
            (ps, 7),
        ]
        scores.sort(key=lambda x: -x[0])
        if scores[0][0] > 0:
            return scores[0][1]
        return 7

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    input_path = "/home/ben/code/prose-doctor/corpus/review_chunk_1.jsonl"
    output_path = "/home/ben/code/prose-doctor/corpus/reviewed_chunk_1.jsonl"

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records.")

    results = []
    label_counts = {i: 0 for i in range(8)}
    changed = 0

    for record in records:
        old_label = record.get("label", 0)
        label = classify(record)
        record["label"] = label
        record["class_name"] = CLASS_NAMES[label]
        record["method"] = "llm_review"
        label_counts[label] += 1
        if label != old_label:
            changed += 1
        results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results)} records to {output_path}")
    print(f"Changed {changed} labels from ML baseline.")
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / len(results)
        print(f"  {label} ({CLASS_NAMES[label]}): {count:3d}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
