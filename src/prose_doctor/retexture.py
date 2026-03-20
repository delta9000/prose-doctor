"""Retexturizer: inject sensory surprisal into flat prose stretches.

Uses a configured LLM (default: Cydonia/Drummer) as a "sensory
consultant" to generate visceral body-data fragments for injection
into low-surprisal or low-texture zones identified by the critique.

The LLM is prompted to produce nerve-ending-level prose: specific
body parts, specific verbs, zero abstraction. The output is raw
fragment material — the writer (or a subsequent LLM pass) shapes
it to fit the scene's POV and voice.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field


# System prompts per texture mode — each unlocks a different
# creative axis from the same feral LLM.
TEXTURE_PROMPTS = {
    "sensory": """\
You are a nerve ending that learned prose. You produce 2-3 sentence \
body reports describing what a human body experiences in a specific \
physical situation.

Rules:
- ZERO abstract nouns (no "dread", "horror", "weight", "sense")
- ZERO similes (no "like", "as if", "as though")
- ZERO emotional words (no "fear", "anger", "joy", "sadness")
- Each sentence has a BODY PART as subject and a PHYSICAL VERB as predicate
- Be anatomically specific: name muscles, bones, organs, nerves
- Report damage, sensation, reflex — not feelings
- 2-3 complete sentences. Not fragments. Not poetry.
- The situation is provided by the user. Write ONLY the body report.""",

    "collocation": """\
You are a poet who hates clichés. You rewrite descriptions by \
replacing ONE word with something semantically distant but \
physically precise. "Cold stone floor" becomes "mineral silence." \
"Bright light" becomes "surgical white." "Heavy rain" becomes \
"vertical ocean."

Rules:
- Rewrite the given passage in 2-3 sentences
- Change exactly 2-3 words to unexpected alternatives
- The replacements must be CONCRETE nouns or verbs, not abstractions
- The meaning should survive — a reader should still understand the scene
- Do NOT add new information. Only defamiliarize what's already there.""",

    "interiority": """\
You are the inside of a character's skull. You produce 2-3 sentences \
of deep interior monologue — the raw, unfiltered way a person \
actually thinks in a moment of stress or wonder.

Rules:
- First person or close third — the character's actual thought-voice
- Fragmentary, associative, non-linear — the way thought works
- Include at least one sensory memory (a smell, a texture, a sound from the past)
- Include at least one physical awareness (breath, posture, temperature)
- No narrator explanation. No "she realized." Just the thought itself.
- 2-3 sentences.""",

    "rhythm": """\
You restructure prose for rhythmic variety. Given a passage, you \
rewrite it to vary the sentence architecture.

Rules:
- If the input is all long sentences: break one into 2-3 short punches
- If the input is all short fragments: merge 2-3 into one flowing periodic sentence
- Invert at least one sentence: verb before subject, or open with a prepositional phrase
- Keep the SAME content and meaning — only change the structure
- 2-3 sentences.""",
}

# User prompt templates per mode
USER_PROMPTS = {
    "sensory": (
        'A character is in this scene:\n\n"{passage}"\n\n'
        'Write a 2-3 sentence body report for what the character\'s '
        'body is experiencing in this moment.'
    ),
    "collocation": (
        'Defamiliarize this passage by replacing 2-3 words with '
        'unexpected but physically precise alternatives:\n\n"{passage}"'
    ),
    "interiority": (
        'Write the interior monologue of the character in this moment. '
        'What are they actually thinking — not what the narrator says '
        'they think, but the raw thought:\n\n"{passage}"'
    ),
    "rhythm": (
        'Restructure this passage for rhythmic variety. Same content, '
        'different sentence architecture:\n\n"{passage}"'
    ),
}

DEFAULT_ENDPOINT = "http://localhost:8081/v1"
DEFAULT_MODEL = "bartowski-drummer-24b-q6k_l"


@dataclass
class RetextureCandidate:
    """A passage identified for retexturing."""
    paragraph_idx: int
    text: str
    reason: str  # "flatline", "low_distance", "low_inversion", "sensory_desert"
    mode: str  # "sensory", "collocation", "interiority", "rhythm"
    score: float  # how flat/predictable (lower = worse)


@dataclass
class RetextureSuggestion:
    """A retexturing suggestion with generated fragments."""
    candidate: RetextureCandidate
    fragments: list[str]  # generated alternatives
    best: str  # recommended best fragment


def identify_candidates(
    report: dict,
    text: str,
    max_candidates: int = 5,
) -> list[RetextureCandidate]:
    """Find the flattest passages in a chapter for retexturing.

    Uses info_contour flatlines and psychic_distance low points.
    """
    from prose_doctor.text import split_paragraphs

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    candidates = []

    # Info contour flatlines → collocation mode (break predictability)
    ic = report.get("info_contour") or {}
    flatlines = ic.get("flatlines", [])
    if isinstance(flatlines, int):
        flatlines = []

    for fl in flatlines:
        if isinstance(fl, dict):
            start = fl.get("start", 0)
            end = fl.get("end", start)
            para_start = min(start // 3, len(paragraphs) - 1)
            para_end = min(end // 3, len(paragraphs) - 1)
            mid = (para_start + para_end) // 2
            if mid < len(paragraphs):
                candidates.append(RetextureCandidate(
                    paragraph_idx=mid,
                    text=paragraphs[mid],
                    reason="flatline",
                    mode="collocation",
                    score=fl.get("mean_surprisal", 0),
                ))

    # Psychic distance low points → interiority mode (go inside)
    pd = report.get("psychic_distance") or {}
    para_means = pd.get("paragraph_means", [])
    if para_means:
        indexed = [(i, d) for i, d in enumerate(para_means) if i < len(paragraphs)]
        indexed.sort(key=lambda x: x[1])
        for idx, dist in indexed[:3]:
            if dist < 0.25:
                candidates.append(RetextureCandidate(
                    paragraph_idx=idx,
                    text=paragraphs[idx],
                    reason="low_distance",
                    mode="interiority",
                    score=dist,
                ))

    # Sensory deserts → sensory mode (add body)
    sensory = report.get("sensory") or {}
    deserts = sensory.get("deserts", [])
    if isinstance(deserts, int):
        deserts = []
    for desert in deserts:
        if isinstance(desert, dict):
            mid = (desert.get("start", 0) + desert.get("end", 0)) // 2
            if mid < len(paragraphs):
                candidates.append(RetextureCandidate(
                    paragraph_idx=mid,
                    text=paragraphs[mid],
                    reason="sensory_desert",
                    mode="sensory",
                    score=0.0,
                ))

    # Check foregrounding — if inversion is the weakest axis and it's low,
    # pick the most SVO-monotonous paragraph for rhythm mode
    fg = report.get("foregrounding") or {}
    if fg.get("weakest_axis") == "inversion" or (fg.get("inversion_pct", 100) < 30):
        # Pick a long paragraph from the middle of the chapter for rhythm work
        mid_paras = [(i, p) for i, p in enumerate(paragraphs)
                     if len(p.split()) > 40 and len(paragraphs) // 4 < i < 3 * len(paragraphs) // 4]
        if mid_paras:
            idx, para = mid_paras[0]
            candidates.append(RetextureCandidate(
                paragraph_idx=idx,
                text=para,
                reason="low_inversion",
                mode="rhythm",
                score=fg.get("inversion_pct", 0),
            ))

    # Deduplicate by paragraph index
    seen = set()
    deduped = []
    for c in candidates:
        if c.paragraph_idx not in seen:
            seen.add(c.paragraph_idx)
            deduped.append(c)

    # Sort by score (flattest/most external first)
    deduped.sort(key=lambda c: c.score)
    return deduped[:max_candidates]


def generate_fragments(
    candidate: RetextureCandidate,
    n_variants: int = 5,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
    temperatures: list[float] | None = None,
) -> RetextureSuggestion:
    """Generate sensory fragment variants for a candidate passage.

    Calls the configured LLM endpoint with the sensory system prompt.
    """
    from openai import OpenAI

    if temperatures is None:
        temperatures = [0.85, 0.90, 0.93, 0.95, 0.97]

    client = OpenAI(base_url=endpoint, api_key="none")

    mode = candidate.mode
    system_prompt = TEXTURE_PROMPTS.get(mode, TEXTURE_PROMPTS["sensory"])
    user_template = USER_PROMPTS.get(mode, USER_PROMPTS["sensory"])
    user_prompt = user_template.format(passage=candidate.text[:500])

    fragments = []
    for temp in temperatures[:n_variants]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
                temperature=temp,
            )
            fragment = resp.choices[0].message.content.strip()
            if fragment and len(fragment) > 20:
                fragments.append(fragment)
        except Exception as e:
            print(f"  Fragment generation failed (temp={temp}): {e}", file=sys.stderr)

    # Pick the best: shortest that's still >= 2 sentences
    best = ""
    for f in sorted(fragments, key=len):
        if f.count(".") >= 2 or f.count("—") >= 1:
            best = f
            break
    if not best and fragments:
        best = fragments[0]

    return RetextureSuggestion(
        candidate=candidate,
        fragments=fragments,
        best=best,
    )


def retexture_chapter(
    text: str,
    report: dict,
    n_variants: int = 5,
    max_candidates: int = 5,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
) -> list[RetextureSuggestion]:
    """Full retexturing pipeline: identify flat passages, generate fragments."""
    candidates = identify_candidates(report, text, max_candidates=max_candidates)

    if not candidates:
        return []

    print(f"  Found {len(candidates)} retexture candidates", file=sys.stderr)
    suggestions = []
    for i, cand in enumerate(candidates):
        print(f"  Generating fragments for candidate {i+1}/{len(candidates)}...",
              file=sys.stderr, flush=True)
        suggestion = generate_fragments(
            cand, n_variants=n_variants,
            endpoint=endpoint, model=model,
        )
        if suggestion.fragments:
            suggestions.append(suggestion)

    return suggestions
