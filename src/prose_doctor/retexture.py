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


# Default system prompt for the sensory consultant
SENSORY_SYSTEM_PROMPT = """\
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
- The situation is provided by the user. Write ONLY the body report."""

DEFAULT_ENDPOINT = "http://localhost:8081/v1"
DEFAULT_MODEL = "bartowski-drummer-24b-q6k_l"


@dataclass
class RetextureCandidate:
    """A passage identified for retexturing."""
    paragraph_idx: int
    text: str
    reason: str  # "flatline", "low_surprisal", "low_texture"
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

    # Info contour flatlines — stretches of uniform surprisal
    ic = report.get("info_contour") or {}
    flatlines = ic.get("flatlines", [])
    if isinstance(flatlines, int):
        flatlines = []  # just a count, no detail

    for fl in flatlines:
        if isinstance(fl, dict):
            start = fl.get("start", 0)
            end = fl.get("end", start)
            # Map sentence indices to paragraph indices (rough: ~3 sentences per para)
            para_start = min(start // 3, len(paragraphs) - 1)
            para_end = min(end // 3, len(paragraphs) - 1)
            mid = (para_start + para_end) // 2
            if mid < len(paragraphs):
                candidates.append(RetextureCandidate(
                    paragraph_idx=mid,
                    text=paragraphs[mid],
                    reason="flatline",
                    score=fl.get("mean_surprisal", 0),
                ))

    # Psychic distance low points — most external/distant paragraphs
    pd = report.get("psychic_distance") or {}
    para_means = pd.get("paragraph_means", [])
    if para_means:
        # Find the paragraphs with lowest psychic distance (most external)
        indexed = [(i, d) for i, d in enumerate(para_means) if i < len(paragraphs)]
        indexed.sort(key=lambda x: x[1])
        for idx, dist in indexed[:3]:
            if dist < 0.25:  # very external
                candidates.append(RetextureCandidate(
                    paragraph_idx=idx,
                    text=paragraphs[idx],
                    reason="low_distance",
                    score=dist,
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

    # Build the user prompt from the passage context
    user_prompt = (
        f"A character is in this scene:\n\n"
        f'"{candidate.text[:500]}"\n\n'
        f"Write a 2-3 sentence body report for what the character's "
        f"body is experiencing in this moment. Be specific to the "
        f"physical situation described."
    )

    fragments = []
    for temp in temperatures[:n_variants]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SENSORY_SYSTEM_PROMPT},
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
