"""Blind pairwise judge for Prose Arena head-to-head revision comparisons."""
from __future__ import annotations

import json
import random
import re
from typing import Optional

import httpx

_MAX_CHARS = 8000

_SYSTEM_MESSAGE = """\
You are a senior literary editor specializing in fiction prose craft. You are \
comparing two revised versions of the same AI-generated chapter. Both versions \
were revised by the same LLM under different editorial guidance. Your job is to \
determine which editorial guidance produced a better revision.

The differences may be subtle. Read closely. Compare sentence by sentence if needed."""

_PROMPT_TEMPLATE = """\
Below is an AI-generated fiction chapter (the original), followed by two \
revised versions. Both revisions aimed to improve the prose craft. Your task: \
determine which revision is the stronger piece of fiction writing.

Pay attention to these craft dimensions, in priority order:
1. SENTENCE STRUCTURE — Does the revision vary sentence openings, lengths, \
and rhythms? Are there inversions, fragments used for impact, and varied \
clause structures? Or is it monotonous subject-verb-object?
2. INTERIORITY — Does the revision give access to the character's inner \
life? Thoughts, sensations, doubts, memories? Or does it stay on the surface?
3. DISCOURSE FLOW — Are sentences connected with varied logical relations \
(causal, contrastive, temporal)? Or is everything joined by "and" or left \
implicit?
4. CONCRETENESS BALANCE — Does the revision mix concrete sensory detail \
with moments of abstraction, reflection, or interpretation? Or is it \
relentlessly one mode?
5. SCENE DYNAMISM — Are there shifts in time, space, or focus? Or does the \
scene feel static?

Do NOT penalize revisions for being similar to the original — focus on which \
revision, as a standalone piece, reads better.

If the differences are genuinely negligible (fewer than 3 changed sentences \
total), call it a tie. Otherwise, pick a winner.

---

ORIGINAL:
{original}

---

VERSION X:
{version_x}

---

VERSION Y:
{version_y}

---

Compare the two revisions on the dimensions above. Then respond with ONLY \
this JSON (no other text):
{{"winner": "X" or "Y" or "tie", "reason": "2-3 sentences explaining your choice, citing specific passages"}}"""


def build_judge_prompt(
    original: str,
    revised_a: str,
    revised_b: str,
    rng: Optional[random.Random] = None,
) -> tuple[str, dict[str, str]]:
    """Build a blinded judge prompt with randomised position assignment.

    Returns
    -------
    prompt:
        The formatted prompt string.
    mapping:
        Dict mapping position labels to internal names, e.g. ``{"X": "a", "Y": "b"}``.
    """
    if rng is None:
        rng = random.Random()

    original = original[:_MAX_CHARS]
    revised_a = revised_a[:_MAX_CHARS]
    revised_b = revised_b[:_MAX_CHARS]

    if rng.random() < 0.5:
        version_x, version_y = revised_a, revised_b
        mapping: dict[str, str] = {"X": "a", "Y": "b"}
    else:
        version_x, version_y = revised_b, revised_a
        mapping = {"X": "b", "Y": "a"}

    prompt = _PROMPT_TEMPLATE.format(
        original=original,
        version_x=version_x,
        version_y=version_y,
    )
    return prompt, mapping


def parse_judge_response(response: str, mapping: dict[str, str]) -> dict:
    """Parse the raw LLM response into a normalised result dict.

    Strips ``<think>`` tags and markdown fences, then locates the JSON blob.
    Maps the position-based winner (X/Y) back to the caller-supplied names via
    *mapping*.  Falls back to ``{"winner": "tie", ...}`` on any parse failure.
    """
    # Strip <think>...</think> blocks (including multi-line)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

    # Strip markdown code fences
    cleaned = re.sub(r"```[^\n]*\n?", "", cleaned)
    cleaned = cleaned.strip()

    # Find the first JSON-like object
    match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
    if not match:
        return {"winner": "tie", "reason": "parse failure: no JSON found"}

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {"winner": "tie", "reason": "parse failure: invalid JSON"}

    winner_raw = data.get("winner", "tie")
    reason = data.get("reason", "")

    if winner_raw in ("X", "Y"):
        winner = mapping.get(winner_raw, "tie")
    elif winner_raw == "tie":
        winner = "tie"
    else:
        winner = "tie"

    return {"winner": winner, "reason": reason}


async def judge_pair(
    original: str,
    revised_a: str,
    revised_b: str,
    config_a_name: str,
    config_b_name: str,
    endpoint: str,
    model: str,
    rng: Optional[random.Random] = None,
) -> dict:
    """Call the LLM judge asynchronously and return a normalised verdict.

    Parameters
    ----------
    original:
        The source passage before revision.
    revised_a, revised_b:
        The two candidate revisions to compare.
    config_a_name, config_b_name:
        Human-readable names for each revision config (returned in results).
    endpoint:
        OpenAI-compatible chat completions URL.
    model:
        Model identifier to pass to the endpoint.
    rng:
        Optional seeded ``random.Random`` for deterministic position assignment.

    Returns
    -------
    dict with keys ``winner`` (config name or ``"tie"``), ``reason``, and
    ``position_map`` (the X/Y -> a/b assignment used).
    """
    prompt, position_map = build_judge_prompt(original, revised_a, revised_b, rng)

    # Build the name lookup so we can resolve internal a/b to config names
    name_map = {"a": config_a_name, "b": config_b_name}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        url = endpoint.rstrip("/") + "/chat/completions"
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()

    raw_content = body["choices"][0]["message"]["content"]
    result = parse_judge_response(raw_content, position_map)

    # Resolve internal a/b winner to config names
    internal_winner = result["winner"]
    if internal_winner in name_map:
        resolved_winner = name_map[internal_winner]
    else:
        resolved_winner = internal_winner  # "tie" or unexpected value

    return {
        "winner": resolved_winner,
        "reason": result["reason"],
        "position_map": position_map,
    }
