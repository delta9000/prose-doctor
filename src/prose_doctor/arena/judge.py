"""Blind pairwise judge for Prose Arena head-to-head revision comparisons."""
from __future__ import annotations

import json
import random
import re
from typing import Optional

import httpx

_MAX_CHARS = 3000

_PROMPT_TEMPLATE = """\
Read the original, then both revisions. Which revision is better prose?

Original:
{original}

Version X:
{version_x}

Version Y:
{version_y}

Respond in JSON: {{"winner": "X" | "Y" | "tie", "reason": "..."}}"""

_SYSTEM_MESSAGE = "You are a literary editor comparing two revisions of a passage of prose. Evaluate which revision improves the writing quality, style, and clarity."


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
        resp = await client.post(endpoint, json=payload)
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
