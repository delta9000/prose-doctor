"""Blind pairwise judge for Prose Arena head-to-head revision comparisons."""
from __future__ import annotations

import difflib
import json
import random
import re
from typing import Optional

import httpx

_SYSTEM_MESSAGE = """\
You are a senior literary editor specializing in fiction prose craft. You are \
comparing how two different editorial strategies revised the same passages. \
Each pair shows the original passage and two alternative revisions. Your job: \
determine which set of revisions produces better prose overall."""

_PROMPT_TEMPLATE = """\
Below are the passages that differ between two revisions of the same chapter. \
For each changed passage, you see the ORIGINAL, then how VERSION X and \
VERSION Y each revised it. Everything not shown is identical between versions.

Your task: which revision strategy produced better prose overall?

Evaluate each change on:
1. SENTENCE STRUCTURE — varied openings, lengths, rhythms? Or mechanical?
2. INTERIORITY — access to character's inner life? Or surface-level?
3. DISCOURSE FLOW — logical connectives (causal, contrastive, temporal)? Or flat?
4. COMPLETENESS — is the revision a full, grammatical passage? Any truncation or corruption?

{diff_blocks}

---

Considering ALL the changes above, which version's revisions are stronger \
overall? If genuinely equal, say tie.

Respond with ONLY this JSON (no other text):
{{"winner": "X" or "Y" or "tie", "reason": "2-3 sentences citing specific passages"}}"""


def _extract_diff_blocks(
    original: str,
    revised_a: str,
    revised_b: str,
    context_paragraphs: int = 1,
) -> list[dict]:
    """Extract passages that differ, with surrounding context.

    Returns list of {original, revised_a, revised_b, context_before, context_after}.
    """
    orig_paras = original.split("\n\n")
    rev_a_paras = revised_a.split("\n\n")
    rev_b_paras = revised_b.split("\n\n")

    # Find paragraphs that changed in either revision
    changed_indices = set()
    for i, orig_p in enumerate(orig_paras):
        a_p = rev_a_paras[i] if i < len(rev_a_paras) else ""
        b_p = rev_b_paras[i] if i < len(rev_b_paras) else ""
        if orig_p != a_p or orig_p != b_p:
            changed_indices.add(i)

    blocks = []
    for i in sorted(changed_indices):
        ctx_before = orig_paras[i - 1][:200] if i > 0 else ""
        ctx_after = orig_paras[i + 1][:200] if i + 1 < len(orig_paras) else ""
        blocks.append({
            "index": i,
            "original": orig_paras[i],
            "revised_a": rev_a_paras[i] if i < len(rev_a_paras) else "(missing)",
            "revised_b": rev_b_paras[i] if i < len(rev_b_paras) else "(missing)",
            "context_before": ctx_before,
            "context_after": ctx_after,
        })

    return blocks


def build_judge_prompt(
    original: str,
    revised_a: str,
    revised_b: str,
    rng: Optional[random.Random] = None,
) -> tuple[str, dict[str, str]]:
    """Build a diff-focused judge prompt with randomised position assignment.

    Instead of sending full chapters, extracts only the passages that differ
    between the two revisions, with ±1 paragraph of context.
    """
    if rng is None:
        rng = random.Random()

    # Randomize position assignment
    if rng.random() < 0.5:
        a_label, b_label = "X", "Y"
        mapping: dict[str, str] = {"X": "a", "Y": "b"}
    else:
        a_label, b_label = "Y", "X"
        mapping = {"X": "b", "Y": "a"}

    blocks = _extract_diff_blocks(original, revised_a, revised_b)

    if not blocks:
        # No differences found — will be a tie
        diff_text = "(No differences found between the two versions.)"
    else:
        parts = []
        for n, block in enumerate(blocks[:10], 1):  # cap at 10 diffs
            part = f"### Change {n}"
            if block["context_before"]:
                part += f"\n[context]: {block['context_before'][:150]}..."
            part += f"\n\nORIGINAL:\n{block['original'][:500]}"
            # Map a/b to X/Y based on randomization
            rev_x = block[f"revised_{'a' if a_label == 'X' else 'b'}"]
            rev_y = block[f"revised_{'a' if a_label == 'Y' else 'b'}"]
            part += f"\n\nVERSION X:\n{rev_x[:500]}"
            part += f"\n\nVERSION Y:\n{rev_y[:500]}"
            if block["context_after"]:
                part += f"\n\n[context]: ...{block['context_after'][:150]}"
            parts.append(part)

        diff_text = "\n\n---\n\n".join(parts)
        diff_text += f"\n\n({len(blocks)} total changes, {len(blocks) - min(len(blocks), 10)} not shown)" if len(blocks) > 10 else ""

    prompt = _PROMPT_TEMPLATE.format(diff_blocks=diff_text)
    return prompt, mapping


def parse_judge_response(response: str, mapping: dict[str, str]) -> dict:
    """Parse the raw LLM response into a normalised result dict."""
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    cleaned = re.sub(r"```[^\n]*\n?", "", cleaned)
    cleaned = cleaned.strip()

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
    """Call the LLM judge asynchronously and return a normalised verdict."""
    prompt, position_map = build_judge_prompt(original, revised_a, revised_b, rng)
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

    internal_winner = result["winner"]
    if internal_winner in name_map:
        resolved_winner = name_map[internal_winner]
    else:
        resolved_winner = internal_winner

    return {
        "winner": resolved_winner,
        "reason": result["reason"],
        "position_map": position_map,
    }
