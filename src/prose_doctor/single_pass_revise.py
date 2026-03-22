"""Single-pass revision: scan once, send all issues, LLM calls replace_text tool per edit.

Flow:
1. Scan chapter → find all issues
2. Send chapter + all prescriptions to LLM with a replace_text tool
3. LLM calls replace_text(old_text, new_text) for each fix — one tool call per edit
4. Validate each edit (exact match, truncation check), collect results
5. One final rescan to measure improvement

Uses OpenAI function calling — the LLM produces structured tool calls,
not free-text JSON we have to parse.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from prose_doctor.agent_models import ProseMetrics, RevisionResult
from prose_doctor.critique_config import CritiqueConfig
from prose_doctor.text import split_paragraphs


DEFAULT_ENDPOINT = "http://localhost:8081/v1"
DEFAULT_MODEL = "gpt-oss-120b"

REPLACE_TOOL = {
    "type": "function",
    "function": {
        "name": "replace_text",
        "description": (
            "Replace a passage in the chapter. The old_text must be an EXACT "
            "substring copied from the chapter. The new_text is your revised "
            "version. Keep the replacement roughly the same length — do not "
            "delete large sections or add new scenes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace (copy from chapter verbatim)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Revised text to substitute in",
                },
            },
            "required": ["old_text", "new_text"],
        },
    },
}

SYSTEM_PROMPT = """\
You are a prose revision specialist. You will be given a fiction chapter and \
a list of specific issues found by automated analysis. Fix each issue by \
calling the replace_text tool with the exact passage to change and your \
revised version.

Rules:
- Call replace_text once per issue. One edit per tool call.
- The old_text must be EXACTLY copied from the chapter — even one wrong character will fail.
- Keep the replacement roughly the same length as the original.
- Preserve all plot content, character names, and dialogue.
- Do not add commentary or explanations — just make the tool calls.
- Match the surrounding voice and tense.
- If an issue doesn't need fixing or you can't improve it, skip it."""


def _build_issue_list(issues: list[tuple[str, object]]) -> str:
    """Format issues into a numbered list for the LLM."""
    lines = []
    for i, (metric, issue) in enumerate(issues, 1):
        lines.append(
            f"{i}. [paragraph {issue.paragraph_idx}] ({metric}) {issue.reason}\n"
            f"   Text: \"{issue.sentence_text[:150]}\""
        )
    return "\n\n".join(lines)


def run_single_pass(
    text: str,
    filename: str = "chapter.md",
    endpoint: str = DEFAULT_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
    api_key: str = "none",
    verbose: bool = False,
    critique_config: CritiqueConfig | None = None,
    max_edits: int = 10,
) -> RevisionResult:
    """Single-pass revision with tool-call edits."""
    from openai import OpenAI
    from prose_doctor.agent_scan import scan_deep
    from prose_doctor.agent_issues import find_issues

    cfg = critique_config or CritiqueConfig()
    client = OpenAI(base_url=endpoint, api_key=api_key or "none")

    # 1. Initial scan
    if verbose:
        print(f"Single-pass revision: {filename}", file=sys.stderr)

    metrics, report = scan_deep(text, filename=filename, critique_config=cfg)
    initial_metrics = metrics
    report_dict = report

    if verbose:
        print(f"  Initial: distance={metrics.total_distance:.4f} worst={metrics.worst_metric}", file=sys.stderr)

    # 2. Collect all issues
    distances = metrics.distances()
    fixable = [(k, d) for k, d in distances.items() if d > 0.10]
    fixable.sort(key=lambda x: -x[1])
    metric_queue = [k for k, _ in fixable[:5]]

    all_issues = []
    seen_paragraphs = set()
    for metric in metric_queue:
        issues = find_issues(metric, text, report_dict, config=cfg)
        for issue in issues:
            if issue.preserve:
                continue
            if issue.paragraph_idx in seen_paragraphs:
                continue
            seen_paragraphs.add(issue.paragraph_idx)
            all_issues.append((metric, issue))

    all_issues = all_issues[:max_edits]

    if verbose:
        print(f"  Found {len(all_issues)} issues across {len(metric_queue)} metrics", file=sys.stderr)

    if not all_issues:
        return RevisionResult(
            final_text=text,
            metrics_initial=initial_metrics,
            metrics_final=initial_metrics,
            turns_used=0,
            edits_accepted=0,
            edits_rejected=0,
            metrics_improved=[],
            metrics_worsened=[],
        )

    # 3. Multi-turn tool-call loop: feed issues one at a time
    current_text = text
    edits_accepted = 0
    edits_rejected = 0
    remaining_issues = list(all_issues)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Here is the chapter to revise:\n\n---\n{text}\n---\n\n"
            f"I will give you issues to fix one at a time. "
            f"For each issue, call replace_text with the exact passage to change "
            f"and your revised version.\n\n"
            f"There are {len(remaining_issues)} issues total."
        )},
    ]

    for issue_num, (metric, issue) in enumerate(remaining_issues):
        # Feed the next issue
        issue_prompt = (
            f"Issue {issue_num + 1}/{len(remaining_issues)}: "
            f"[paragraph {issue.paragraph_idx}] ({metric}) {issue.reason}\n"
            f"Text: \"{issue.sentence_text[:200]}\"\n\n"
            f"Call replace_text to fix this. The old_text must be copied EXACTLY "
            f"from the chapter."
        )
        messages.append({"role": "user", "content": issue_prompt})

        if verbose:
            print(f"  [{issue_num + 1}/{len(remaining_issues)}] para {issue.paragraph_idx} ({metric}): {issue.reason[:60]}", file=sys.stderr)

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=[REPLACE_TOOL],
                tool_choice={"type": "function", "function": {"name": "replace_text"}},
                max_tokens=2000,
                temperature=cfg.temperature,
            )
        except Exception as e:
            if verbose:
                print(f"    → LLM call failed: {e}", file=sys.stderr)
            edits_rejected += 1
            continue

        message = resp.choices[0].message
        messages.append(message)  # add assistant response to history

        tool_calls = message.tool_calls or []
        if not tool_calls:
            edits_rejected += 1
            if verbose:
                print(f"    → skipped (no tool call)", file=sys.stderr)
            messages.append({"role": "user", "content": "No tool call received. Moving to next issue."})
            continue

        tc = tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            edits_rejected += 1
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": "Error: invalid JSON in tool call arguments."
            })
            continue

        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")

        # Validate
        if not old_text or not new_text or old_text == new_text:
            edits_rejected += 1
            result_msg = "Skipped: empty or no change."
        elif old_text not in current_text:
            edits_rejected += 1
            result_msg = f"Error: old_text not found in chapter. Must be an EXACT copy."
        elif len(new_text.split()) < len(old_text.split()) * 0.5:
            edits_rejected += 1
            result_msg = f"Rejected: replacement too short ({len(new_text.split())}/{len(old_text.split())} words). Do not truncate."
        else:
            current_text = current_text.replace(old_text, new_text, 1)
            edits_accepted += 1
            result_msg = f"Edit applied. ({len(old_text.split())}→{len(new_text.split())} words)"

        if verbose:
            print(f"    → {result_msg}", file=sys.stderr)

        # Feed result back to LLM
        messages.append({
            "role": "tool", "tool_call_id": tc.id,
            "content": result_msg,
        })

    if verbose:
        print(f"  Applied {edits_accepted} edits, rejected {edits_rejected}", file=sys.stderr)

    # 5. Final validation scan
    if edits_accepted > 0:
        final_metrics, _ = scan_deep(
            current_text, filename=filename,
            metrics_only=True, previous_report=report_dict,
            critique_config=cfg,
        )
    else:
        final_metrics = initial_metrics

    if verbose:
        print(f"  Final: distance={final_metrics.total_distance:.4f}", file=sys.stderr)
        delta = initial_metrics.total_distance - final_metrics.total_distance
        print(f"  Delta: {delta:+.4f}", file=sys.stderr)

    init_d = initial_metrics.distances()
    final_d = final_metrics.distances()
    improved = [k for k in cfg.baselines if final_d.get(k, 0) < init_d.get(k, 0)]
    worsened = [k for k in cfg.baselines if final_d.get(k, 0) > init_d.get(k, 0)]

    return RevisionResult(
        final_text=current_text,
        metrics_initial=initial_metrics,
        metrics_final=final_metrics,
        turns_used=edits_accepted + edits_rejected,
        edits_accepted=edits_accepted,
        edits_rejected=edits_rejected,
        metrics_improved=improved,
        metrics_worsened=worsened,
    )
