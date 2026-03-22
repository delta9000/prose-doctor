"""Single-pass revision: scan once, feed issues as a todo list, LLM makes targeted edits.

The LLM receives:
- The full chapter as context
- One issue at a time, with the FULL paragraph, surrounding context, and a craft-aware prescription
- A replace_text tool to make exact find/replace edits

Each edit targets a paragraph or sentence within a paragraph. The LLM sees
enough context to make a craft decision, not just a mechanical patch.
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
            "substring copied from the chapter — at least one full sentence, "
            "ideally the whole paragraph. The new_text is your revised version."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find (copy verbatim — at least one full sentence)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Revised text to substitute in (same content, better craft)",
                },
            },
            "required": ["old_text", "new_text"],
        },
    },
}

SYSTEM_PROMPT = """\
You are a prose editor working on a fiction chapter. You'll receive specific \
craft issues found by analysis, one at a time. For each issue, you see the \
full paragraph with context and a diagnosis explaining what's wrong and why.

Your job: call replace_text to fix each issue. Replace at least a full \
sentence — ideally the whole paragraph when multiple sentences in it need work.

Craft principles:
- Varied sentence structure creates rhythm. Monotonous SVO order is flat.
- Interiority (what the character feels, thinks, fears) makes prose live.
- Logical connectives (because, but, although, then) show how ideas relate.
- Concrete detail and abstract reflection should alternate, not flatline.
- Every fragment should earn its place through impact. Weak fragments are clutter.

Preserve plot, characters, and dialogue content. Change technique, not story."""


def _build_issue_prompt(
    metric: str,
    issue: object,
    paragraphs: list[str],
    cfg: CritiqueConfig,
) -> str:
    """Build a rich prompt for one issue with full paragraph context."""
    pi = issue.paragraph_idx
    paragraph = paragraphs[pi] if pi < len(paragraphs) else ""
    ctx_before = paragraphs[pi - 1][:300] if pi > 0 else "(start of chapter)"
    ctx_after = paragraphs[pi + 1][:300] if pi + 1 < len(paragraphs) else "(end of chapter)"

    # Get the config's prescription for this metric (the "why")
    prescription = cfg.prescriptions.get(metric, "")

    # Build a craft-aware diagnosis
    diagnosis = issue.reason

    return (
        f"CONTEXT BEFORE:\n{ctx_before}\n\n"
        f"PARAGRAPH TO REVISE:\n{paragraph}\n\n"
        f"CONTEXT AFTER:\n{ctx_after}\n\n"
        f"DIAGNOSIS: {diagnosis}\n\n"
        f"CRAFT GUIDANCE: {prescription}\n\n"
        f"Call replace_text. Copy the passage you want to change EXACTLY from "
        f"the paragraph above (at least one full sentence), then provide your "
        f"revised version."
    )


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
    """Single-pass revision with multi-turn tool-call edits."""
    from openai import OpenAI
    from prose_doctor.agent_scan import scan_deep
    from prose_doctor.agent_issues import find_issues

    cfg = critique_config or CritiqueConfig()
    client = OpenAI(base_url=endpoint, api_key=api_key or "none")

    # 1. Scan
    if verbose:
        print(f"Single-pass revision: {filename}", file=sys.stderr)

    metrics, report = scan_deep(text, filename=filename, critique_config=cfg)
    initial_metrics = metrics
    report_dict = report

    if verbose:
        print(f"  Initial: distance={metrics.total_distance:.4f} worst={metrics.worst_metric}", file=sys.stderr)

    # 2. Collect issues (one per paragraph, across worst metrics)
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

    # 3. Multi-turn: feed one issue at a time, get tool call, validate, feed result back
    current_text = text
    paragraphs = split_paragraphs(text)
    edits_accepted = 0
    edits_rejected = 0

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Here is the chapter ({len(text.split())} words). "
            f"I'll give you {len(all_issues)} issues to fix, one at a time.\n\n"
            f"---\n{text}\n---"
        )},
    ]

    for issue_num, (metric, issue) in enumerate(all_issues):
        issue_prompt = _build_issue_prompt(metric, issue, paragraphs, cfg)
        messages.append({
            "role": "user",
            "content": f"Issue {issue_num + 1}/{len(all_issues)}:\n\n{issue_prompt}",
        })

        if verbose:
            print(f"  [{issue_num + 1}/{len(all_issues)}] para {issue.paragraph_idx} ({metric})", file=sys.stderr)

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
                print(f"    → LLM error: {e}", file=sys.stderr)
            edits_rejected += 1
            continue

        message = resp.choices[0].message
        messages.append(message)

        tool_calls = message.tool_calls or []
        if not tool_calls:
            edits_rejected += 1
            if verbose:
                print(f"    → no tool call", file=sys.stderr)
            messages.append({"role": "user", "content": "No edit made. Moving on."})
            continue

        tc = tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            edits_rejected += 1
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": "Error: could not parse arguments.",
            })
            continue

        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")

        # Validate
        if not old_text or not new_text or old_text == new_text:
            result_msg = "Skipped: no change."
            edits_rejected += 1
        elif old_text not in current_text:
            result_msg = "Error: old_text not found. Copy it EXACTLY from the chapter."
            edits_rejected += 1
        elif len(new_text.split()) < len(old_text.split()) * 0.5:
            result_msg = f"Rejected: too short ({len(new_text.split())}/{len(old_text.split())} words)."
            edits_rejected += 1
        elif len(old_text.split()) < 4:
            result_msg = "Rejected: old_text too small. Replace at least a full sentence."
            edits_rejected += 1
        elif "\n\n" in old_text or "\n---\n" in old_text or "---" in old_text:
            result_msg = "Rejected: old_text spans multiple paragraphs or sections. Edit one paragraph at a time."
            edits_rejected += 1
        elif len(old_text) > 1500:
            result_msg = "Rejected: old_text too long. Target one paragraph at a time (max ~300 words)."
            edits_rejected += 1
        else:
            current_text = current_text.replace(old_text, new_text, 1)
            edits_accepted += 1
            result_msg = f"Applied. ({len(old_text.split())}→{len(new_text.split())} words)"

        if verbose:
            print(f"    → {result_msg}", file=sys.stderr)

        messages.append({
            "role": "tool", "tool_call_id": tc.id,
            "content": result_msg,
        })

    # 4. Final validation scan
    if verbose:
        print(f"\n  {edits_accepted} applied, {edits_rejected} rejected", file=sys.stderr)

    if edits_accepted > 0:
        final_metrics, _ = scan_deep(
            current_text, filename=filename,
            metrics_only=True, previous_report=report_dict,
            critique_config=cfg,
        )
    else:
        final_metrics = initial_metrics

    if verbose:
        delta = initial_metrics.total_distance - final_metrics.total_distance
        print(f"  Distance: {initial_metrics.total_distance:.4f} → {final_metrics.total_distance:.4f} ({delta:+.4f})", file=sys.stderr)

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
