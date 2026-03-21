"""Orchestrated revision: Python drives the loop, LLM just rewrites passages.

Instead of trusting the LLM to navigate tools and exact-match text,
Python picks the target passage from find_issues, hands the model ONE
passage with ONE prescription, gets back ONE rewrite. No agentic loop,
no tool calling — just focused prose revision.

The loop:
1. scan_deep → get metrics
2. find_issues for worst metric → get specific passages
3. For each fixable passage:
   a. Send the paragraph + prescription to the LLM
   b. Get revised paragraph back
   c. Substitute in the text
   d. Re-scan, accept or rollback
4. Repeat with next worst metric until max_turns or convergence
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from prose_doctor.agent_models import BASELINES, ProseMetrics, RevisionResult
from prose_doctor.agent_scan import scan_deep
from prose_doctor.agent_issues import find_issues, Issue
from prose_doctor.text import split_paragraphs


DEFAULT_ENDPOINT = "http://localhost:8081/v1"
DEFAULT_MODEL = "gpt-oss-120b"
MAX_TURNS = 8
METRIC_REGRESSION_LIMIT = 0.20


REWRITE_SYSTEM = """\
You are a prose revision specialist. You will be given a single paragraph \
from a fiction chapter, along with a specific prescription for how to \
improve it. Rewrite ONLY that paragraph following the prescription.

Rules:
- Output ONLY the rewritten paragraph, nothing else
- Preserve all plot content, character names, and dialogue
- Do not add new information or change what happens
- Do not add commentary, labels, or explanations
- Match the surrounding voice and tense"""


def _call_llm(
    paragraph: str,
    prescription: str,
    context_before: str,
    context_after: str,
    endpoint: str,
    model: str,
    api_key: str = "none",
) -> str | None:
    """Send one paragraph + prescription to LLM, get rewrite back."""
    from openai import OpenAI

    client = OpenAI(base_url=endpoint, api_key=api_key)

    user_prompt = f"""Rewrite this paragraph following the prescription below.

CONTEXT BEFORE:
{context_before[:300] if context_before else "(start of chapter)"}

PARAGRAPH TO REWRITE:
{paragraph}

CONTEXT AFTER:
{context_after[:300] if context_after else "(end of chapter)"}

PRESCRIPTION:
{prescription}

Output ONLY the rewritten paragraph:"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=len(paragraph.split()) * 3 + 100,
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        # Strip think tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        # Strip any markdown wrapping
        if content.startswith("```"):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
            content = content.strip()
        return content if content else None
    except Exception as e:
        print(f"  LLM call failed: {e}", file=sys.stderr)
        return None


def _pick_metrics_to_fix(metrics: ProseMetrics, max_metrics: int = 3) -> list[str]:
    """Pick the worst metrics to fix, in order."""
    distances = metrics.distances()
    # Only fix metrics with meaningful gap
    fixable = [(k, d) for k, d in distances.items() if d > 0.10]
    fixable.sort(key=lambda x: -x[1])
    return [k for k, _ in fixable[:max_metrics]]


def _prescription_for_issue(issue: Issue) -> str:
    """Build a focused prescription from an Issue."""
    return issue.reason


def run_orchestrated(
    text: str,
    filename: str = "chapter.md",
    max_turns: int = MAX_TURNS,
    endpoint: str = DEFAULT_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
    api_key: str = "none",
    verbose: bool = False,
) -> RevisionResult:
    """Run the orchestrated revision loop."""
    current_text = text
    turn = 0
    edits_accepted = 0
    edits_rejected = 0

    if verbose:
        print(f"Orchestrated revision: {filename}, max {max_turns} turns", file=sys.stderr)

    # Initial scan
    metrics, report = scan_deep(current_text, filename=filename)
    initial_metrics = metrics

    if verbose:
        print(f"  Initial: distance={metrics.total_distance} worst={metrics.worst_metric}", file=sys.stderr)

    # Pick metrics to fix
    metric_queue = _pick_metrics_to_fix(metrics)
    if verbose:
        print(f"  Fixing: {metric_queue}", file=sys.stderr)

    # Add generic pass at the end
    metric_queue.append("generic")

    for metric in metric_queue:
        if turn >= max_turns:
            break

        if verbose:
            print(f"\n  --- Targeting: {metric} ---", file=sys.stderr)

        # Get issues for this metric
        issues = find_issues(metric, current_text, report)
        fixable = [i for i in issues if not i.preserve]

        if not fixable:
            if verbose:
                print(f"  No fixable issues for {metric}", file=sys.stderr)
            continue

        if verbose:
            print(f"  Found {len(fixable)} fixable passages", file=sys.stderr)

        for issue in fixable:
            if turn >= max_turns:
                break

            paragraphs = split_paragraphs(current_text)
            pi = issue.paragraph_idx
            if pi >= len(paragraphs):
                continue

            target_para = paragraphs[pi]
            if len(target_para.split()) < 5:
                continue  # too short to meaningfully revise

            # Context
            ctx_before = paragraphs[pi - 1] if pi > 0 else ""
            ctx_after = paragraphs[pi + 1] if pi < len(paragraphs) - 1 else ""

            prescription = _prescription_for_issue(issue)

            if verbose:
                print(f"  [{turn+1}] para {pi}: {prescription[:80]}", file=sys.stderr)

            # Call LLM for rewrite
            revised_para = _call_llm(
                target_para, prescription, ctx_before, ctx_after,
                endpoint, model_name, api_key,
            )

            if not revised_para or revised_para == target_para:
                if verbose:
                    print(f"    → skipped (no change)", file=sys.stderr)
                continue

            # Substitute and validate
            previous_text = current_text
            current_text = current_text.replace(target_para, revised_para, 1)

            if current_text == previous_text:
                if verbose:
                    print(f"    → skipped (substitution failed)", file=sys.stderr)
                continue

            turn += 1

            # Re-scan
            new_metrics, new_report = scan_deep(
                current_text, filename=filename,
                metrics_only=True, previous_report=report,
            )

            # Check for improvement
            if new_metrics.total_distance >= metrics.total_distance:
                current_text = previous_text
                edits_rejected += 1
                if verbose:
                    print(f"    → REJECTED: {metrics.total_distance:.4f} → {new_metrics.total_distance:.4f}", file=sys.stderr)
                continue

            # Check per-metric regression
            old_d = metrics.distances()
            new_d = new_metrics.distances()
            regressed = [k for k in BASELINES if new_d[k] - old_d[k] > METRIC_REGRESSION_LIMIT]
            if regressed:
                current_text = previous_text
                edits_rejected += 1
                if verbose:
                    print(f"    → REJECTED: {', '.join(regressed)} regressed > {METRIC_REGRESSION_LIMIT}", file=sys.stderr)
                continue

            # Accept
            metrics = new_metrics
            report = new_report
            edits_accepted += 1

            improved = [k for k in BASELINES if new_d[k] < old_d[k]]
            if verbose:
                print(f"    → ACCEPTED: {metrics.total_distance:.4f} improved={improved}", file=sys.stderr)

    # Final result
    final_metrics = metrics
    init_d = initial_metrics.distances()
    final_d = final_metrics.distances()
    improved = [k for k in BASELINES if final_d[k] < init_d[k]]
    worsened = [k for k in BASELINES if final_d[k] > init_d[k]]

    return RevisionResult(
        final_text=current_text,
        metrics_initial=initial_metrics,
        metrics_final=final_metrics,
        turns_used=turn,
        edits_accepted=edits_accepted,
        edits_rejected=edits_rejected,
        metrics_improved=improved,
        metrics_worsened=worsened,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrated prose revision")
    parser.add_argument("file", help="Chapter file to revise")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    text = Path(args.file).read_text()
    result = run_orchestrated(
        text,
        filename=Path(args.file).name,
        max_turns=args.max_turns,
        endpoint=args.endpoint,
        model_name=args.model,
        api_key=args.api_key,
        verbose=args.verbose,
    )

    if args.output:
        Path(args.output).write_text(result.final_text)
        print(f"Wrote to {args.output}", file=sys.stderr)
    else:
        print(result.final_text)

    print(f"\n--- Summary ---", file=sys.stderr)
    print(f"Turns: {result.turns_used} | Accepted: {result.edits_accepted} | Rejected: {result.edits_rejected}", file=sys.stderr)
    print(f"Distance: {result.metrics_initial.total_distance:.4f} → {result.metrics_final.total_distance:.4f}", file=sys.stderr)
    print(f"Improved: {result.metrics_improved}", file=sys.stderr)
    print(f"Worsened: {result.metrics_worsened}", file=sys.stderr)
