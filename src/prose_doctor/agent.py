"""Pydantic-AI revision agent for iterative prose improvement."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from prose_doctor.agent_models import (
    BASELINES,
    EditResult,
    ProseMetrics,
    RevisionResult,
)
from prose_doctor.text import split_paragraphs

DEFAULT_ENDPOINT = "http://localhost:8081/v1"
DEFAULT_MODEL = "gpt-oss-120b"
MAX_TURNS = 8
METRIC_REGRESSION_LIMIT = 0.20  # reject if any single metric regresses > 20% of its distance


@dataclass
class RevisionContext:
    """Mutable state carried through the agent run."""
    current_text: str
    filename: str = "chapter.md"
    last_metrics: ProseMetrics | None = None
    last_report: dict | None = None
    initial_metrics: ProseMetrics | None = None
    edits_accepted: int = 0
    edits_rejected: int = 0
    turn: int = 0
    max_turns: int = MAX_TURNS
    verbose: bool = False


SYSTEM_PROMPT = """\
You are a prose revision specialist. Your job is to improve a chapter's prose \
quality by making targeted edits, one metric at a time.

## Workflow

1. Call `scan_deep` to measure the current text against human prose baselines.
2. Call `critique` to see prioritized prescriptions for the worst metrics.
3. Call `read_passage` to find the specific passage you want to improve.
4. Rewrite ONLY that passage — keep plot, characters, and dialogue intact.
5. Call `replace_passage` with the old and new text. It will tell you if the \
edit was accepted (metric improved) or rejected (regression detected, auto-reverted).
6. Repeat from step 1, targeting the next worst metric.

## Rules

- Fix ONE metric per edit. Do not try to fix everything at once.
- Preserve the story content. Change prose technique, not plot.
- Keep edits surgical — replace a paragraph or a few sentences, not the whole chapter.
- If an edit is rejected, try a different approach to the same passage, or move on.
- Stop when you've made 6-8 accepted edits or the critique shows no major issues.
- When you're done, return a brief summary of what you changed.
"""


def _do_scan(ctx: RevisionContext) -> tuple[ProseMetrics, dict]:
    """Run scan_deep — extracted for testability."""
    from prose_doctor.agent_scan import scan_deep
    return scan_deep(ctx.current_text, filename=ctx.filename)


def _do_replace(
    ctx: RevisionContext,
    old_text: str,
    new_text: str,
) -> EditResult:
    """Execute a passage replacement with validation and rollback."""
    if ctx.turn >= ctx.max_turns:
        return EditResult(
            accepted=False,
            reason=f"Turn limit reached ({ctx.max_turns}). No more edits allowed.",
            metrics_before=ctx.last_metrics or ProseMetrics(
                pd_mean=0, pd_std=0, fg_inversion=0, fg_sl_cv=0,
                fg_fragment=0, ic_rhythmicity=0, ic_spikes=0, ic_flatlines=0,
            ),
        )

    if old_text not in ctx.current_text:
        ctx.edits_rejected += 1
        return EditResult(
            accepted=False,
            reason=f"Old text not found in current chapter. Check exact wording.",
            metrics_before=ctx.last_metrics or ProseMetrics(
                pd_mean=0, pd_std=0, fg_inversion=0, fg_sl_cv=0,
                fg_fragment=0, ic_rhythmicity=0, ic_spikes=0, ic_flatlines=0,
            ),
        )

    before_metrics = ctx.last_metrics
    previous_text = ctx.current_text
    ctx.current_text = ctx.current_text.replace(old_text, new_text, 1)
    ctx.turn += 1

    after_metrics, after_report = _do_scan(ctx)

    # Per-metric delta analysis
    improved = []
    worsened = []
    deltas: dict[str, float] = {}
    if before_metrics:
        before_d = before_metrics.distances()
        after_d = after_metrics.distances()
        for k in BASELINES:
            delta = after_d[k] - before_d[k]
            deltas[k] = round(delta, 4)
            if after_d[k] < before_d[k]:
                improved.append(k)
            elif after_d[k] > before_d[k]:
                worsened.append(k)

    # Verbose per-metric logging
    if ctx.verbose and before_metrics:
        import sys as _sys
        print(f"  [delta] per-metric changes:", file=_sys.stderr)
        for k, d in sorted(deltas.items(), key=lambda x: x[1]):
            marker = "+" if d > 0 else ("-" if d < 0 else " ")
            print(f"    {marker} {k}: {d:+.4f}", file=_sys.stderr)

    # Check for regression: reject if total_distance didn't decrease
    if before_metrics and after_metrics.total_distance >= before_metrics.total_distance:
        ctx.current_text = previous_text
        ctx.edits_rejected += 1
        return EditResult(
            accepted=False,
            reason=(
                f"Edit rejected: total_distance went from "
                f"{before_metrics.total_distance:.4f} to "
                f"{after_metrics.total_distance:.4f} (no improvement). Reverted."
            ),
            metrics_before=before_metrics,
            metrics_after=after_metrics,
        )

    # Check for individual metric regression > threshold
    if before_metrics:
        before_d = before_metrics.distances()
        after_d = after_metrics.distances()
        for k in BASELINES:
            regression = after_d[k] - before_d[k]
            if regression > METRIC_REGRESSION_LIMIT:
                ctx.current_text = previous_text
                ctx.edits_rejected += 1
                return EditResult(
                    accepted=False,
                    reason=(
                        f"Edit rejected: {k} regressed by {regression:.4f} "
                        f"(limit: {METRIC_REGRESSION_LIMIT}). "
                        f"Worsened: {', '.join(worsened)}. Reverted."
                    ),
                    metrics_before=before_metrics,
                    metrics_after=after_metrics,
                )

    # Accept
    ctx.last_metrics = after_metrics
    ctx.last_report = after_report
    ctx.edits_accepted += 1

    return EditResult(
        accepted=True,
        reason=f"Edit accepted. Improved: {', '.join(improved) or 'marginal'}. "
               f"Worsened: {', '.join(worsened) or 'none'}. "
               f"total_distance: {before_metrics.total_distance:.4f} → "
               f"{after_metrics.total_distance:.4f}",
        metrics_before=before_metrics,
        metrics_after=after_metrics,
    )


def create_agent(
    endpoint: str = DEFAULT_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
) -> Agent[RevisionContext, str]:
    """Create the revision agent with all tools registered."""
    model = OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=endpoint, api_key="none"),
    )

    agent = Agent(
        model,
        deps_type=RevisionContext,
        system_prompt=SYSTEM_PROMPT,
    )

    @agent.tool
    def scan_deep(ctx: RunContext[RevisionContext]) -> str:
        """Scan the current text with all ML analyzers. Returns metrics vs human baselines."""
        metrics, report = _do_scan(ctx.deps)
        ctx.deps.last_metrics = metrics
        ctx.deps.last_report = report
        if ctx.deps.initial_metrics is None:
            ctx.deps.initial_metrics = metrics

        if ctx.deps.verbose:
            print(f"  [scan] total_distance={metrics.total_distance} worst={metrics.worst_metric}", file=sys.stderr)

        distances = metrics.distances()
        lines = ["## Scan Results\n"]
        for key, dist in sorted(distances.items(), key=lambda x: -x[1]):
            baseline, direction = BASELINES[key]
            value = getattr(metrics, key)
            status = "PROBLEM" if dist > 0.15 else ("ok" if dist > 0 else "GOOD")
            lines.append(f"  {key}: {value} (baseline: {baseline}, {direction}) [{status}]")
        lines.append(f"\n  total_distance: {metrics.total_distance}")
        lines.append(f"  worst: {metrics.worst_metric}")
        return "\n".join(lines)

    @agent.tool
    def critique(ctx: RunContext[RevisionContext]) -> str:
        """Get prioritized prescriptions for the worst metrics. Run scan_deep first."""
        if ctx.deps.last_report is None:
            return "ERROR: Run scan_deep first before calling critique."
        from prose_doctor.critique import build_critique, format_critique_prompt
        sections = build_critique(ctx.deps.last_report)
        return format_critique_prompt(
            ctx.deps.filename, sections,
            word_count=ctx.deps.last_report.get("word_count", 0),
        )

    @agent.tool
    def read_passage(ctx: RunContext[RevisionContext], start_paragraph: int, end_paragraph: int) -> str:
        """Read paragraphs from the current text by index (0-based, inclusive).

        Args:
            start_paragraph: First paragraph index to read (0-based).
            end_paragraph: Last paragraph index to read (0-based, inclusive).
        """
        paragraphs = split_paragraphs(ctx.deps.current_text)
        end = min(end_paragraph + 1, len(paragraphs))
        start = max(0, start_paragraph)
        if start >= len(paragraphs):
            return f"ERROR: Only {len(paragraphs)} paragraphs (0-{len(paragraphs)-1})."
        selected = paragraphs[start:end]
        lines = []
        for i, p in enumerate(selected, start):
            lines.append(f"[{i}] {p}")
        return "\n\n".join(lines)

    @agent.tool
    def replace_passage(ctx: RunContext[RevisionContext], old_text: str, new_text: str) -> str:
        """Replace a passage in the chapter. Auto-validates metrics and reverts if regression.

        Args:
            old_text: The exact text to find and replace. Must match verbatim.
            new_text: The replacement text.
        """
        result = _do_replace(ctx.deps, old_text, new_text)

        if ctx.deps.verbose:
            status = "ACCEPTED" if result.accepted else "REJECTED"
            print(f"  [replace] {status}: {result.reason}", file=sys.stderr)

        return result.reason

    @agent.tool
    def retexture(ctx: RunContext[RevisionContext], paragraph_idx: int) -> str:
        """Generate sensory/texture fragments for a flat paragraph via creative LLM.

        Args:
            paragraph_idx: Index of the paragraph to retexture (0-based).
        """
        if ctx.deps.last_report is None:
            return "ERROR: Run scan_deep first."
        paragraphs = split_paragraphs(ctx.deps.current_text)
        if paragraph_idx >= len(paragraphs):
            return f"ERROR: Only {len(paragraphs)} paragraphs."
        from prose_doctor.retexture import RetextureCandidate, generate_fragments
        candidate = RetextureCandidate(
            paragraph_idx=paragraph_idx,
            text=paragraphs[paragraph_idx],
            reason="agent_request",
            mode="sensory",
            score=0.0,
        )
        suggestion = generate_fragments(candidate, n_variants=3)
        if not suggestion.fragments:
            return "No fragments generated — Cydonia endpoint may be down."
        lines = [f"## Fragments for paragraph [{paragraph_idx}]\n"]
        for i, frag in enumerate(suggestion.fragments, 1):
            lines.append(f"{i}. {frag}\n")
        lines.append(f"\nRecommended: {suggestion.best}")
        return "\n".join(lines)

    return agent


def run_revision(
    text: str,
    filename: str = "chapter.md",
    max_turns: int = MAX_TURNS,
    endpoint: str = DEFAULT_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> RevisionResult:
    """Run the revision agent on a chapter and return the result."""
    agent = create_agent(endpoint=endpoint, model_name=model_name)
    ctx = RevisionContext(
        current_text=text,
        filename=filename,
        max_turns=max_turns,
        verbose=verbose,
    )

    prompt = (
        f"Revise this chapter ({filename}). "
        f"You have {max_turns} turns maximum. "
        f"Start by scanning, then fix the worst metrics one at a time."
    )

    if verbose:
        print(f"Starting revision agent (max {max_turns} turns)...", file=sys.stderr)

    result = agent.run_sync(prompt, deps=ctx)

    initial = ctx.initial_metrics
    final = ctx.last_metrics or initial

    improved = []
    worsened = []
    if initial and final:
        init_d = initial.distances()
        final_d = final.distances()
        for k in BASELINES:
            if final_d[k] < init_d[k]:
                improved.append(k)
            elif final_d[k] > init_d[k]:
                worsened.append(k)

    return RevisionResult(
        final_text=ctx.current_text,
        metrics_initial=initial or final,
        metrics_final=final,
        turns_used=ctx.turn,
        edits_accepted=ctx.edits_accepted,
        edits_rejected=ctx.edits_rejected,
        metrics_improved=improved,
        metrics_worsened=worsened,
    )
