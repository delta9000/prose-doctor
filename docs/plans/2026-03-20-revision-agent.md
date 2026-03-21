# Revision Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pydantic-ai agent that iteratively revises prose chapters by scanning metrics, identifying the worst deviation from human baselines, rewriting targeted passages, and validating improvement — one metric per turn until convergence.

**Architecture:** A single `Agent` with `RevisionContext` deps carrying mutable state (current text, scan history). Five tools: `scan_deep`, `critique`, `read_passage`, `replace_passage`, `retexture`. The agent drives its own revision loop via tool calls. Python-side rollback enforced in `replace_passage`. CLI exposed as `prose-doctor revise`.

**Tech Stack:** pydantic-ai 1.60+, pydantic v2, OpenAI-compatible local endpoint (gpt-oss-120b on localhost:8081), prose-doctor ML analyzers.

---

### Task 1: Add pydantic-ai dependency

**Files:**
- Modify: `/home/ben/code/prose-doctor/pyproject.toml`

- [ ] **Step 1: Add pydantic-ai to dependencies**

Add `pydantic-ai` to the `[project.optional-dependencies]` section under a new `agent` extra, since it doesn't need torch/spacy:

```toml
[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "transformers>=4.40",
    "sentence-transformers>=3.0",
    "spacy>=3.7",
    "numpy>=1.26",
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
]
agent = [
    "pydantic-ai>=1.60.0",
    "prose-doctor[ml]",
]
```

- [ ] **Step 2: Install and verify**

Run: `cd /home/ben/code/prose-doctor && uv pip install -e ".[agent]"`
Expected: Clean install, pydantic-ai available

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add pydantic-ai agent extra"
```

---

### Task 2: Pydantic models for revision agent

**Files:**
- Create: `/home/ben/code/prose-doctor/src/prose_doctor/agent_models.py`
- Test: `/home/ben/code/prose-doctor/tests/test_agent_models.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_agent_models.py
from prose_doctor.agent_models import ProseMetrics, EditResult, RevisionResult, BASELINES


def test_prose_metrics_total_distance():
    """Total distance sums normalized gaps to baselines."""
    # All at baseline => distance 0
    m = ProseMetrics(
        pd_mean=0.336, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=7, ic_flatlines=3,
    )
    assert m.total_distance == 0.0


def test_prose_metrics_worst_metric():
    """worst_metric returns the metric farthest from baseline."""
    m = ProseMetrics(
        pd_mean=0.1, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=7, ic_flatlines=3,
    )
    assert m.worst_metric == "pd_mean"


def test_edit_result_serialization():
    before = ProseMetrics(
        pd_mean=0.2, pd_std=0.05, fg_inversion=30.0,
        fg_sl_cv=0.5, fg_fragment=10.0, ic_rhythmicity=0.2,
        ic_spikes=3, ic_flatlines=5,
    )
    er = EditResult(
        accepted=True, reason="improved pd_mean by 0.05",
        metrics_before=before, metrics_after=before,
    )
    d = er.model_dump()
    assert d["accepted"] is True
    assert "metrics_before" in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prose_doctor.agent_models'`

- [ ] **Step 3: Write the models**

```python
# src/prose_doctor/agent_models.py
"""Pydantic models for the revision agent."""
from __future__ import annotations

from pydantic import BaseModel, computed_field

# Human prose baselines — same as critique.py but structured for distance calc
BASELINES: dict[str, tuple[float, str]] = {
    "pd_mean":       (0.336, "higher"),
    "pd_std":        (0.093, "higher"),
    "fg_inversion":  (44.2,  "higher"),
    "fg_sl_cv":      (0.706, "higher"),
    "fg_fragment":   (6.7,   "lower"),
    "ic_rhythmicity":(0.129, "lower"),
    "ic_spikes":     (7.7,   "higher"),
    "ic_flatlines":  (3.1,   "lower"),
}


class ProseMetrics(BaseModel):
    """Structured metrics from a deep scan."""
    pd_mean: float
    pd_std: float
    fg_inversion: float
    fg_sl_cv: float
    fg_fragment: float
    ic_rhythmicity: float
    ic_spikes: int
    ic_flatlines: int

    def _metric_distance(self, key: str) -> float:
        """Normalized distance from baseline for one metric. 0 = at/past baseline."""
        baseline, direction = BASELINES[key]
        value = getattr(self, key)
        if baseline == 0:
            return 0.0
        if direction == "higher":
            gap = (baseline - value) / baseline
        else:
            gap = (value - baseline) / baseline
        return max(0.0, gap)

    @computed_field
    @property
    def total_distance(self) -> float:
        return round(sum(self._metric_distance(k) for k in BASELINES), 4)

    @computed_field
    @property
    def worst_metric(self) -> str:
        return max(BASELINES, key=lambda k: self._metric_distance(k))

    def distances(self) -> dict[str, float]:
        return {k: round(self._metric_distance(k), 4) for k in BASELINES}


class EditResult(BaseModel):
    """Result of a replace_passage call."""
    accepted: bool
    reason: str
    metrics_before: ProseMetrics
    metrics_after: ProseMetrics | None = None


class RevisionResult(BaseModel):
    """Final output of the revision agent."""
    final_text: str
    metrics_initial: ProseMetrics
    metrics_final: ProseMetrics
    turns_used: int
    edits_accepted: int
    edits_rejected: int
    metrics_improved: list[str]
    metrics_worsened: list[str]
```

- [ ] **Step 4: Run tests**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/agent_models.py tests/test_agent_models.py
git commit -m "feat: add pydantic models for revision agent"
```

---

### Task 3: Deep scan helper function

**Files:**
- Create: `/home/ben/code/prose-doctor/src/prose_doctor/agent_scan.py`
- Test: `/home/ben/code/prose-doctor/tests/test_agent_scan.py`

This extracts the scan-deep logic from `cli.py` into a reusable function that returns `ProseMetrics` + the full report dict. The CLI currently does this inline — we need it callable from the agent.

- [ ] **Step 1: Write the test**

```python
# tests/test_agent_scan.py
import pytest
from prose_doctor.agent_scan import scan_deep
from prose_doctor.agent_models import ProseMetrics


SAMPLE_TEXT = """
The rain hammered the corrugated roof like a thousand tiny fists. Marcus
pressed his back against the wall, feeling the cold seep through his jacket.
His breath came in short, ragged bursts.

Down the corridor, something scraped. Metal on concrete. He counted to three,
then moved — low, fast, keeping to the shadows. The flashlight in his left
hand was dead weight now, batteries drained hours ago.

She was waiting at the junction, rifle across her knees. "Took you long
enough," she said, not looking up. Her fingers worked the bolt mechanism
with practiced ease, the sound crisp in the wet air.

"They're coming from the east side," he said. "Four, maybe five." He knelt
beside her, pulling the map from his vest pocket. The paper was damp, ink
bleeding at the folds. "We go north through the service tunnel."

"And if it's flooded?"

"Then we swim."
""".strip()


@pytest.mark.slow
def test_scan_deep_returns_prose_metrics():
    metrics, report = scan_deep(SAMPLE_TEXT, filename="test.md")
    assert isinstance(metrics, ProseMetrics)
    assert metrics.pd_mean > 0
    assert metrics.ic_spikes >= 0
    assert "psychic_distance" in report
    assert "info_contour" in report
    assert "foregrounding" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_scan.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement scan_deep**

```python
# src/prose_doctor/agent_scan.py
"""Reusable deep-scan function for the revision agent."""
from __future__ import annotations

from prose_doctor.agent_models import ProseMetrics
from prose_doctor.analyzers.doctor import ChapterHealth, diagnose
from prose_doctor.config import ProjectConfig


def scan_deep(
    text: str,
    filename: str = "chapter.md",
    config: ProjectConfig | None = None,
) -> tuple[ProseMetrics, dict]:
    """Run the full ML analyzer suite and return structured metrics + raw report.

    Returns (ProseMetrics, report_dict) where report_dict is the full
    ChapterHealth.to_dict() for use by critique/retexture.
    """
    from prose_doctor.ml import require_ml
    require_ml()
    from prose_doctor.ml.models import ModelManager

    cfg = config or ProjectConfig()
    mm = ModelManager()

    report = diagnose(text, filename=filename, config=cfg)

    # Psychic distance
    from prose_doctor.ml.psychic_distance import analyze_chapter as pd_analyze
    pd = pd_analyze(text, filename, mm)
    report.psychic_distance = {
        "mean_distance": pd.mean_distance,
        "std_distance": pd.std_distance,
        "label": pd.label,
        "zoom_jumps": len(pd.zoom_jumps),
        "paragraph_means": [round(m, 3) for m in pd.paragraph_means],
    }

    # Information contour
    from prose_doctor.ml.info_contour import analyze_chapter as ic_analyze
    ic = ic_analyze(text, filename, mm)
    report.info_contour = {
        "mean_surprisal": ic.mean_surprisal,
        "cv_surprisal": ic.cv_surprisal,
        "label": ic.label,
        "dominant_period": ic.dominant_period,
        "dominant_period_words": ic.dominant_period_words,
        "rhythmicity": ic.rhythmicity,
        "flatlines": ic.flatlines,
        "spikes": len(ic.spikes),
    }

    # Foregrounding
    from prose_doctor.ml.foregrounding import score_chapter
    fg = score_chapter(text, filename, mm)
    report.foregrounding = {
        "index": round(fg.index, 2),
        "inversion_pct": round(fg.inversion_pct, 1),
        "sentence_length_cv": round(fg.sl_cv, 2),
        "fragment_pct": round(fg.fragment_pct, 1),
        "weakest_axis": fg.weakest_axis,
        "prescription": fg.prescription,
    }

    # Sensory
    from prose_doctor.ml.sensory import profile_chapter
    sp = profile_chapter(text, filename, mm)
    report.sensory = {
        "dominant": sp.dominant_modality,
        "weakest": sp.weakest_modality,
        "balance": round(sp.balance_ratio, 3),
        "scores": sp.scores,
        "deserts": len(sp.deserts),
        "prescription": sp.prescription,
    }

    report_dict = report.to_dict()

    # Build ProseMetrics from report
    ic_dict = report_dict.get("info_contour") or {}
    spikes_val = ic_dict.get("spikes", 0)
    if isinstance(spikes_val, list):
        spikes_val = len(spikes_val)
    flatlines_val = ic_dict.get("flatlines", 0)
    if isinstance(flatlines_val, list):
        flatlines_val = len(flatlines_val)

    fg_dict = report_dict.get("foregrounding") or {}
    pd_dict = report_dict.get("psychic_distance") or {}

    metrics = ProseMetrics(
        pd_mean=pd_dict.get("mean_distance", 0),
        pd_std=pd_dict.get("std_distance", 0),
        fg_inversion=fg_dict.get("inversion_pct", 0),
        fg_sl_cv=fg_dict.get("sentence_length_cv", 0),
        fg_fragment=fg_dict.get("fragment_pct", 0),
        ic_rhythmicity=ic_dict.get("rhythmicity", 0),
        ic_spikes=spikes_val,
        ic_flatlines=flatlines_val,
    )

    return metrics, report_dict
```

- [ ] **Step 4: Run test**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_scan.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/agent_scan.py tests/test_agent_scan.py
git commit -m "feat: extract scan_deep as reusable function for agent"
```

---

### Task 4: Revision agent core

**Files:**
- Create: `/home/ben/code/prose-doctor/src/prose_doctor/agent.py`
- Test: `/home/ben/code/prose-doctor/tests/test_agent.py`

This is the main agent — `RevisionContext` deps, five tools, system prompt, convergence logic.

- [ ] **Step 1: Write the test**

```python
# tests/test_agent.py
"""Tests for the revision agent.

These tests mock the LLM endpoint and scan_deep to test tool logic
without requiring GPU or a running model server.
"""
import pytest
from unittest.mock import patch, MagicMock
from prose_doctor.agent import RevisionContext, _do_replace, _do_scan
from prose_doctor.agent_models import ProseMetrics


def _make_metrics(**overrides) -> ProseMetrics:
    defaults = dict(
        pd_mean=0.336, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=7, ic_flatlines=3,
    )
    defaults.update(overrides)
    return ProseMetrics(**defaults)


def test_replace_passage_accepts_improvement():
    """replace_passage accepts when targeted metric improves."""
    ctx = RevisionContext(
        current_text="The old boring sentence was here.",
        filename="test.md",
    )
    ctx.last_metrics = _make_metrics(pd_mean=0.2)
    improved = _make_metrics(pd_mean=0.3)

    with patch("prose_doctor.agent._do_scan", return_value=(improved, {})):
        result = _do_replace(
            ctx,
            old_text="The old boring sentence was here.",
            new_text="Rain hit the tin roof. She flinched.",
        )
    assert result.accepted
    assert ctx.current_text == "Rain hit the tin roof. She flinched."
    assert ctx.edits_accepted == 1


def test_replace_passage_rejects_regression():
    """replace_passage rejects when total_distance increases."""
    ctx = RevisionContext(
        current_text="The old sentence.",
        filename="test.md",
    )
    ctx.last_metrics = _make_metrics(pd_mean=0.3)
    worse = _make_metrics(pd_mean=0.1)  # regression

    with patch("prose_doctor.agent._do_scan", return_value=(worse, {})):
        result = _do_replace(
            ctx,
            old_text="The old sentence.",
            new_text="Bad rewrite.",
        )
    assert not result.accepted
    assert ctx.current_text == "The old sentence."  # reverted
    assert ctx.edits_rejected == 1


def test_replace_passage_old_text_not_found():
    """replace_passage fails gracefully when old_text doesn't match."""
    ctx = RevisionContext(
        current_text="Actual text in the chapter.",
        filename="test.md",
    )
    ctx.last_metrics = _make_metrics()

    result = _do_replace(ctx, old_text="nonexistent passage", new_text="whatever")
    assert not result.accepted
    assert "not found" in result.reason.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement the agent**

```python
# src/prose_doctor/agent.py
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
REGRESSION_THRESHOLD = 0.20  # reject if any metric regresses > 20%


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

    # Check for regression
    if before_metrics and after_metrics.total_distance >= before_metrics.total_distance:
        # Revert
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

    # Accept
    ctx.last_metrics = after_metrics
    ctx.last_report = after_report
    ctx.edits_accepted += 1

    # Summarize what improved
    improved = []
    if before_metrics:
        before_d = before_metrics.distances()
        after_d = after_metrics.distances()
        for k in BASELINES:
            if after_d[k] < before_d[k]:
                improved.append(k)

    return EditResult(
        accepted=True,
        reason=f"Edit accepted. Improved: {', '.join(improved) or 'marginal'}. "
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

    @agent.tool_plain
    def read_passage(start_paragraph: int, end_paragraph: int) -> str:
        """Read paragraphs from the current text by index (0-based, inclusive)."""
        # Note: we access the agent's deps through closure since tool_plain
        # doesn't get RunContext. We'll switch to @agent.tool if needed.
        return "ERROR: Use the tool version with context."

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
        return result.reason

    @agent.tool
    def retexture(ctx: RunContext[RevisionContext], paragraph_idx: int) -> str:
        """Generate sensory/texture fragments for a flat paragraph via creative LLM.

        Args:
            paragraph_idx: Index of the paragraph to retexture (0-based).
        """
        if ctx.deps.last_report is None:
            return "ERROR: Run scan_deep first."
        from prose_doctor.retexture import identify_candidates, generate_fragments
        paragraphs = split_paragraphs(ctx.deps.current_text)
        if paragraph_idx >= len(paragraphs):
            return f"ERROR: Only {len(paragraphs)} paragraphs."
        from prose_doctor.retexture import RetextureCandidate
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

    # Build final result
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
```

- [ ] **Step 4: Fix duplicate tool name**

The code above has two functions named `read_passage`. Remove the `tool_plain` version — it was a false start. Keep only the `@agent.tool` version with `RunContext`.

- [ ] **Step 5: Run tests**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent.py -v`
Expected: PASS (3 tests — accepts improvement, rejects regression, handles missing text)

- [ ] **Step 6: Commit**

```bash
git add src/prose_doctor/agent.py tests/test_agent.py
git commit -m "feat: revision agent with scan/critique/replace/retexture tools"
```

---

### Task 5: CLI integration

**Files:**
- Modify: `/home/ben/code/prose-doctor/src/prose_doctor/cli.py`

- [ ] **Step 1: Add the `revise` subcommand**

Add after the `critique` parser setup in `main()`:

```python
# revise (agent)
revise_p = subparsers.add_parser("revise", help="Agentic iterative revision [ML+Agent]")
revise_p.add_argument("files", nargs="+", help="Files to revise")
revise_p.add_argument("-o", "--output", type=str, default=None,
                      help="Output file (default: stdout)")
revise_p.add_argument("--max-turns", type=int, default=8,
                      help="Maximum revision turns (default: 8)")
revise_p.add_argument("--dry-run", action="store_true",
                      help="Scan and critique only, don't rewrite")
revise_p.add_argument("--verbose", action="store_true",
                      help="Print each turn's metrics delta")
revise_p.add_argument("--endpoint", type=str, default="http://localhost:8081/v1",
                      help="LLM endpoint")
revise_p.add_argument("--model", type=str, default="gpt-oss-120b",
                      help="Model name")
```

And the handler function:

```python
def cmd_revise(args: argparse.Namespace) -> None:
    """Run the agentic revision loop."""
    try:
        from prose_doctor.ml import require_ml
        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        # Just run scan + critique
        from prose_doctor.agent_scan import scan_deep
        from prose_doctor.critique import build_critique, format_critique_prompt
        for f in files:
            text = f.read_text()
            metrics, report = scan_deep(text, filename=f.name)
            sections = build_critique(report)
            print(format_critique_prompt(f.name, sections, word_count=report.get("word_count", 0)))
            print(f"\nMetrics: {metrics.distances()}")
            print(f"Total distance: {metrics.total_distance}")
            print(f"Worst: {metrics.worst_metric}")
        return

    from prose_doctor.agent import run_revision

    for f in files:
        text = f.read_text()
        result = run_revision(
            text,
            filename=f.name,
            max_turns=args.max_turns,
            endpoint=args.endpoint,
            model_name=args.model,
            verbose=args.verbose,
        )

        # Output
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(result.final_text)
            print(f"Wrote revised chapter to {args.output}", file=sys.stderr)
        else:
            print(result.final_text)

        # Summary to stderr
        print(f"\n--- Revision Summary ---", file=sys.stderr)
        print(f"Turns: {result.turns_used}", file=sys.stderr)
        print(f"Edits accepted: {result.edits_accepted}", file=sys.stderr)
        print(f"Edits rejected: {result.edits_rejected}", file=sys.stderr)
        print(f"Improved: {', '.join(result.metrics_improved) or 'none'}", file=sys.stderr)
        print(f"Worsened: {', '.join(result.metrics_worsened) or 'none'}", file=sys.stderr)
        print(f"Distance: {result.metrics_initial.total_distance:.4f} → {result.metrics_final.total_distance:.4f}", file=sys.stderr)
```

Add `"revise": cmd_revise` to the `handlers` dict.

- [ ] **Step 2: Smoke test — dry run**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor revise --dry-run /home/ben/code/nladfg/story_tracker/prose/book1/ch01.md`
Expected: Prints critique and metrics without calling the LLM

- [ ] **Step 3: Smoke test — live run**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor revise /home/ben/code/nladfg/story_tracker/prose/book1/ch01.md --max-turns 2 --verbose -o /tmp/revised_ch01.md`
Expected: Agent runs 1-2 turns, writes revised output, prints summary

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/cli.py
git commit -m "feat: add 'revise' CLI command for agentic revision"
```

---

### Task 6: End-to-end validation

- [ ] **Step 1: Run full test suite**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/ -v --ignore=tests/test_agent_scan.py`
Expected: All existing tests still pass, new agent tests pass

- [ ] **Step 2: Run agent on a real chapter**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor revise /home/ben/code/nladfg/story_tracker/prose/book1/ch01.md --max-turns 4 --verbose -o /tmp/revised_ch01.md`
Expected: 2-4 accepted edits, total_distance decreases, output is valid prose

- [ ] **Step 3: Compare before/after metrics**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor scan --deep --json /tmp/revised_ch01.md | python3 -m json.tool`
Expected: Metrics closer to baselines than the original

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: complete agentic revision pipeline v1"
```
