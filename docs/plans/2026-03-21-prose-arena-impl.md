# Prose Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated ELO tournament that tunes prose-doctor's critique hyperparameters by revising AI stories under competing configurations and having an independent LLM judge pick winners.

**Architecture:** Extract all hardcoded baselines/thresholds/prescriptions into a `CritiqueConfig` dataclass, thread it through the scanning and revision pipeline, then build an arena runner that samples stories, revises them under competing configs in parallel, judges results blind, and tracks ELO ratings.

**Tech Stack:** Existing prose-doctor + `httpx` (async LLM calls) + `pyyaml` (config serialization). Local gpt-oss for revision, minimax for judging. No new ML models.

**Prior context:** Design spec at `docs/plans/2026-03-21-prose-arena-design.md`.

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `src/prose_doctor/critique_config.py` | `CritiqueConfig` dataclass, YAML serialization, default values |
| `src/prose_doctor/arena/__init__.py` | Package init |
| `src/prose_doctor/arena/sampler.py` | Story sampling from Novelist dataset, chapter extraction |
| `src/prose_doctor/arena/scanner.py` | Parallel scan worker pool (ProcessPoolExecutor, spawn) |
| `src/prose_doctor/arena/revision.py` | Async concurrent revision runner (httpx) |
| `src/prose_doctor/arena/judge.py` | Minimax blind pairwise judge |
| `src/prose_doctor/arena/elo.py` | ELO tracker, match logging, ratings |
| `src/prose_doctor/arena/runner.py` | Arena orchestrator — ties it all together |
| `tests/test_critique_config.py` | Config dataclass tests |
| `tests/test_arena_elo.py` | ELO calculation tests |
| `tests/test_arena_sampler.py` | Story sampler tests |
| `tests/test_arena_judge.py` | Judge prompt/parsing tests |
| `arena/configs/default.yaml` | Current hardcoded values as YAML |

### Modified files

| File | Change |
|------|--------|
| `src/prose_doctor/agent_models.py` | `ProseMetrics` accepts optional baselines dict; remove module-global `BASELINES` |
| `src/prose_doctor/critique.py` | `build_critique()` accepts `CritiqueConfig`; remove local `BASELINES` and `prescriptions` dicts |
| `src/prose_doctor/agent_issues.py` | All `find_*_issues()` accept optional `CritiqueConfig`; extract hardcoded thresholds |
| `src/prose_doctor/agent_scan.py` | `scan_deep()` accepts optional `CritiqueConfig`, passes to `ProseMetrics` |
| `src/prose_doctor/orchestrated_revise.py` | `run_orchestrated()` accepts optional `CritiqueConfig` |
| `src/prose_doctor/cli.py` | Add `arena` subcommand with `run` and `ratings` |
| `pyproject.toml` | Add `httpx` and `pyyaml` to dependencies |

---

## Task 1: CritiqueConfig dataclass

**Files:**
- Create: `src/prose_doctor/critique_config.py`
- Test: `tests/test_critique_config.py`

- [ ] **Step 1: Write the test — default config matches current hardcoded values**

```python
# tests/test_critique_config.py
from prose_doctor.critique_config import CritiqueConfig


def test_default_baselines_match_current():
    """Default CritiqueConfig baselines must match the current hardcoded values."""
    cfg = CritiqueConfig()
    assert cfg.baselines["pd_mean"] == (0.336, "higher")
    assert cfg.baselines["fg_fragment"] == (6.7, "lower")
    assert cfg.baselines["dr_entropy"] == (0.65, "higher")
    assert len(cfg.baselines) == 12


def test_default_prescriptions_exist_for_all_baselines():
    """Every baseline metric should have a prescription."""
    cfg = CritiqueConfig()
    for key in cfg.baselines:
        assert key in cfg.prescriptions, f"Missing prescription for {key}"
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_critique_config.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement CritiqueConfig**

```python
# src/prose_doctor/critique_config.py
"""Centralized configuration for critique baselines, thresholds, and prescriptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Labels for human-readable critique output (keyed by metric name)
METRIC_LABELS: dict[str, str] = {
    "pd_mean": "Psychic distance",
    "pd_std": "Distance variation",
    "fg_inversion": "Sentence inversion",
    "fg_sl_cv": "Sentence length variation",
    "fg_fragment": "Fragment ratio",
    "ic_rhythmicity": "Information rhythmicity",
    "ic_spikes": "Surprisal spikes",
    "ic_flatlines": "Information flatlines",
    "dr_entropy": "Discourse relation diversity",
    "dr_implicit": "Implicit relation ratio",
    "cn_abstract": "Abstractness ratio",
    "ss_shift_rate": "Situation shift rate",
}


@dataclass
class CritiqueConfig:
    """All tunable parameters for critique, scanning, and issue finding."""

    name: str = "default"
    parent: str | None = None

    # --- Baselines: (target_value, direction) ---
    baselines: dict[str, tuple[float, str]] = field(default_factory=lambda: {
        "pd_mean":       (0.336, "higher"),
        "pd_std":        (0.093, "higher"),
        "fg_inversion":  (44.2,  "higher"),
        "fg_sl_cv":      (0.706, "higher"),
        "fg_fragment":   (6.7,   "lower"),
        "ic_rhythmicity": (0.129, "lower"),
        "ic_spikes":     (7.7,   "higher"),
        "ic_flatlines":  (3.1,   "lower"),
        "dr_entropy":    (0.65,  "higher"),
        "dr_implicit":   (0.90,  "lower"),
        "cn_abstract":   (0.27,  "higher"),
        "ss_shift_rate": (1.5,   "higher"),
    })

    # --- Severity gates ---
    major_threshold: float = 0.25
    minor_threshold: float = 0.15
    strength_threshold: float = 0.05

    # --- Issue finder thresholds ---
    # Discourse relations
    discourse_entropy_gate: float = 0.55
    discourse_implicit_gate: float = 0.92
    consecutive_implicit_trigger: int = 3
    additive_count_trigger: int = 2
    # Concreteness
    concrete_run_trigger: int = 4
    concrete_para_mean_threshold: float = 3.2
    abstract_ratio_gate: float = 0.15
    vague_density_gate: float = 0.5
    # Situation shifts
    shift_rate_gate: float = 1.2
    no_shift_run_trigger: int = 5
    # Psychic distance
    pd_baseline_margin: float = 0.05
    pd_cause_threshold: int = 2
    # Inversions
    inversion_pct_gate: float = 15.0
    # Spikes
    spike_surprisal_margin: float = 0.1

    # --- Prescriptions ---
    prescriptions: dict[str, str] = field(default_factory=lambda: {
        "pd_mean": (
            "You're narrating from outside the character. Pick the 3-5 most "
            "emotionally charged moments and rewrite them from deep inside the "
            "character's perception — what they feel in their body, what they "
            "notice, what they're afraid of. Use perception verbs (saw, felt, "
            "heard) and interior thought."
        ),
        "pd_std": (
            "Your narrative distance is flat — you stay at the same zoom level "
            "throughout. Pull back to establishing-shot distance for scene "
            "transitions, then push in close for confrontations and revelations. "
            "The contrast creates rhythm."
        ),
        "fg_inversion": (
            "Too many sentences follow subject-verb-object order. Restructure "
            "5-8 sentences to lead with a prepositional phrase, a verb, or a "
            "subordinate clause. 'Down the corridor she ran' instead of "
            "'She ran down the corridor.'"
        ),
        "fg_sl_cv": (
            "Your sentences are too uniform in length. Break 3-4 long sentences "
            "into staccato fragments at high-tension moments. Merge 3-4 short "
            "sentences into complex, flowing ones during reflective passages."
        ),
        "fg_fragment": (
            "You're overusing short fragments for manufactured emphasis. "
            "Each fragment should earn its weight — save them for genuine "
            "moments of impact. Merge the weaker ones back into full sentences."
        ),
        "ic_rhythmicity": (
            "Your information density is too metronomic — every paragraph "
            "delivers roughly the same amount of new information. Vary the "
            "density: follow a dense paragraph with a sparse, atmospheric one."
        ),
        "ic_spikes": (
            "Your prose is too predictable — not enough moments where the "
            "language itself surprises. At 3-4 key moments, choose an "
            "unexpected word, an unusual metaphor, or a syntactic structure "
            "the reader doesn't see coming."
        ),
        "ic_flatlines": (
            "Your prose has long stretches of uniform information density. "
            "Break these up by inserting a moment of high tension, a surprising "
            "detail, or a shift in narrative mode."
        ),
        "dr_entropy": (
            "Replace some 'and' connectives with 'because', 'so', 'but', or "
            "'although' to show logical relationships between sentences."
        ),
        "dr_implicit": (
            "Add connectives to show how sentences relate — causal, contrastive, "
            "temporal. Readers shouldn't have to guess the logical connection."
        ),
        "cn_abstract": (
            "Add moments of reflection, interpretation, memory, or opinion "
            "to break the sensory surface."
        ),
        "ss_shift_rate": (
            "Add time references, location changes, or new characters entering "
            "to break static scenes."
        ),
    })

    # --- Revision loop ---
    max_turns: int = 8
    regression_limit: float = 0.20
    temperature: float = 0.7

    @classmethod
    def from_yaml(cls, path: Path) -> "CritiqueConfig":
        """Load config from YAML file."""
        import yaml
        data = yaml.safe_load(path.read_text())
        # Convert baselines from list to tuple
        if "baselines" in data:
            data["baselines"] = {
                k: (v[0], v[1]) for k, v in data["baselines"].items()
            }
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        import yaml
        data = {
            "name": self.name,
            "parent": self.parent,
            "baselines": {k: list(v) for k, v in self.baselines.items()},
            "major_threshold": self.major_threshold,
            "minor_threshold": self.minor_threshold,
            "strength_threshold": self.strength_threshold,
            "discourse_entropy_gate": self.discourse_entropy_gate,
            "discourse_implicit_gate": self.discourse_implicit_gate,
            "consecutive_implicit_trigger": self.consecutive_implicit_trigger,
            "additive_count_trigger": self.additive_count_trigger,
            "concrete_run_trigger": self.concrete_run_trigger,
            "concrete_para_mean_threshold": self.concrete_para_mean_threshold,
            "abstract_ratio_gate": self.abstract_ratio_gate,
            "vague_density_gate": self.vague_density_gate,
            "shift_rate_gate": self.shift_rate_gate,
            "no_shift_run_trigger": self.no_shift_run_trigger,
            "pd_baseline_margin": self.pd_baseline_margin,
            "pd_cause_threshold": self.pd_cause_threshold,
            "inversion_pct_gate": self.inversion_pct_gate,
            "spike_surprisal_margin": self.spike_surprisal_margin,
            "prescriptions": self.prescriptions,
            "max_turns": self.max_turns,
            "regression_limit": self.regression_limit,
            "temperature": self.temperature,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
```

- [ ] **Step 4: Add YAML round-trip test**

```python
def test_yaml_round_trip(tmp_path):
    """Config survives YAML serialization."""
    cfg = CritiqueConfig(name="test_config")
    path = tmp_path / "test.yaml"
    cfg.to_yaml(path)
    loaded = CritiqueConfig.from_yaml(path)
    assert loaded.name == "test_config"
    assert loaded.baselines == cfg.baselines
    assert loaded.prescriptions == cfg.prescriptions
    assert loaded.major_threshold == cfg.major_threshold
```

- [ ] **Step 5: Run tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_critique_config.py -v`
Expected: PASS

- [ ] **Step 6: Add pyyaml dependency**

Run: `cd /home/ben/code/prose-doctor && uv add pyyaml`

- [ ] **Step 7: Commit**

```bash
git add src/prose_doctor/critique_config.py tests/test_critique_config.py pyproject.toml uv.lock
git commit -m "feat: CritiqueConfig dataclass — centralized baselines, thresholds, prescriptions"
```

---

## Task 2: Thread CritiqueConfig through ProseMetrics

**Files:**
- Modify: `src/prose_doctor/agent_models.py`
- Modify: `tests/test_agent_models.py`

The key change: `ProseMetrics._metric_distance()` currently reads the module-global `BASELINES`. It needs to accept baselines from a config. Since `ProseMetrics` is a Pydantic `BaseModel`, we store the baselines as a private field.

- [ ] **Step 1: Update ProseMetrics to accept baselines**

In `src/prose_doctor/agent_models.py`, replace the module-global `BASELINES` import pattern:

```python
"""Pydantic models for the revision agent."""
from __future__ import annotations

from pydantic import BaseModel, PrivateAttr, computed_field

# Default baselines — used when no CritiqueConfig is provided.
# Import CritiqueConfig defaults to avoid duplication.
def _default_baselines() -> dict[str, tuple[float, str]]:
    from prose_doctor.critique_config import CritiqueConfig
    return CritiqueConfig().baselines


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
    dr_entropy: float = 0.0
    dr_implicit: float = 1.0
    cn_abstract: float = 0.0
    ss_shift_rate: float = 0.0

    # Optional baselines override — not serialized
    _baselines: dict[str, tuple[float, str]] | None = PrivateAttr(default=None)

    def __init__(self, baselines: dict[str, tuple[float, str]] | None = None, **data):
        super().__init__(**data)
        self._baselines = baselines

    @property
    def baselines(self) -> dict[str, tuple[float, str]]:
        if self._baselines is not None:
            return self._baselines
        return _default_baselines()

    def _metric_distance(self, key: str) -> float:
        """Normalized distance from baseline for one metric. 0 = at/past baseline."""
        baseline, direction = self.baselines[key]
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
        return round(sum(self._metric_distance(k) for k in self.baselines), 4)

    @computed_field
    @property
    def worst_metric(self) -> str:
        return max(self.baselines, key=lambda k: self._metric_distance(k))

    def distances(self) -> dict[str, float]:
        return {k: round(self._metric_distance(k), 4) for k in self.baselines}
```

Also keep a module-level `BASELINES` property for backward compatibility (other modules import it):

```python
# Backward compat — modules that import BASELINES still work
BASELINES = _default_baselines()
```

But note: this is now a function call, not a constant. Any module that does `from prose_doctor.agent_models import BASELINES` at import time will get the default dict. This is fine — the arena passes overrides through `ProseMetrics(baselines=config.baselines)`.

- [ ] **Step 2: Run existing tests — verify they still pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_models.py tests/test_agent.py -v`
Expected: PASS (default baselines = same values as before)

- [ ] **Step 3: Add test for custom baselines**

```python
# Add to tests/test_agent_models.py
def test_prose_metrics_custom_baselines():
    """ProseMetrics uses custom baselines when provided."""
    custom = {"pd_mean": (0.5, "higher"), "pd_std": (0.1, "higher")}
    m = ProseMetrics(
        pd_mean=0.5, pd_std=0.1,
        fg_inversion=44.2, fg_sl_cv=0.706, fg_fragment=6.7,
        ic_rhythmicity=0.129, ic_spikes=8, ic_flatlines=3,
        baselines=custom,
    )
    # Only 2 metrics in custom baselines, both at baseline → distance 0
    assert m.total_distance == 0.0
    assert len(m.distances()) == 2
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/agent_models.py tests/test_agent_models.py
git commit -m "refactor: ProseMetrics accepts optional baselines override

BASELINES module-global preserved for backward compat. Arena will
pass CritiqueConfig.baselines through to ProseMetrics."
```

---

## Task 3: Thread CritiqueConfig through critique.py

**Files:**
- Modify: `src/prose_doctor/critique.py`

- [ ] **Step 1: Update build_critique to accept CritiqueConfig**

Replace the local `BASELINES` dict (lines 22-33) and the local `prescriptions` dict (lines 157-205) with a `CritiqueConfig` parameter:

```python
# At top of critique.py, replace the BASELINES dict with:
from prose_doctor.critique_config import CritiqueConfig, METRIC_LABELS
```

Update `build_critique` signature (line 132):

```python
def build_critique(
    report: dict,
    twins: list | None = None,
    config: CritiqueConfig | None = None,
) -> list[CritiqueSection]:
    """Build critique sections from a scan --deep report."""
    cfg = config or CritiqueConfig()
    sections: list[CritiqueSection] = []
    # ... existing code for extracting values from report ...

    for key, (baseline, direction) in cfg.baselines.items():
        label = METRIC_LABELS.get(key, key)
        value = values.get(key, 0)
        if _is_problem(value, baseline, direction, cfg.minor_threshold):
            severity = "major" if _is_problem(value, baseline, direction, cfg.major_threshold) else "minor"
            sections.append(CritiqueSection(
                dimension=label,
                severity=severity,
                value=value,
                baseline=baseline,
                direction=direction,
                prescription=cfg.prescriptions.get(key, ""),
            ))
        elif _is_strength(value, baseline, direction):
            # ... existing strength logic ...
```

Update `_is_problem` and `_is_strength` to be usable with config thresholds (they already accept threshold params, just wire through).

Also add the new metrics to the `values` dict extraction (lines 144-154) — add `dr_entropy`, `dr_implicit`, `cn_abstract`, `ss_shift_rate` from the report's discourse_relations, concreteness, and situation_shifts sections.

- [ ] **Step 2: Remove the local BASELINES and prescriptions dicts**

Delete lines 20-33 (local `BASELINES`) and lines 156-205 (local `prescriptions`). These now live in `CritiqueConfig`.

- [ ] **Step 3: Run existing tests**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/ -v -k "not slow"`
Expected: PASS (default config = same values)

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/critique.py
git commit -m "refactor: build_critique accepts CritiqueConfig, removes local baselines/prescriptions"
```

---

## Task 4: Thread CritiqueConfig through issue finders

**Files:**
- Modify: `src/prose_doctor/agent_issues.py`

Each `find_*_issues()` function has hardcoded thresholds. Add an optional `config: CritiqueConfig | None = None` parameter to each, and use `config.{threshold}` instead of hardcoded values.

- [ ] **Step 1: Update find_discourse_issues**

Add `config` param, replace hardcoded gates:
- `0.55` → `cfg.discourse_entropy_gate`
- `0.92` → `cfg.discourse_implicit_gate`
- `>= 3` → `>= cfg.consecutive_implicit_trigger`
- `>= 2` → `>= cfg.additive_count_trigger`

- [ ] **Step 2: Update find_concreteness_issues**

- `> 0.5` → `> cfg.vague_density_gate`
- `< 0.15` → `< cfg.abstract_ratio_gate`
- `> 3.2` → `> cfg.concrete_para_mean_threshold`
- `>= 4` → `>= cfg.concrete_run_trigger`

- [ ] **Step 3: Update find_shift_issues**

- `> 1.2` → `> cfg.shift_rate_gate`
- `>= 5` → `>= cfg.no_shift_run_trigger`

- [ ] **Step 4: Update find_psychic_distance_issues**

- `baseline = 0.336` → `baseline = cfg.baselines["pd_mean"][0]`
- `baseline - 0.05` → `baseline - cfg.pd_baseline_margin`
- `len(causes) >= 2` → `len(causes) >= cfg.pd_cause_threshold`

- [ ] **Step 5: Update find_inversion_issues**

- `inv_pct < 15` → `inv_pct < cfg.inversion_pct_gate`

- [ ] **Step 6: Update find_spike_issues**

- `mean_s - 0.1` → `mean_s - cfg.spike_surprisal_margin`

- [ ] **Step 7: Update find_issues and METRIC_FINDERS**

The `find_issues()` dispatch function needs to pass config through. Update its signature:

```python
def find_issues(metric: str, text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find specific problematic passages for a metric."""
    finder = METRIC_FINDERS.get(metric)
    if finder is None:
        return []
    return finder(text, report, config=config)
```

- [ ] **Step 8: Run existing tests — verify they still pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_issues.py -v`
Expected: PASS (default config = same thresholds)

- [ ] **Step 9: Commit**

```bash
git add src/prose_doctor/agent_issues.py
git commit -m "refactor: issue finders accept CritiqueConfig, extract all hardcoded thresholds"
```

---

## Task 5: Thread CritiqueConfig through scan_deep and orchestrated_revise

**Files:**
- Modify: `src/prose_doctor/agent_scan.py`
- Modify: `src/prose_doctor/orchestrated_revise.py`

- [ ] **Step 1: Update scan_deep**

Add `critique_config: CritiqueConfig | None = None` parameter. Pass `config.baselines` to `ProseMetrics(baselines=...)` constructor.

- [ ] **Step 2: Update run_orchestrated**

In `src/prose_doctor/orchestrated_revise.py`:

Replace line 25 import:
```python
# OLD: from prose_doctor.agent_models import BASELINES, ProseMetrics, RevisionResult
from prose_doctor.agent_models import ProseMetrics, RevisionResult
from prose_doctor.critique_config import CritiqueConfig
```

Update `run_orchestrated` signature (line 118):
```python
def run_orchestrated(
    text: str,
    filename: str = "chapter.md",
    max_turns: int | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    model_name: str = DEFAULT_MODEL,
    api_key: str = "none",
    verbose: bool = False,
    critique_config: CritiqueConfig | None = None,
) -> RevisionResult:
    cfg = critique_config or CritiqueConfig()
    effective_max_turns = max_turns if max_turns is not None else cfg.max_turns
```

Thread config through internal calls:
- Line 137 `scan_deep(...)` → `scan_deep(..., critique_config=cfg)`
- Line 159 `find_issues(metric, ...)` → `find_issues(metric, ..., config=cfg)`
- Line 89 `temperature=0.7` in `_call_llm` → pass `cfg.temperature` as parameter
- Line 231 `for k in BASELINES` → `for k in cfg.baselines`
- Line 108 `d > 0.10` threshold stays hardcoded (fixable_gap is not a tunable)
- Line 231 `METRIC_REGRESSION_LIMIT` → `cfg.regression_limit`

Update `_call_llm` to accept temperature:
```python
def _call_llm(
    paragraph: str,
    prescription: str,
    context_before: str,
    context_after: str,
    endpoint: str,
    model: str,
    api_key: str = "none",
    temperature: float = 0.7,
) -> str | None:
```

Update call sites in `run_orchestrated` to pass `temperature=cfg.temperature`.

- [ ] **Step 3: Run full test suite**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/agent_scan.py src/prose_doctor/orchestrated_revise.py
git commit -m "refactor: scan_deep and orchestrated_revise accept CritiqueConfig"
```

---

## Task 6: Write default.yaml and verify round-trip

**Files:**
- Create: `arena/configs/default.yaml`

- [ ] **Step 1: Write test — default.yaml produces identical behavior to hardcoded defaults**

```python
# tests/test_critique_config.py
def test_default_yaml_matches_code_defaults():
    """default.yaml must produce the same config as CritiqueConfig()."""
    from pathlib import Path
    cfg_code = CritiqueConfig()
    cfg_yaml = CritiqueConfig.from_yaml(Path("arena/configs/default.yaml"))
    assert cfg_yaml.baselines == cfg_code.baselines
    assert cfg_yaml.prescriptions == cfg_code.prescriptions
    assert cfg_yaml.major_threshold == cfg_code.major_threshold
    assert cfg_yaml.discourse_entropy_gate == cfg_code.discourse_entropy_gate
```

- [ ] **Step 2: Generate default.yaml**

```python
# Run once:
from prose_doctor.critique_config import CritiqueConfig
from pathlib import Path
CritiqueConfig().to_yaml(Path("arena/configs/default.yaml"))
```

- [ ] **Step 3: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_critique_config.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add arena/configs/default.yaml tests/test_critique_config.py
git commit -m "chore: write default.yaml from CritiqueConfig defaults"
```

---

## Task 7: Story sampler

**Files:**
- Create: `src/prose_doctor/arena/sampler.py`
- Create: `src/prose_doctor/arena/__init__.py`
- Test: `tests/test_arena_sampler.py`

- [ ] **Step 1: Write test — chapter extraction**

```python
# tests/test_arena_sampler.py
from prose_doctor.arena.sampler import extract_chapter


def test_extract_chapter_by_heading():
    """Extract chapter N from markdown with bold headings."""
    text = (
        "**Chapter 1: The Start**\n\nFirst chapter text here. "
        "Enough words to count.\n\n"
        "**Chapter 2: The Middle**\n\nSecond chapter text with "
        "more words to satisfy the minimum.\n\n"
        "**Chapter 3: The End**\n\nThird chapter wraps it up."
    )
    ch2 = extract_chapter(text, chapter_num=2)
    assert ch2 is not None
    assert "Second chapter" in ch2
    assert "First chapter" not in ch2
    assert "Third chapter" not in ch2
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement sampler**

```python
# src/prose_doctor/arena/__init__.py
# (empty)

# src/prose_doctor/arena/sampler.py
"""Story sampling from the Novelist dataset."""
from __future__ import annotations

import json
import re
from pathlib import Path


def extract_chapter(text: str, chapter_num: int = 1) -> str | None:
    """Extract a chapter from a full novel text by markdown heading."""
    # Match **Chapter N: Title** or ## Chapter N patterns
    pattern = re.compile(
        r'(?:\*\*|#{1,3}\s*)Chapter\s+(\d+)[:\s].*?(?:\*\*)?',
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    # Find the target chapter
    for i, m in enumerate(matches):
        if int(m.group(1)) == chapter_num:
            start = m.end()
            # End at next chapter heading or end of text
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chapter = text[start:end].strip()
            return chapter if chapter else None
    return None


def sample_stories(
    dataset_path: Path,
    n: int = 20,
    chapter_range: tuple[int, int] = (3, 8),
    min_words: int = 1500,
    max_words: int = 4000,
    exclude_ids: set[str] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Sample stories from the Novelist JSONL dataset.

    Returns list of dicts with keys: story_id, genre, text, word_count, chapter_num.
    """
    import random
    rng = random.Random(seed)
    exclude = exclude_ids or set()

    candidates = []
    with open(dataset_path) as f:
        for line in f:
            record = json.loads(line)
            if record.get("record_type") != "full_book":
                continue
            story_id = record.get("story_id", "")
            if story_id in exclude:
                continue

            book_text = record.get("text", "")
            # Try chapters in range, pick first that fits word count
            for ch_num in range(chapter_range[0], chapter_range[1] + 1):
                chapter = extract_chapter(book_text, ch_num)
                if chapter is None:
                    continue
                wc = len(chapter.split())
                if min_words <= wc <= max_words:
                    candidates.append({
                        "story_id": story_id,
                        "genre": record.get("genre", "unknown"),
                        "text": chapter,
                        "word_count": wc,
                        "chapter_num": ch_num,
                    })
                    break  # one chapter per book

    # Stratify by genre — take proportional samples from each genre
    from collections import defaultdict
    by_genre = defaultdict(list)
    for c in candidates:
        by_genre[c["genre"]].append(c)
    for g in by_genre:
        rng.shuffle(by_genre[g])

    result = []
    genres = list(by_genre.keys())
    per_genre = max(1, n // len(genres)) if genres else n
    for g in genres:
        result.extend(by_genre[g][:per_genre])
    # Fill remaining slots from the overflow
    if len(result) < n:
        remaining = [c for c in candidates if c not in result]
        rng.shuffle(remaining)
        result.extend(remaining[:n - len(result)])
    return result[:n]


def create_holdout(
    dataset_path: Path,
    holdout_path: Path,
    n: int = 50,
    seed: int = 99,
) -> set[str]:
    """Create a held-out set of story IDs for validation."""
    stories = sample_stories(dataset_path, n=n, seed=seed)
    ids = {s["story_id"] for s in stories}
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text("\n".join(sorted(ids)))
    return ids
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_arena_sampler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/arena/__init__.py src/prose_doctor/arena/sampler.py tests/test_arena_sampler.py
git commit -m "feat: story sampler — chapter extraction from Novelist dataset"
```

---

## Task 8: ELO tracker

**Files:**
- Create: `src/prose_doctor/arena/elo.py`
- Test: `tests/test_arena_elo.py`

- [ ] **Step 1: Write test**

```python
# tests/test_arena_elo.py
from prose_doctor.arena.elo import EloTracker


def test_elo_winner_gains_rating():
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="a", story_id="s1", reason="better")
    assert tracker.rating("a") > 1500
    assert tracker.rating("b") < 1500


def test_elo_tie_splits_points():
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="tie", story_id="s1", reason="equal")
    # Both should still be 1500 (equal expected, tie = no movement)
    assert tracker.rating("a") == 1500
    assert tracker.rating("b") == 1500
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement ELO tracker**

```python
# src/prose_doctor/arena/elo.py
"""ELO rating tracker with JSONL match logging."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


K = 32  # Standard K-factor


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


@dataclass
class EloTracker:
    """Track ELO ratings for critique configs."""

    ratings: dict[str, float] = field(default_factory=dict)
    matches: list[dict] = field(default_factory=list)

    def add_config(self, name: str, rating: float = 1500.0) -> None:
        if name not in self.ratings:
            self.ratings[name] = rating

    def rating(self, name: str) -> float:
        return self.ratings[name]

    def record_match(
        self,
        config_a: str,
        config_b: str,
        winner: str,  # config_a name, config_b name, or "tie"
        story_id: str,
        reason: str = "",
        metadata: dict | None = None,
    ) -> None:
        ra, rb = self.ratings[config_a], self.ratings[config_b]
        ea, eb = _expected(ra, rb), _expected(rb, ra)

        if winner == config_a:
            sa, sb = 1.0, 0.0
        elif winner == config_b:
            sa, sb = 0.0, 1.0
        else:  # tie
            sa, sb = 0.5, 0.5

        self.ratings[config_a] = ra + K * (sa - ea)
        self.ratings[config_b] = rb + K * (sb - eb)

        record = {
            "config_a": config_a,
            "config_b": config_b,
            "winner": winner,
            "story_id": story_id,
            "reason": reason,
            "rating_a_after": self.ratings[config_a],
            "rating_b_after": self.ratings[config_b],
            **(metadata or {}),
        }
        self.matches.append(record)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])

    def save(self, matches_path: Path, ratings_path: Path) -> None:
        matches_path.parent.mkdir(parents=True, exist_ok=True)
        with open(matches_path, "a") as f:
            for m in self.matches:
                f.write(json.dumps(m) + "\n")
        ratings_path.write_text(json.dumps(
            {"configs": {k: {"elo": round(v, 1)} for k, v in self.ratings.items()}},
            indent=2,
        ))
        self.matches.clear()  # flushed

    @classmethod
    def load(cls, matches_path: Path, ratings_path: Path) -> "EloTracker":
        tracker = cls()
        if ratings_path.exists():
            data = json.loads(ratings_path.read_text())
            for name, info in data.get("configs", {}).items():
                tracker.ratings[name] = info["elo"]
        return tracker
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_arena_elo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/arena/elo.py tests/test_arena_elo.py
git commit -m "feat: ELO tracker with JSONL match logging"
```

---

## Task 9: Minimax judge

**Files:**
- Create: `src/prose_doctor/arena/judge.py`
- Test: `tests/test_arena_judge.py`

- [ ] **Step 1: Write test — prompt construction and response parsing**

```python
# tests/test_arena_judge.py
from prose_doctor.arena.judge import build_judge_prompt, parse_judge_response


def test_build_judge_prompt_randomizes_position():
    """Judge prompt should randomly assign configs to X/Y positions."""
    import random
    rng = random.Random(42)
    prompt, mapping = build_judge_prompt("orig", "rev_a", "rev_b", rng)
    assert "Version X:" in prompt
    assert "Version Y:" in prompt
    assert "orig" not in prompt  # original is labeled as "Original:"
    assert mapping["X"] in ("a", "b")
    assert mapping["Y"] in ("a", "b")
    assert mapping["X"] != mapping["Y"]


def test_parse_judge_response_valid_json():
    resp = '{"winner": "X", "reason": "better rhythm"}'
    result = parse_judge_response(resp, {"X": "config_a", "Y": "config_b"})
    assert result["winner"] == "config_a"
    assert result["reason"] == "better rhythm"


def test_parse_judge_response_tie():
    resp = '{"winner": "tie", "reason": "too close"}'
    result = parse_judge_response(resp, {"X": "a", "Y": "b"})
    assert result["winner"] == "tie"
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement judge**

```python
# src/prose_doctor/arena/judge.py
"""Blind pairwise judge using minimax (or any LLM)."""
from __future__ import annotations

import json
import random
import re

JUDGE_SYSTEM = """\
You are a literary editor comparing two revisions of the same passage. \
Judge on: sentence variety, interiority, sensory grounding, dialogue \
naturalness, rhythm, and overall craft. Ignore plot — focus on technique."""

JUDGE_TEMPLATE = """\
Read the original, then both revisions. Which revision is better prose?

Original:
{original}

Version X:
{version_x}

Version Y:
{version_y}

Respond in JSON: {{"winner": "X" | "Y" | "tie", "reason": "..."}}"""


def build_judge_prompt(
    original: str,
    revised_a: str,
    revised_b: str,
    rng: random.Random | None = None,
) -> tuple[str, dict[str, str]]:
    """Build judge prompt with randomized position assignment."""
    rng = rng or random.Random()
    if rng.random() < 0.5:
        mapping = {"X": "a", "Y": "b"}
        version_x, version_y = revised_a, revised_b
    else:
        mapping = {"X": "b", "Y": "a"}
        version_x, version_y = revised_b, revised_a

    prompt = JUDGE_TEMPLATE.format(
        original=original[:3000],
        version_x=version_x[:3000],
        version_y=version_y[:3000],
    )
    return prompt, mapping


def parse_judge_response(
    response: str,
    mapping: dict[str, str],
) -> dict:
    """Parse judge JSON response, map X/Y back to config names."""
    # Strip think tags and markdown
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # Find JSON in response
    match = re.search(r'\{[^}]+\}', cleaned)
    if not match:
        return {"winner": "tie", "reason": "failed to parse judge response"}

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {"winner": "tie", "reason": "invalid JSON from judge"}

    raw_winner = data.get("winner", "tie").strip().upper()
    reason = data.get("reason", "")

    if raw_winner in ("X", "Y"):
        winner = mapping[raw_winner]
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
    rng: random.Random | None = None,
) -> dict:
    """Run blind pairwise judgment. Returns {winner, reason, position_map}."""
    import httpx

    prompt, mapping = build_judge_prompt(original, revised_a, revised_b, rng)
    # Map internal a/b to config names
    name_map = {}
    for pos, ab in mapping.items():
        name_map[pos] = config_a_name if ab == "a" else config_b_name

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{endpoint}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

    result = parse_judge_response(content, name_map)
    result["position_map"] = {pos: name_map[pos] for pos in mapping}
    return result
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_arena_judge.py -v`
Expected: PASS

- [ ] **Step 5: Add httpx dependency**

Run: `cd /home/ben/code/prose-doctor && uv add httpx`

- [ ] **Step 6: Commit**

```bash
git add src/prose_doctor/arena/judge.py tests/test_arena_judge.py pyproject.toml uv.lock
git commit -m "feat: minimax blind pairwise judge with position debiasing"
```

---

## Task 10: Parallel scan worker pool

**Files:**
- Create: `src/prose_doctor/arena/scanner.py`

- [ ] **Step 1: Implement scanner**

```python
# src/prose_doctor/arena/scanner.py
"""Parallel scan worker pool using ProcessPoolExecutor with spawn context."""
from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _scan_one(story: dict) -> dict | None:
    """Scan a single story. Runs in a spawned child process."""
    try:
        from prose_doctor.providers import require_ml
        require_ml()
        from prose_doctor.agent_scan import scan_deep

        metrics, report = scan_deep(story["text"], filename=f"{story['story_id']}.md")
        return {
            "story_id": story["story_id"],
            "genre": story["genre"],
            "text": story["text"],
            "word_count": story["word_count"],
            "chapter_num": story["chapter_num"],
            "metrics": metrics.model_dump(),
            "report": report,
        }
    except Exception as e:
        import sys
        print(f"Scan failed for {story['story_id']}: {e}", file=sys.stderr)
        return None


def scan_stories(
    stories: list[dict],
    cache_dir: Path,
    max_workers: int = 2,
) -> list[dict]:
    """Scan stories in parallel, caching results.

    Uses spawn context to avoid CUDA/spaCy fork corruption.
    """
    results = []
    to_scan = []

    # Check cache first
    cache_dir.mkdir(parents=True, exist_ok=True)
    for story in stories:
        cache_path = cache_dir / f"{story['story_id']}.json"
        if cache_path.exists():
            results.append(json.loads(cache_path.read_text()))
        else:
            to_scan.append(story)

    if not to_scan:
        return results

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        for result in pool.map(_scan_one, to_scan):
            if result is not None:
                # Cache it
                cache_path = cache_dir / f"{result['story_id']}.json"
                cache_path.write_text(json.dumps(result))
                results.append(result)

    return results
```

- [ ] **Step 2: Commit**

```bash
git add src/prose_doctor/arena/scanner.py
git commit -m "feat: parallel scan worker pool (spawn context, cached results)"
```

---

## Task 11: Async revision runner

**Files:**
- Create: `src/prose_doctor/arena/revision.py`

- [ ] **Step 1: Implement async revision runner**

```python
# src/prose_doctor/arena/revision.py
"""Async concurrent revision runner for arena matches."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from prose_doctor.critique_config import CritiqueConfig


async def revise_story(
    text: str,
    config: CritiqueConfig,
    endpoint: str,
    model: str,
) -> tuple[str, dict]:
    """Run orchestrated revision in a thread (it's sync internally).

    Returns (revised_text, metrics_after_dict).
    """
    loop = asyncio.get_event_loop()

    def _run():
        from prose_doctor.orchestrated_revise import run_orchestrated
        result = run_orchestrated(
            text,
            max_turns=config.max_turns,
            endpoint=endpoint,
            model_name=model,
            critique_config=config,
        )
        return result.final_text, result.metrics_final.model_dump()

    return await loop.run_in_executor(None, _run)


async def run_match(
    story: dict,
    config_a: CritiqueConfig,
    config_b: CritiqueConfig,
    endpoint: str,
    revision_model: str,
    judge_model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run one arena match: revise under both configs, judge the results."""
    import asyncio
    from prose_doctor.arena.judge import judge_pair

    original = story["text"]
    story_id = story["story_id"]

    print(f"  Match: {config_a.name} vs {config_b.name} on {story_id}", file=sys.stderr)

    # Revise under both configs concurrently (bounded by semaphore)
    async def _bounded(cfg):
        async with semaphore:
            return await revise_story(original, cfg, endpoint, revision_model)

    (revised_a_text, metrics_a), (revised_b_text, metrics_b) = await asyncio.gather(
        _bounded(config_a),
        _bounded(config_b),
    )

    # Judge
    judge_result = await judge_pair(
        original, revised_a_text, revised_b_text,
        config_a.name, config_b.name,
        endpoint, judge_model,
    )

    return {
        "story_id": story_id,
        "config_a": config_a.name,
        "config_b": config_b.name,
        "winner": judge_result["winner"],
        "reason": judge_result["reason"],
        "position_map": judge_result.get("position_map", {}),
        "metrics_a_before": story.get("metrics", {}),
        "metrics_a_after": metrics_a,
        "metrics_b_before": story.get("metrics", {}),
        "metrics_b_after": metrics_b,
    }
```

- [ ] **Step 2: Commit**

```bash
git add src/prose_doctor/arena/revision.py
git commit -m "feat: async revision runner for arena matches"
```

---

## Task 12: Arena orchestrator + CLI

**Files:**
- Create: `src/prose_doctor/arena/runner.py`
- Modify: `src/prose_doctor/cli.py`

- [ ] **Step 1: Implement arena runner**

```python
# src/prose_doctor/arena/runner.py
"""Arena orchestrator — ties sampling, scanning, revision, judging, and ELO together."""
from __future__ import annotations

import asyncio
import itertools
import sys
from pathlib import Path

from prose_doctor.critique_config import CritiqueConfig
from prose_doctor.arena.sampler import sample_stories
from prose_doctor.arena.scanner import scan_stories
from prose_doctor.arena.revision import run_match
from prose_doctor.arena.elo import EloTracker


def run_arena(
    config_paths: list[Path],
    dataset_path: Path,
    n_stories: int = 20,
    endpoint: str = "http://localhost:8081/v1",
    revision_model: str = "gpt-oss-120b",
    judge_model: str = "minimax",
    scan_workers: int = 2,
    revision_slots: int = 3,
    arena_dir: Path = Path("arena"),
) -> None:
    """Run a full arena tournament."""
    # Load configs
    configs = [CritiqueConfig.from_yaml(p) for p in config_paths]
    print(f"Loaded {len(configs)} configs: {[c.name for c in configs]}", file=sys.stderr)

    # Load or create ELO tracker
    matches_path = arena_dir / "results" / "matches.jsonl"
    ratings_path = arena_dir / "results" / "ratings.json"
    tracker = EloTracker.load(matches_path, ratings_path)
    for c in configs:
        tracker.add_config(c.name)

    # Sample stories
    # Create held-out set if it doesn't exist
    holdout_path = arena_dir / "holdout" / "story_ids.txt"
    if not holdout_path.exists():
        from prose_doctor.arena.sampler import create_holdout
        print("Creating held-out validation set (50 stories)...", file=sys.stderr)
        create_holdout(dataset_path, holdout_path, n=50)
    exclude = set(holdout_path.read_text().strip().split("\n"))

    print(f"Sampling {n_stories} stories...", file=sys.stderr)
    stories = sample_stories(dataset_path, n=n_stories, exclude_ids=exclude)
    print(f"  Got {len(stories)} stories", file=sys.stderr)

    # Scan all stories
    print("Scanning stories...", file=sys.stderr)
    scanned = scan_stories(stories, arena_dir / "scanned", max_workers=scan_workers)
    # Filter by minimum distance
    scanned = [s for s in scanned if s.get("metrics", {}).get("total_distance", 0) > 2.0]
    print(f"  {len(scanned)} stories with total_distance > 2.0", file=sys.stderr)

    # Generate round-robin match schedule
    config_pairs = list(itertools.combinations(configs, 2))
    print(f"Round-robin: {len(config_pairs)} config pairs × {len(scanned)} stories = {len(config_pairs) * len(scanned)} matches", file=sys.stderr)

    # Run matches
    sem = asyncio.Semaphore(revision_slots)

    async def _run_all():
        for story in scanned:
            for cfg_a, cfg_b in config_pairs:
                try:
                    result = await run_match(
                        story, cfg_a, cfg_b,
                        endpoint, revision_model, judge_model, sem,
                    )
                    # Compute per-metric winners
                    per_metric = {}
                    ma = result.get("metrics_a_after", {})
                    mb = result.get("metrics_b_after", {})
                    for k in ma:
                        if k in ("total_distance", "worst_metric"):
                            continue
                        da = ma.get("total_distance", 99)
                        db = mb.get("total_distance", 99)
                        # Lower distance = better
                        if ma.get(k, 0) != mb.get(k, 0):
                            per_metric[k] = result["config_a"] if da < db else result["config_b"]

                    tracker.record_match(
                        result["config_a"], result["config_b"],
                        winner=result["winner"],
                        story_id=result["story_id"],
                        reason=result["reason"],
                        metadata={
                            "metrics_a_after": result["metrics_a_after"],
                            "metrics_b_after": result["metrics_b_after"],
                            "position_map": result["position_map"],
                            "per_metric_winner": per_metric,
                        },
                    )
                    tracker.save(matches_path, ratings_path)
                    _print_leaderboard(tracker)
                except Exception as e:
                    print(f"  Match failed: {e}", file=sys.stderr)
                    # Log to error JSONL
                    import json
                    errors_path = arena_dir / "results" / "errors.jsonl"
                    errors_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(errors_path, "a") as ef:
                        ef.write(json.dumps({
                            "story_id": story.get("story_id", "unknown"),
                            "config_a": cfg_a.name,
                            "config_b": cfg_b.name,
                            "error": str(e),
                        }) + "\n")

    asyncio.run(_run_all())

    # Final leaderboard
    print("\n=== Final Leaderboard ===", file=sys.stderr)
    _print_leaderboard(tracker)


def _print_leaderboard(tracker: EloTracker) -> None:
    for name, elo in tracker.leaderboard():
        print(f"  {name:30s} {elo:7.1f}", file=sys.stderr)


def print_ratings(arena_dir: Path = Path("arena")) -> None:
    """Print current ELO ratings."""
    ratings_path = arena_dir / "results" / "ratings.json"
    if not ratings_path.exists():
        print("No ratings found. Run `prose-doctor arena run` first.")
        return
    tracker = EloTracker.load(Path("/dev/null"), ratings_path)
    for name, elo in tracker.leaderboard():
        print(f"  {name:30s} {elo:7.1f}")
```

- [ ] **Step 2: Add arena subcommand to CLI**

In `src/prose_doctor/cli.py`, add to the argument parser and handler:

```python
# In the subparsers section:
p_arena = subparsers.add_parser("arena", help="Run critique tuning arena")
arena_sub = p_arena.add_subparsers(dest="arena_command")

p_arena_run = arena_sub.add_parser("run", help="Run a tournament round")
p_arena_run.add_argument("--configs", nargs="+", required=True, help="Config YAML files")
p_arena_run.add_argument("--stories", type=int, default=20, help="Stories per round")
p_arena_run.add_argument("--dataset", default="refs/novelist/data.jsonl", help="Dataset path")
p_arena_run.add_argument("--endpoint", default="http://localhost:8081/v1")
p_arena_run.add_argument("--revision-model", default="gpt-oss-120b")
p_arena_run.add_argument("--judge-model", default="minimax")
p_arena_run.add_argument("--scan-workers", type=int, default=2)

p_arena_ratings = arena_sub.add_parser("ratings", help="Show current ratings")

# In the handlers dict:
def cmd_arena(args):
    if args.arena_command == "run":
        from prose_doctor.arena.runner import run_arena
        run_arena(
            config_paths=[Path(p) for p in args.configs],
            dataset_path=Path(args.dataset),
            n_stories=args.stories,
            endpoint=args.endpoint,
            revision_model=args.revision_model,
            judge_model=args.judge_model,
            scan_workers=args.scan_workers,
        )
    elif args.arena_command == "ratings":
        from prose_doctor.arena.runner import print_ratings
        print_ratings()

handlers["arena"] = cmd_arena
```

- [ ] **Step 3: Run full test suite to verify nothing broken**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/arena/runner.py src/prose_doctor/cli.py
git commit -m "feat: arena runner + CLI command (prose-doctor arena run/ratings)"
```

---

## Task 13: Integration smoke test

- [ ] **Step 1: Create a second config for testing**

Create `arena/configs/fragment_relaxed.yaml` — same as default but with `fg_fragment` baseline raised to 10.0 (more tolerant of fragments):

```bash
cd /home/ben/code/prose-doctor && uv run python -c "
from prose_doctor.critique_config import CritiqueConfig
from pathlib import Path
cfg = CritiqueConfig(name='fragment_relaxed')
cfg.baselines['fg_fragment'] = (10.0, 'lower')
cfg.to_yaml(Path('arena/configs/fragment_relaxed.yaml'))
print('Written.')
"
```

- [ ] **Step 2: Dry-run arena with 2 configs, 1 story**

```bash
cd /home/ben/code/prose-doctor && uv run python -m prose_doctor arena run \
  --configs arena/configs/default.yaml arena/configs/fragment_relaxed.yaml \
  --stories 1 \
  --endpoint http://localhost:8081/v1 \
  --revision-model gpt-oss-120b \
  --judge-model minimax
```

Expected: Scans 1 story, revises under both configs, judges, prints leaderboard with 2 configs.

- [ ] **Step 3: Verify match log and ratings**

```bash
cat arena/results/matches.jsonl
cat arena/results/ratings.json
```

Expected: 1 match record in JSONL, ratings.json with 2 configs.

- [ ] **Step 4: Commit any fixes**

```bash
git add -u arena/
git commit -m "chore: arena integration smoke test + fragment_relaxed config"
```

---

## Summary

| Task | What it does | Depends on |
|------|-------------|------------|
| 1 | CritiqueConfig dataclass | — |
| 2 | Thread config through ProseMetrics | 1 |
| 3 | Thread config through critique.py | 1 |
| 4 | Thread config through issue finders | 1 |
| 5 | Thread config through scan_deep + orchestrated_revise | 1, 2, 3, 4 |
| 6 | Write default.yaml | 1 |
| 7 | Story sampler | — |
| 8 | ELO tracker | — |
| 9 | Minimax judge | — |
| 10 | Parallel scan pool | — |
| 11 | Async revision runner | 5 |
| 12 | Arena orchestrator + CLI | 7, 8, 9, 10, 11 |
| 13 | Integration smoke test | 12 |

Tasks 1-6 are Phase 1 (config extraction). Tasks 7-13 are Phase 2 (arena core). Tasks 7-10 are independent and can be parallelized.
