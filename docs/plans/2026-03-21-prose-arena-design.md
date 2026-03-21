# Prose Arena: Automated Critique Tuning via ELO Tournament

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Tune prose-doctor's critique hyperparameters (baselines, thresholds, prescriptions) by running automated tournaments — revise AI stories under competing configurations, have an independent LLM judge pick winners blind, and use ELO ratings to surface the best configuration.

**Why:** The current baselines are derived from corpus statistics (3,313 AI novels vs 7 human novels). The thresholds and prescriptions are hand-tuned. We have no closed-loop signal on whether flagging an issue and prescribing a fix actually improves the prose. This system creates that signal.

## Architecture Overview

```
                    ┌─────────────┐
                    │  Config Pool │  (N critique configs)
                    └──────┬──────┘
                           │ sample pair (A, B)
                    ┌──────▼──────┐
                    │ Story Sampler│  (picks from 3.3K Novelist dataset)
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │     Scan Worker Pool     │  (parallel, local GPU)
              │  scan_deep() per story   │
              └────────────┬────────────┘
                           │ ProseMetrics + report
              ┌────────────▼────────────┐
              │   Critique + Revision    │  (3 concurrent gpt-oss slots)
              │  config_A → revised_A    │
              │  config_B → revised_B    │
              └────────────┬────────────┘
                           │ original + revised_A + revised_B
              ┌────────────▼────────────┐
              │     Minimax Judge        │  (1 dedicated slot)
              │  blind pairwise compare  │
              └────────────┬────────────┘
                           │ winner + reasoning
              ┌────────────▼────────────┐
              │     ELO Tracker          │  (append-only JSONL)
              │  update ratings          │
              └─────────────────────────┘
```

## Components

### 1. CritiqueConfig — The Config Dataclass

The central data structure. All tunable parameters live here. This replaces the module-global `BASELINES` in both `agent_models.py` and `critique.py` (which currently have divergent copies — see Consolidation Note below).

```python
@dataclass
class CritiqueConfig:
    """All tunable parameters for critique, scanning, and issue finding."""

    name: str = "default"
    parent: str | None = None

    # --- Baselines (single source of truth) ---
    # Replaces BASELINES in agent_models.py AND the separate copy in critique.py
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

    # --- Severity gates (critique.py _is_problem) ---
    major_threshold: float = 0.25
    minor_threshold: float = 0.15
    strength_threshold: float = 0.05

    # --- Issue finder thresholds ---
    # Discourse relations (agent_issues.py find_discourse_issues)
    discourse_entropy_gate: float = 0.55
    discourse_implicit_gate: float = 0.92
    consecutive_implicit_trigger: int = 3
    additive_count_trigger: int = 2
    # Concreteness (agent_issues.py find_concreteness_issues)
    concrete_run_trigger: int = 4
    concrete_para_mean_threshold: float = 3.2
    abstract_ratio_gate: float = 0.15
    vague_density_gate: float = 0.5
    # Situation shifts (agent_issues.py find_shift_issues)
    shift_rate_gate: float = 1.2
    no_shift_run_trigger: int = 5
    # Psychic distance (agent_issues.py find_psychic_distance_issues)
    pd_baseline_margin: float = 0.05   # flag if mean < baseline - this
    pd_cause_threshold: int = 2        # require N causes to flag
    # Inversions (agent_issues.py find_inversion_issues)
    inversion_pct_gate: float = 15.0   # flag paragraph if inv% < this
    # Spikes (agent_issues.py find_spike_issues)
    spike_surprisal_margin: float = 0.1

    # --- Prescriptions (keyed by metric name) ---
    prescriptions: dict[str, str] = field(default_factory=lambda: {
        "fg_inversion": "Restructure 5-8 sentences to lead with a prepositional phrase, verb, or subordinate clause.",
        "fg_fragment": "Merge weaker fragments back into full sentences. Save fragments for genuine impact.",
        "pd_mean": "Pick 3-5 emotionally charged moments and rewrite from deep inside the character's perception.",
        "pd_std": "Pull back to establishing-shot distance for scene transitions, then push in close for confrontations.",
        "dr_entropy": "Replace some 'and' connectives with 'because', 'so', 'but', or 'although'.",
        "dr_implicit": "Add connectives to show how sentences relate — causal, contrastive, temporal.",
        "cn_abstract": "Add moments of reflection, interpretation, memory, or opinion.",
        "ss_shift_rate": "Add time references, location changes, or new characters entering.",
        "ic_rhythmicity": "Break information flatlines with unexpected detail, sensory shift, or register change.",
        "ic_flatlines": "Break information flatlines with unexpected detail, sensory shift, or register change.",
        "ic_spikes": "Inject unexpected word choice or syntactic surprise in predictable passages.",
    })

    # --- Revision loop ---
    max_turns: int = 8
    regression_limit: float = 0.20
    temperature: float = 0.7

    @classmethod
    def from_yaml(cls, path: Path) -> "CritiqueConfig":
        """Load config from YAML file."""
        ...

    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        ...
```

**Consolidation Note:** The current codebase defines baselines in two places:
- `agent_models.py:BASELINES` — 12 metrics, `(float, str)` tuples, used by `ProseMetrics._metric_distance()`
- `critique.py` lines 22-33 — 8 metrics, `(float, str, str)` tuples (includes label), used by `build_critique()`

Phase 1 must consolidate these into the single `CritiqueConfig.baselines` dict. `ProseMetrics` must accept a baselines dict in its constructor (or a config reference) rather than reading the module global.

**Threading through the system:**
- `ProseMetrics.__init__()` gains an optional `baselines` parameter; `_metric_distance()` uses it instead of the global
- `build_critique()` gains a `config: CritiqueConfig` parameter
- Each `find_*_issues()` function gains a `config: CritiqueConfig` parameter
- `scan_deep()` gains an optional `config: CritiqueConfig` parameter, passes it through to `ProseMetrics`
- `orchestrated_revise()` gains a `config: CritiqueConfig` parameter
- Default: `CritiqueConfig()` with all current hardcoded values — existing behavior unchanged

### 2. Story Sampler

Draws stories from the Novelist dataset for arena matches.

**Strategy:**
- Sample chapters of 1,500-4,000 words (sufficient for meaningful revision)
- **Sample from chapters 3-8** rather than chapter 1 (ch1 is typically setup/exposition with less variety)
- Stratify by genre to avoid genre bias in tuning
- **Filter by minimum total_distance > 2.0** — stories already close to baseline produce ties, which is noise
- Each story is used at most once per tournament round (avoid memorization effects)
- Pre-scan all sampled stories before the round starts (amortize GPU cost)
- **Reserve 50 stories as a held-out validation set** — never used in matches, used to evaluate final winner

**Chapter extraction:** The `full_book` text field contains the entire novel with chapters separated by markdown headings (`**Chapter N: Title**` or `## Chapter N`). The sampler detects chapter boundaries via regex, extracts the target chapter, and filters for word count within the 1,500-4,000 range.

**Implementation:** Reads `refs/novelist/data.jsonl`, filters for `full_book` records, extracts target chapter, caches scanned results in `arena/scanned/`.

### 3. Scan Worker Pool

Parallelizes lens scanning across stories. This is the main local compute bottleneck.

**Current state:** All lenses run sequentially in `LensRunner.run_all()`.

**Optimization approach:**
- **Process-level parallelism** via `concurrent.futures.ProcessPoolExecutor` with `mp_context=multiprocessing.get_context("spawn")`
- **Critical: must use `spawn` start method** — `ProviderPool` loads PyTorch models and spaCy pipelines that are not fork-safe. CUDA contexts and spaCy internal state will corrupt across a `fork` boundary.
- Each worker process creates its own `ProviderPool` via an `initializer` function
- Limit to 2 GPU workers to avoid VRAM exhaustion (~750MB per worker for GPT-2 + sentence-transformer + spaCy)
- Pre-scan produces `(story_id, ProseMetrics, report_dict, text)` tuples written to `arena/scanned/{story_id}.json`

**Expected throughput:** ~2 stories/minute with 2 GPU workers (vs ~1/min current sequential).

### 4. Revision Runner

Takes a scanned story + critique config, runs the orchestrated revision loop.

**Key change:** `orchestrated_revise.py` currently reads baselines from `BASELINES` global. Refactor to accept a `CritiqueConfig` that overrides baselines, thresholds, and prescriptions.

**Concurrency:** 3 gpt-oss slots running revisions in parallel via `asyncio` + `httpx`. Each slot processes one (story, config) pair at a time.

**Pipeline:** For each match (story S, config A vs config B):
1. Generate critique for S using config A → prompt_A
2. Generate critique for S using config B → prompt_B
3. Submit revision(S, prompt_A) to gpt-oss slot → revised_A
4. Submit revision(S, prompt_B) to gpt-oss slot → revised_B
5. Submit judge(original, revised_A, revised_B) to minimax slot

Steps 3 and 4 run concurrently (2 of the 3 slots). Step 5 uses the dedicated judge slot. Matches are processed **one at a time** (no pipelining across matches) to keep state management simple.

**Revised throughput estimate:** Each revision turn is LLM call (~20s) + metrics-only rescan (~15-30s) = ~35-50s per turn. At 8 turns, that's ~5-7 minutes per story per config. Two configs per match = ~10-14 min per match (concurrent), plus ~15s judging. **Net: ~5-6 matches/hour.**

A 50-story round with 5 configs (round-robin, 10 pairings × 50 stories = 500 matches but each story needs only 5 revision pairs) ≈ 50 stories × 10 config-pairs / 2 concurrent = **250 revision jobs at ~6/hr ≈ 40-42 hours.** This is too slow for a single run.

**Practical scaling:** For the first tournament, use 20 stories × 5 configs (round-robin) = 100 revision pairs = ~17 hours overnight. Or reduce to 3 configs × 30 stories = 90 pairs = ~15 hours.

### 5. Minimax Judge

Blind pairwise comparison. The judge sees:
- The original text
- "Version X" and "Version Y" (randomized assignment to avoid position bias)
- A structured prompt asking for: winner (X/Y/tie), 2-3 sentence reasoning

**Judge prompt template:**
```
You are a literary editor comparing two revisions of the same passage.
Read the original, then both revisions. Which revision is better prose?

Judge on: sentence variety, interiority, sensory grounding, dialogue
naturalness, rhythm, and overall craft. Ignore plot — focus on technique.

Original:
{original}

Version X:
{version_x}

Version Y:
{version_y}

Respond in JSON:
{"winner": "X" | "Y" | "tie", "reason": "..."}
```

**Position debiasing:** Randomly assign config A/B to X/Y positions. Log the mapping for analysis.

### 6. ELO Tracker

Append-only JSONL log + in-memory rating state.

**Match record:**
```json
{
  "match_id": "m_001",
  "story_id": "423f44f581fe",
  "config_a": "baseline_v1",
  "config_b": "fragment_relaxed_v1",
  "winner": "config_b",
  "judge_reason": "Version Y preserved fragment rhythm while fixing inversion...",
  "position_map": {"X": "config_a", "Y": "config_b"},
  "config_a_metrics_before": {"total_distance": 4.11},
  "config_a_metrics_after": {"total_distance": 3.61},
  "config_b_metrics_before": {"total_distance": 4.11},
  "config_b_metrics_after": {"total_distance": 3.45},
  "per_metric_winner": {"fg_inversion": "config_b", "pd_mean": "config_a"},
  "timestamp": "2026-03-21T14:30:00Z"
}
```

**ELO calculation:** Standard ELO with K=32. Ties split the points. No confidence weighting — let the judge's binary decision stand, track reasoning for analysis only.

**Per-metric win rates:** Computed inline after each match. For each metric, track which config produced a larger improvement. Stored in the match record and aggregated in `ratings.json`.

**Rating file:** `arena/ratings.json` — updated after each match:
```json
{
  "configs": {
    "baseline_v1": {"elo": 1523, "matches": 40, "wins": 22, "losses": 15, "ties": 3},
    "fragment_relaxed_v1": {"elo": 1477, "matches": 40, "wins": 15, "losses": 22, "ties": 3}
  },
  "per_metric_win_rates": {
    "fg_fragment": {"baseline_v1": 0.45, "fragment_relaxed_v1": 0.55},
    "fg_inversion": {"baseline_v1": 0.60, "fragment_relaxed_v1": 0.40}
  }
}
```

### 7. Arena Runner (CLI)

New CLI command: `prose-doctor arena`

```bash
# Run a tournament round
prose-doctor arena run \
  --configs arena/configs/*.yaml \
  --stories 20 \
  --endpoint localhost:8081 \
  --judge-model minimax \
  --revision-model gpt-oss

# Show current ratings
prose-doctor arena ratings

# Export winning config as new defaults
prose-doctor arena export --config baseline_v1
```

**`arena run` flow:**
1. Load all configs from the configs directory
2. Pre-flight check: verify gpt-oss and minimax are reachable, check slot count
3. Sample N stories from Novelist dataset (respecting held-out set)
4. Pre-scan all stories (parallel, 2 GPU workers, `spawn` context)
5. Generate match schedule: **round-robin** for N <= 6 configs (simple, sufficient)
6. Execute matches: revise + judge (3 gpt-oss + 1 minimax concurrent)
7. Update ELO + per-metric win rates after each match
8. Print leaderboard at end

### 8. Error Handling

The arena runs unattended for hours. Explicit policies:

| Error | Policy |
|-------|--------|
| gpt-oss timeout (>120s) | Retry once, then skip this revision turn (proceed with fewer turns) |
| gpt-oss returns empty/garbage | Skip turn, log warning |
| minimax judge returns malformed JSON | Retry with temperature=0, then record as tie |
| Scan crash on degenerate story | Skip story, log, sample a replacement |
| Revision produces >50% length change | Reject edit (existing regression_limit handles this) |
| Story has <5 paragraphs or is >90% dialogue | Filter during sampling, not at runtime |

All errors logged to `arena/results/errors.jsonl` with story_id, config, error type, and timestamp.

### 9. Config Mutation (Phase 3 — deferred)

For evolutionary rounds after the initial manual tournament. Simplified to baseline/threshold shifts only — no prescription rewriting in v1.

**Mutation operators:**
- **Baseline shift:** Perturb one baseline value by +/-10% of its current value, clamped to observed human range
- **Threshold shift:** Perturb one issue finder threshold by +/-1 step (e.g., trigger 3 → 2 or 4)

**Constraints:**
- Baselines must stay within observed human range (from the 7-novel corpus)
- Each mutation produces a new config file with `parent` field set

## Parallelism Budget

| Resource | Slots | Assignment |
|----------|-------|------------|
| gpt-oss | 4 | 3 revision, 1 minimax judge |
| GPU (RTX 4080) | 1 | 2 scan workers (shared, spawn context) |
| CPU | 16+ | scan workers, config gen, ELO updates |

**Realistic throughput:**
- Scan: ~2 stories/min (GPU bottleneck)
- Revision: ~5-7 min per story per config (8 turns × ~40s including rescan)
- Judge: ~15s per comparison
- **Net: ~5-6 matches/hour** with concurrent revision slots
- 20 stories × 3 configs (round-robin, 3 pairings) = 60 matches ≈ **10-12 hours**
- 20 stories × 5 configs (round-robin, 10 pairings) = 200 matches ≈ **33-40 hours**

**Recommendation:** Start with 3 configs × 20 stories for the first overnight run. Add configs based on per-metric analysis.

## File Layout

```
arena/
  configs/           # critique config YAML files
    default.yaml     # current hardcoded values
    fragment_relaxed.yaml
    inversion_lower.yaml
    ...
  scanned/           # pre-scanned story cache
    {story_id}.json
  holdout/           # held-out validation stories (never used in matches)
    story_ids.txt
  results/           # match results
    matches.jsonl    # append-only match log
    ratings.json     # current ELO ratings
    errors.jsonl     # error log
  reports/           # per-round summary reports
    round_001.md
```

## Implementation Phases

**Phase 1: Config extraction + parameterized critique**
- Define `CritiqueConfig` dataclass with all tunable parameters (see Section 1)
- Consolidate the two divergent `BASELINES` dicts (agent_models.py + critique.py) into single source
- Thread `CritiqueConfig` through: `ProseMetrics._metric_distance()`, `build_critique()`, all `find_*_issues()` functions, `scan_deep()`, `orchestrated_revise()`
- Extract all hardcoded thresholds from issue finders (pd_baseline_margin, inversion_pct_gate, concrete_para_mean_threshold, spike_surprisal_margin, etc.)
- Write `configs/default.yaml` matching current hardcoded values
- `CritiqueConfig.from_yaml()` and `.to_yaml()` serialization
- Tests: existing tests pass with default config; test that loading default.yaml produces identical behavior

**Phase 2: Arena core — scan pool + revision runner + judge**
- Story sampler: chapter extraction (regex for markdown headings), genre stratification, min total_distance filter, held-out set reservation
- Parallel scan worker pool (`ProcessPoolExecutor`, `spawn` context, 2 workers)
- Async revision runner (3 concurrent gpt-oss slots via `asyncio` + `httpx`)
- Minimax judge with position debiasing
- ELO tracker with JSONL logging + per-metric win rates (computed inline)
- Error handling (retry/skip/log policies per table above)
- `prose-doctor arena run` and `prose-doctor arena ratings` CLI commands

**Phase 3: Config mutation + evolutionary rounds (deferred)**
- Mutation operators (baseline shift, threshold shift only)
- `prose-doctor arena mutate` command
- Swiss pairing (if scaling past 6 configs)
- `prose-doctor arena export` to promote winner to defaults

## Dependencies

- `httpx` (async HTTP for concurrent LLM calls)
- `pyyaml` (config serialization)
- No new ML models or heavy dependencies
- minimax model accessible on same endpoint as gpt-oss (localhost:8081)

## Success Criteria

- Running a 20-story tournament produces a config that the judge prefers over the default in >60% of head-to-head matches
- The winning config's baselines remain within the observed human range
- Total distance on the 50-story held-out set decreases by >10% vs default config
- No metric regresses past the human floor (we don't over-optimize one axis at the expense of others)
- Per-metric win rates identify which parameters are load-bearing vs noise
