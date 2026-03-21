# Lens Architecture for prose-doctor

## Goal

Reorganize prose-doctor's analyzers into **lens modules** — self-contained analytical perspectives with a common interface, explicit dependencies, and data-driven promotion through maturity tiers. Bump to version 0.2.0. Clean break from the flat `ml/` layout, no backwards compatibility shims.

## Architecture Overview

Three layers:

- **Providers** — shared resources (spaCy, GPT-2, sentence-transformers, LLM client). Lazy-loaded, each instantiated once per session.
- **Lenses** — analytical modules. Each answers one question about prose. Declares its provider dependencies and any lens-to-lens data dependencies. Returns a standard `LensResult` at multiple resolutions.
- **Validation** — test harness that runs lenses against a human/LLM corpus, computes discrimination statistics, and manages tier promotion.

```
Providers (shared resources)
    ↑ used by
Lenses (analytical modules)
    ↑ consumed by
Meta-lenses (e.g., narrative_attention — analyzes other lenses' outputs)
    ↑ checked by
Validation (corpus-based promotion)
```

## Directory Structure

```
src/prose_doctor/
  lenses/
    __init__.py              # Lens ABC, LensResult, LensRegistry
    psychic_distance.py
    info_contour.py
    foregrounding.py
    sensory.py
    dialogue_voice.py
    pacing.py
    emotion_arc.py
    slop_classifier.py
    boyd_narrative_role.py
    narrative_attention.py   # meta-lens
    uncertainty_reduction.py
    fragment_classifier.py
  providers/
    __init__.py              # ProviderPool
    spacy.py
    gpt2.py
    sentence_transformer.py
    llm.py                   # OpenAI-compatible client for retexture/uncertainty
  validation/
    __init__.py
    corpus.py                # human/LLM corpus management
    discriminator.py         # runs lenses on corpus, computes stats
    promotion.py             # tier checks, promotion logic
    tiers.toml               # current tier assignments + evidence
  analyzers/                 # rule-based (unchanged): proof_scanner, density, vocabulary
    __init__.py
    doctor.py
    proof_scanner.py
    density.py
    vocabulary.py
  # Existing (rewired to use lenses/)
  critique.py
  retexture.py
  agent.py
  agent_models.py
  agent_scan.py
  agent_issues.py
  orchestrated_revise.py
  cli.py
  config.py
  text.py
```

The `ml/` directory is deleted entirely. Version bumps to 0.2.0.

## Lens Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LensResult:
    """Standard output from any lens."""
    lens_name: str

    # Multi-resolution scores. Each lens populates the resolutions
    # it naturally operates at. None = this lens doesn't produce
    # scores at this resolution.
    per_sentence: dict[str, list[float]] | None = None
    per_paragraph: dict[str, list[float]] | None = None
    per_scene: dict[str, list[float]] | None = None
    per_chapter: dict[str, float] | None = None

    # Actionable findings for the revision agent's find_issues
    issues: list = field(default_factory=list)

    # Lens-specific detail (per-speaker data, zoom jumps, flatline
    # locations, etc.) — anything that doesn't fit the standard
    # resolutions goes here
    raw: dict = field(default_factory=dict)


class Lens(ABC):
    """Base class for all analytical lenses."""

    name: str                              # e.g., "psychic_distance"
    requires_providers: list[str] = []     # e.g., ["spacy", "gpt2"]
    consumes_lenses: list[str] = []        # e.g., ["psychic_distance", "info_contour"]

    @abstractmethod
    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        """Run the lens on a chapter.

        Args:
            text: Full chapter text.
            filename: For reporting.
            providers: Shared resource pool (spaCy, GPT-2, etc.).
            consumed: Results from lenses listed in consumes_lenses.
                      Only populated for meta-lenses.

        Returns:
            LensResult with scores at appropriate resolutions.
        """
        ...
```

### Resolution semantics

| Resolution | Meaning | Example |
|------------|---------|---------|
| `per_sentence` | One value per sentence (indexed by sentence position in chapter) | Psychic distance score, surprisal |
| `per_paragraph` | One value per paragraph (indexed by paragraph position) | Mean psychic distance, fragment ratio |
| `per_scene` | One value per scene (indexed by scene number) | Scene coherence, Boyd dominant mode |
| `per_chapter` | One aggregate value for the whole chapter | Mean psychic distance, total fragment % |

Lenses fill the resolutions they compute naturally. Consumers that need a resolution the lens doesn't provide can aggregate from a finer resolution (e.g., mean of `per_sentence` → `per_paragraph`). The `LensRunner` provides a helper for this.

### Lens types

**Standard lenses** analyze text directly. They require providers but not other lenses' outputs.

**Meta-lenses** analyze other lenses' outputs. They declare `consumes_lenses` and receive those results via the `consumed` parameter. The `LensRunner` topologically sorts and runs dependencies first.

Only one meta-lens currently: `narrative_attention` (consumes psychic_distance, info_contour, foregrounding, emotion_arc, boyd_narrative_role, uncertainty_reduction).

## Provider Pool

```python
class ProviderPool:
    """Lazy-loading shared resource pool.

    Replaces ModelManager. Each provider is instantiated once on first
    access and reused for all subsequent requests.
    """

    def get(self, name: str) -> Any:
        """Get a provider by name. Lazy-loads on first access."""
        ...

    def get_spacy(self) -> "spacy.Language": ...
    def get_gpt2(self) -> tuple: ...  # (model, tokenizer)
    def get_sentence_transformer(self) -> "SentenceTransformer": ...
    def get_llm_client(self) -> "OpenAI": ...
```

Provider configuration (LLM endpoint, model name, API key) comes from `ProjectConfig` which reads `.prose-doctor.toml`. The pool is initialized once per CLI invocation or agent run.

### Provider dependencies per lens

| Lens | spacy | gpt2 | sentence_transformer | llm |
|------|-------|------|---------------------|-----|
| psychic_distance | x | | x (sensory probe) | |
| info_contour | x | x | | |
| foregrounding | x | x (collocations) | | |
| sensory | x | | x | |
| dialogue_voice | x | | x | |
| pacing | | | | |
| emotion_arc | | | | |*
| slop_classifier | | | x | |
| boyd_narrative_role | | | | |
| uncertainty_reduction | | x | | |
| fragment_classifier | x | | | |
| narrative_attention | x | x | x | |**

\* emotion_arc uses transformers pipeline directly (distilbert), not via provider
\** narrative_attention inherits provider needs from consumed lenses

## Lens Runner

```python
class LensRunner:
    """Executes lenses in dependency order."""

    def __init__(self, providers: ProviderPool, tier_filter: str = "stable"):
        """
        Args:
            providers: Shared resource pool.
            tier_filter: Minimum tier to include. "experimental" = all,
                        "validated" = validated + stable, "stable" = stable only.
        """
        ...

    def run_all(self, text: str, filename: str) -> dict[str, LensResult]:
        """Run all eligible lenses in topological order."""
        ...

    def run_one(self, lens_name: str, text: str, filename: str) -> LensResult:
        """Run a single lens (plus any consumed dependencies)."""
        ...
```

The runner:
1. Filters lenses by tier
2. Topologically sorts by `consumes_lenses` dependencies
3. Runs each lens, passing provider pool and consumed results
4. Returns all results keyed by lens name

## Validation Framework

### Corpus

```
corpus/
  human/              # published human prose (Watts, Doctorow, author chapters)
  llm/                # LLM-generated prose (DeepSeek, MiniMax drafts)
```

Paths configurable in `.prose-doctor.toml`:
```toml
[prose-doctor.validation]
human_corpus = "corpus/human"
llm_corpus = "corpus/llm"
```

### Discriminator

For each lens, runs it on both corpora and computes per-metric discrimination statistics:

- **Cohen's d** — effect size (how separated are the distributions?)
- **Mann-Whitney U** — non-parametric test (are they from different distributions?)
- **KS statistic** — distribution shape difference

Uses `per_chapter` summary metrics for discrimination. If a lens doesn't produce `per_chapter`, aggregates from the finest available resolution.

### Promotion Tiers

| Tier | Criteria | What it unlocks |
|------|----------|----------------|
| `experimental` | Lens runs without errors on the test corpus | Available with `--experimental` flag |
| `validated` | At least one metric with Cohen's d > 0.5 OR p < 0.01 | Available with `--validated` flag, usable in critique |
| `stable` | Validated + at least 3 accepted revision edits guided by this lens's issues across 2+ chapters, with no metric regression | Default in CLI and agent, used in find_issues |

### tiers.toml

```toml
# Auto-generated by `prose-doctor validate`. Do not edit manually.

[psychic_distance]
tier = "stable"
last_validated = "2026-03-20"
metrics.pd_mean.cohens_d = 1.78
metrics.pd_mean.p_value = 0.0001
metrics.pd_std.cohens_d = 0.92
metrics.pd_std.p_value = 0.003
revision_evidence = ["b1ch01:3_accepted", "b2ch06:2_accepted"]

[uncertainty_reduction]
tier = "experimental"
last_validated = "2026-03-20"
metrics.mean_reduction.cohens_d = 0.31
metrics.mean_reduction.p_value = 0.08
```

### CLI

```bash
prose-doctor validate psychic_distance    # run one lens against corpus
prose-doctor validate --all               # run all lenses, print tier table
prose-doctor validate --promote           # auto-promote lenses that cross thresholds

# Tier filtering in existing commands
prose-doctor scan --deep chapter.md                 # stable lenses only (default)
prose-doctor scan --deep --validated chapter.md      # validated + stable
prose-doctor scan --deep --experimental chapter.md   # all lenses

prose-doctor revise chapter.md                       # stable lenses only
prose-doctor revise --experimental chapter.md        # all lenses in find_issues
```

## Migration Plan

Each lens is migrated independently, test suite passing after each:

1. Create `lenses/__init__.py` with `Lens`, `LensResult`, `LensRegistry`
2. Create `providers/__init__.py` with `ProviderPool` (wrapping ModelManager logic)
3. Migrate lenses one at a time: move logic from `ml/<name>.py` → `lenses/<name>.py`, wrap in `Lens` interface
4. Rewire `agent_scan.py` to use `LensRunner` instead of direct ML imports
5. Rewire `agent_issues.py` find_issues functions to use `LensResult`
6. Rewire `cli.py` scan/critique/revise commands
7. Rewire `critique.py` to consume `LensResult` dicts
8. Create `validation/` framework
9. Delete `ml/` directory
10. Bump version to 0.2.0

### Lens migration order (by dependency)

1. **No dependencies**: pacing, boyd_narrative_role, fragment_classifier, emotion_arc
2. **spacy only**: foregrounding, dialogue_voice
3. **spacy + gpt2**: info_contour, uncertainty_reduction
4. **spacy + sentence_transformer**: psychic_distance, sensory, slop_classifier
5. **meta-lens**: narrative_attention (last — consumes all above)

## What stays outside the lens system

- `analyzers/` (proof_scanner, density, vocabulary) — deterministic pattern matching, not statistical lenses. Critique merges both streams.
- `retexture.py` — generation tool, not analysis.
- `agent.py`, `orchestrated_revise.py` — consumers of lenses, not lenses themselves.
- `text.py`, `config.py` — utilities.

## Expected initial tier assignments

| Lens | Expected tier | Evidence |
|------|--------------|---------|
| psychic_distance | stable | d=1.78, baselines validated, find_issues wired, revision edits accepted |
| info_contour | stable | d=-1.52 (rhythmicity), baselines validated, find_issues wired |
| foregrounding | stable | d=1.34 (inversion), baselines validated, find_issues wired, revision edits accepted |
| sensory | validated | Modality scoring works, deserts detected, no revision evidence yet |
| dialogue_voice | validated | Speaker separation works, talking heads detected |
| pacing | experimental | Interiority detection broken (zero hits), needs threshold rework |
| emotion_arc | experimental | Orthogonal feature (max|r|=0.19), but distilbert sentiment is crude |
| slop_classifier | stable | Trained on 13K samples, density budgets validated |
| boyd_narrative_role | validated | Orthogonal, ANOVA confirms scene differentiation (F=4.39**) |
| narrative_attention | validated | Human/LLM discrimination p<0.0001, generic decomposition working |
| uncertainty_reduction | experimental | Signal exists but not wired into prescriptions, d=0.31 |
| fragment_classifier | stable | Crutch/craft classification validated on b1ch01, revision edits accepted |
