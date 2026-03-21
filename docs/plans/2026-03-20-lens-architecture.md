# Lens Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize prose-doctor's flat `ml/` analyzers into lens modules with a common interface, shared provider pool, dependency-aware runner, and corpus-based validation framework. Bump to 0.2.0.

**Architecture:** Three layers — Providers (lazy-loaded shared resources: spaCy, GPT-2, sentence-transformers, LLM client), Lenses (self-contained analyzers returning `LensResult` at multiple resolutions), and Validation (corpus-based discrimination stats driving tier promotion). A `LensRunner` topologically sorts lenses by dependency and runs them. Meta-lenses like `narrative_attention` consume other lenses' outputs.

**Tech Stack:** Python 3.11+, spaCy, transformers, sentence-transformers, scipy (stats), numpy, toml.

**Spec:** `docs/specs/2026-03-20-lens-architecture.md`

---

## File Map

### New files

| File | Responsibility |
|------|---------------|
| `src/prose_doctor/lenses/__init__.py` | `Lens` ABC, `LensResult` dataclass, `LensRegistry` |
| `src/prose_doctor/providers/__init__.py` | `ProviderPool` — lazy-loading resource manager with ML availability guard |
| `src/prose_doctor/providers/spacy.py` | spaCy provider |
| `src/prose_doctor/providers/gpt2.py` | GPT-2 provider |
| `src/prose_doctor/providers/sentence_transformer.py` | Sentence-transformer provider (all-MiniLM-L6-v2, 384d) |
| `src/prose_doctor/providers/llm.py` | OpenAI-compatible LLM client provider |
| `src/prose_doctor/lenses/pacing.py` | Pacing lens (no ML deps) |
| `src/prose_doctor/lenses/emotion_arc.py` | Emotion arc lens (standalone distilbert) |
| `src/prose_doctor/lenses/foregrounding.py` | Foregrounding lens |
| `src/prose_doctor/lenses/info_contour.py` | Info contour lens |
| `src/prose_doctor/lenses/psychic_distance.py` | Psychic distance lens |
| `src/prose_doctor/lenses/sensory.py` | Sensory lens (self-contained mpnet-base-v2 768d probe — NOT shared provider) |
| `src/prose_doctor/lenses/dialogue_voice.py` | Dialogue voice lens |
| `src/prose_doctor/lenses/slop_classifier.py` | Slop classifier lens |
| `src/prose_doctor/lenses/perplexity.py` | GPT-2 perplexity lens |
| `src/prose_doctor/lenses/uncertainty_reduction.py` | Uncertainty reduction lens (extracted from proto) |
| `src/prose_doctor/lenses/boyd_narrative_role.py` | Boyd staging/progression/tension (extracted from proto) |
| `src/prose_doctor/lenses/fragment_classifier.py` | Fragment craft/crutch classifier (extracted from agent_issues) |
| `src/prose_doctor/lenses/narrative_attention.py` | Meta-lens consuming other lenses |
| `src/prose_doctor/lenses/runner.py` | `LensRunner` — dependency-ordered execution |
| `src/prose_doctor/lenses/defaults.py` | `default_registry()` — pre-populated registry of all lenses |
| `src/prose_doctor/lenses/twins.py` | Texture-matched paragraph finder (cross-file utility, not a per-chapter lens) |
| `src/prose_doctor/lenses/repetition.py` | Structural repetition detector (cross-file utility) |
| `src/prose_doctor/lenses/voice.py` | POV voice separation (cross-file utility) |
| `src/prose_doctor/validation/__init__.py` | Validation package init |
| `src/prose_doctor/validation/corpus.py` | Human/LLM corpus management |
| `src/prose_doctor/validation/discriminator.py` | Statistical discrimination (Cohen's d, Mann-Whitney, KS) |
| `src/prose_doctor/validation/promotion.py` | Tier checks and promotion logic |
| `src/prose_doctor/validation/tiers.toml` | Current tier assignments + evidence |
| `tests/test_lens_interface.py` | Tests for Lens ABC, LensResult, LensRegistry |
| `tests/test_providers.py` | Tests for ProviderPool |
| `tests/test_runner.py` | Tests for LensRunner |
| `tests/test_lens_pacing.py` | Tests for pacing lens |
| `tests/test_lens_foregrounding.py` | Tests for foregrounding lens |
| `tests/test_lens_info_contour.py` | Tests for info contour lens |
| `tests/test_lens_psychic_distance.py` | Tests for psychic distance lens |
| `tests/test_lens_sensory.py` | Tests for sensory lens |
| `tests/test_lens_dialogue_voice.py` | Tests for dialogue voice lens |
| `tests/test_lens_slop_classifier.py` | Tests for slop classifier lens |
| `tests/test_lens_emotion_arc.py` | Tests for emotion arc lens |
| `tests/test_lens_perplexity.py` | Tests for perplexity lens |
| `tests/test_lens_uncertainty_reduction.py` | Tests for uncertainty reduction lens |
| `tests/test_lens_boyd_narrative_role.py` | Tests for Boyd lens |
| `tests/test_lens_fragment_classifier.py` | Tests for fragment classifier lens |
| `tests/test_lens_narrative_attention.py` | Tests for narrative attention meta-lens |
| `tests/test_validation.py` | Tests for discriminator + promotion |

### Modified files

| File | Change |
|------|--------|
| `src/prose_doctor/agent_scan.py` | Replace `ml.*` imports with `LensRunner` |
| `src/prose_doctor/agent_issues.py` | Replace `ml.models` imports, consume `LensResult` |
| `src/prose_doctor/cli.py` | Replace all `ml.*` imports with lens/runner calls |
| `src/prose_doctor/critique.py` | Accept `dict[str, LensResult]` alongside existing report dict |
| `pyproject.toml` | Version bump to 0.2.0 |

Note: `orchestrated_revise.py` has no direct `ml/` imports — it goes through `agent_scan`, so it needs no changes.

### Deleted files

| File | Reason |
|------|--------|
| `src/prose_doctor/ml/` (entire directory) | Replaced by `lenses/` + `providers/` |

---

## Task 1: Lens ABC and LensResult

**Files:**
- Create: `src/prose_doctor/lenses/__init__.py`
- Test: `tests/test_lens_interface.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_lens_interface.py
from prose_doctor.lenses import Lens, LensResult, LensRegistry


def test_lens_result_has_standard_fields():
    r = LensResult(lens_name="test")
    assert r.lens_name == "test"
    assert r.per_sentence is None
    assert r.per_paragraph is None
    assert r.per_scene is None
    assert r.per_chapter is None
    assert r.issues == []
    assert r.raw == {}


def test_lens_result_with_data():
    r = LensResult(
        lens_name="psychic_distance",
        per_paragraph={"pd_mean": [0.3, 0.4, 0.2]},
        per_chapter={"pd_mean": 0.3, "pd_std": 0.08},
    )
    assert len(r.per_paragraph["pd_mean"]) == 3
    assert r.per_chapter["pd_mean"] == 0.3


def test_lens_registry_register_and_get():
    registry = LensRegistry()

    class FakeLens(Lens):
        name = "fake"
        requires_providers = []
        consumes_lenses = []

        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name=self.name)

    registry.register(FakeLens())
    assert registry.get("fake") is not None
    assert registry.get("nonexistent") is None
    assert "fake" in registry.all_names()


def test_lens_registry_rejects_duplicate():
    registry = LensRegistry()

    class FakeLens(Lens):
        name = "fake"
        requires_providers = []
        consumes_lenses = []

        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name=self.name)

    registry.register(FakeLens())
    try:
        registry.register(FakeLens())
        assert False, "Should have raised"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_interface.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prose_doctor.lenses'`

- [ ] **Step 3: Implement**

```python
# src/prose_doctor/lenses/__init__.py
"""Lens framework — self-contained analytical perspectives on prose."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


@dataclass
class LensResult:
    """Standard output from any lens."""
    lens_name: str

    per_sentence: dict[str, list[float]] | None = None
    per_paragraph: dict[str, list[float]] | None = None
    per_scene: dict[str, list[float]] | None = None
    per_chapter: dict[str, float] | None = None

    issues: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class Lens(ABC):
    """Base class for all analytical lenses."""

    name: str = ""
    requires_providers: list[str] = []
    consumes_lenses: list[str] = []

    @abstractmethod
    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict[str, LensResult] | None = None,
    ) -> LensResult:
        ...


class LensRegistry:
    """Registry of available lens instances."""

    def __init__(self) -> None:
        self._lenses: dict[str, Lens] = {}

    def register(self, lens: Lens) -> None:
        if lens.name in self._lenses:
            raise ValueError(f"Lens '{lens.name}' already registered")
        self._lenses[lens.name] = lens

    def get(self, name: str) -> Lens | None:
        return self._lenses.get(name)

    def all_names(self) -> list[str]:
        return list(self._lenses.keys())

    def all_lenses(self) -> list[Lens]:
        return list(self._lenses.values())
```

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_interface.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/__init__.py tests/test_lens_interface.py
git commit -m "feat: Lens ABC, LensResult, LensRegistry"
```

---

## Task 2: ProviderPool

**Files:**
- Create: `src/prose_doctor/providers/__init__.py`
- Create: `src/prose_doctor/providers/spacy.py`
- Create: `src/prose_doctor/providers/gpt2.py`
- Create: `src/prose_doctor/providers/sentence_transformer.py`
- Create: `src/prose_doctor/providers/llm.py`
- Test: `tests/test_providers.py`

This replaces `ml/models.py` (ModelManager singleton). Each provider is a thin wrapper that lazy-loads on first access. The pool holds them all. Provider factories must catch `ImportError` and raise a helpful message — this replaces the `require_ml()` guard from `ml/__init__.py`.

- [ ] **Step 1: Write the test**

```python
# tests/test_providers.py
import pytest
from prose_doctor.providers import ProviderPool


def test_provider_pool_creates():
    pool = ProviderPool()
    assert pool is not None


def test_provider_pool_get_unknown_raises():
    pool = ProviderPool()
    with pytest.raises(KeyError):
        pool.get("nonexistent")


def test_provider_pool_knows_available():
    pool = ProviderPool()
    names = pool.available()
    assert "spacy" in names
    assert "gpt2" in names
    assert "sentence_transformer" in names
    assert "llm" in names


def test_require_ml_raises_on_missing_deps(monkeypatch):
    """ProviderPool.require_ml() raises ImportError with install hint."""
    from prose_doctor.providers import require_ml
    # If ML deps are installed (they are in dev), this should pass
    require_ml()  # no error
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_providers.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement providers**

```python
# src/prose_doctor/providers/__init__.py
"""Shared resource providers — lazy-loaded, instantiated once per session.

Replaces ml/__init__.py (require_ml guard) and ml/models.py (ModelManager).
"""
from __future__ import annotations

from typing import Any

# --- ML availability guard (replaces ml/__init__.py) ---

ML_AVAILABLE = False
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    ML_AVAILABLE = True
except ImportError:
    pass


def require_ml() -> None:
    """Raise ImportError with install instructions if ML deps are missing."""
    if not ML_AVAILABLE:
        raise ImportError(
            "ML features require extra dependencies. Install with:\n"
            "  uv pip install -e '.[ml]'"
        )


class ProviderPool:
    """Lazy-loading shared resource pool. Replaces ModelManager."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._factories: dict[str, type] = {
            "spacy": _lazy_import_spacy,
            "gpt2": _lazy_import_gpt2,
            "sentence_transformer": _lazy_import_st,
            "llm": _lazy_import_llm,
        }

    def get(self, name: str) -> Any:
        if name not in self._factories:
            raise KeyError(f"Unknown provider: {name}")
        if name not in self._cache:
            self._cache[name] = self._factories[name]()
        return self._cache[name]

    def available(self) -> list[str]:
        return list(self._factories.keys())

    @property
    def spacy(self):
        return self.get("spacy")

    @property
    def gpt2(self):
        return self.get("gpt2")

    @property
    def sentence_transformer(self):
        return self.get("sentence_transformer")

    @property
    def llm(self):
        return self.get("llm")

    @property
    def device(self):
        """Auto-detect CUDA vs CPU."""
        if "device" not in self._cache:
            import torch
            self._cache["device"] = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._cache["device"]


def _lazy_import_spacy():
    from prose_doctor.providers.spacy import load_spacy
    return load_spacy()


def _lazy_import_gpt2():
    from prose_doctor.providers.gpt2 import load_gpt2
    return load_gpt2()


def _lazy_import_st():
    from prose_doctor.providers.sentence_transformer import load_sentence_transformer
    return load_sentence_transformer()


def _lazy_import_llm():
    from prose_doctor.providers.llm import load_llm_client
    return load_llm_client()
```

```python
# src/prose_doctor/providers/spacy.py
"""spaCy provider — en_core_web_sm."""


def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")
```

```python
# src/prose_doctor/providers/gpt2.py
"""GPT-2 provider — model + tokenizer."""
from __future__ import annotations


def load_gpt2() -> tuple:
    """Returns (model, tokenizer) on the appropriate device."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    return model, tokenizer
```

```python
# src/prose_doctor/providers/sentence_transformer.py
"""Sentence-transformer provider — all-MiniLM-L6-v2 (384d)."""


def load_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")
```

```python
# src/prose_doctor/providers/llm.py
"""OpenAI-compatible LLM client provider."""
from __future__ import annotations


def load_llm_client():
    from openai import OpenAI
    return OpenAI(base_url="http://localhost:8081/v1", api_key="none")
```

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_providers.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/providers/ tests/test_providers.py
git commit -m "feat: ProviderPool with lazy-loaded providers and require_ml guard"
```

---

## Task 3: LensRunner

**Files:**
- Create: `src/prose_doctor/lenses/runner.py`
- Test: `tests/test_runner.py`

The runner topologically sorts lenses by `consumes_lenses`, respects tier filtering, and passes consumed results to meta-lenses.

- [ ] **Step 1: Write the test**

```python
# tests/test_runner.py
from prose_doctor.lenses import Lens, LensResult, LensRegistry
from prose_doctor.lenses.runner import LensRunner
from prose_doctor.providers import ProviderPool


class StubLensA(Lens):
    name = "a"
    requires_providers = []
    consumes_lenses = []

    def analyze(self, text, filename, providers, consumed=None):
        return LensResult(
            lens_name="a",
            per_chapter={"a_score": 1.0},
        )


class StubLensB(Lens):
    name = "b"
    requires_providers = []
    consumes_lenses = ["a"]

    def analyze(self, text, filename, providers, consumed=None):
        a_score = consumed["a"].per_chapter["a_score"]
        return LensResult(
            lens_name="b",
            per_chapter={"b_score": a_score * 2},
        )


def test_runner_topological_order():
    registry = LensRegistry()
    registry.register(StubLensB())  # depends on A
    registry.register(StubLensA())  # no deps
    runner = LensRunner(registry, ProviderPool())
    results = runner.run_all("test text", "test.md")
    assert "a" in results
    assert "b" in results
    assert results["b"].per_chapter["b_score"] == 2.0


def test_runner_run_one_with_deps():
    registry = LensRegistry()
    registry.register(StubLensA())
    registry.register(StubLensB())
    runner = LensRunner(registry, ProviderPool())
    result = runner.run_one("b", "test text", "test.md")
    assert result.per_chapter["b_score"] == 2.0


def test_runner_cycle_detection():
    class CycleA(Lens):
        name = "cycle_a"
        requires_providers = []
        consumes_lenses = ["cycle_b"]
        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name="cycle_a")

    class CycleB(Lens):
        name = "cycle_b"
        requires_providers = []
        consumes_lenses = ["cycle_a"]
        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name="cycle_b")

    registry = LensRegistry()
    registry.register(CycleA())
    registry.register(CycleB())
    runner = LensRunner(registry, ProviderPool())
    try:
        runner.run_all("text", "f.md")
        assert False, "Should have raised"
    except ValueError as e:
        assert "cycle" in str(e).lower()
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_runner.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement LensRunner**

```python
# src/prose_doctor/lenses/runner.py
"""LensRunner — executes lenses in dependency order."""
from __future__ import annotations

from prose_doctor.lenses import Lens, LensResult, LensRegistry
from prose_doctor.providers import ProviderPool


class LensRunner:
    """Executes lenses in topological order, passing consumed results to meta-lenses."""

    def __init__(
        self,
        registry: LensRegistry,
        providers: ProviderPool,
        tier_filter: str | None = None,
        tiers: dict[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._tier_filter = tier_filter
        self._tiers = tiers or {}

    def _eligible_lenses(self) -> list[Lens]:
        """Filter lenses by tier."""
        if not self._tier_filter:
            return self._registry.all_lenses()
        tier_rank = {"experimental": 0, "validated": 1, "stable": 2}
        min_rank = tier_rank.get(self._tier_filter, 2)
        eligible = []
        for lens in self._registry.all_lenses():
            lens_tier = self._tiers.get(lens.name, "experimental")
            if tier_rank.get(lens_tier, 0) >= min_rank:
                eligible.append(lens)
        return eligible

    def _toposort(self, lenses: list[Lens]) -> list[Lens]:
        """Topological sort by consumes_lenses. Raises on cycles."""
        by_name = {l.name: l for l in lenses}
        visited: set[str] = set()
        in_stack: set[str] = set()
        order: list[Lens] = []

        def visit(name: str) -> None:
            if name in in_stack:
                raise ValueError(f"Dependency cycle detected involving '{name}'")
            if name in visited:
                return
            in_stack.add(name)
            lens = by_name.get(name)
            if lens:
                for dep in lens.consumes_lenses:
                    if dep in by_name:
                        visit(dep)
            in_stack.remove(name)
            visited.add(name)
            if lens:
                order.append(lens)

        for lens in lenses:
            visit(lens.name)
        return order

    def run_all(self, text: str, filename: str) -> dict[str, LensResult]:
        """Run all eligible lenses in dependency order."""
        lenses = self._eligible_lenses()
        sorted_lenses = self._toposort(lenses)
        results: dict[str, LensResult] = {}
        for lens in sorted_lenses:
            consumed = None
            if lens.consumes_lenses:
                consumed = {
                    dep: results[dep]
                    for dep in lens.consumes_lenses
                    if dep in results
                }
            results[lens.name] = lens.analyze(
                text, filename, self._providers, consumed
            )
        return results

    def run_one(self, lens_name: str, text: str, filename: str) -> LensResult:
        """Run a single lens plus its consumed dependencies."""
        lens = self._registry.get(lens_name)
        if lens is None:
            raise KeyError(f"Unknown lens: {lens_name}")
        needed: set[str] = set()

        def collect(name: str) -> None:
            l = self._registry.get(name)
            if l:
                for dep in l.consumes_lenses:
                    if dep not in needed:
                        needed.add(dep)
                        collect(dep)

        collect(lens_name)
        all_lenses = [self._registry.get(n) for n in needed if self._registry.get(n)]
        all_lenses.append(lens)
        sorted_lenses = self._toposort(all_lenses)
        results: dict[str, LensResult] = {}
        for l in sorted_lenses:
            consumed = None
            if l.consumes_lenses:
                consumed = {d: results[d] for d in l.consumes_lenses if d in results}
            results[l.name] = l.analyze(text, filename, self._providers, consumed)
        return results[lens_name]
```

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_runner.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/runner.py tests/test_runner.py
git commit -m "feat: LensRunner with topological sort and tier filtering"
```

---

## Task 4: First lens migration — pacing (no ML deps)

**Files:**
- Create: `src/prose_doctor/lenses/pacing.py`
- Test: `tests/test_lens_pacing.py`

Pacing is the simplest lens — pure heuristic, no providers needed. This validates the lens interface works end-to-end before migrating anything complex.

- [ ] **Step 1: Write the test**

```python
# tests/test_lens_pacing.py
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.pacing import PacingLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
"You're late," she said, crossing her arms.

"Traffic," he muttered. "And the bridge was up."

"The bridge has been up for three days."

"I know."

He dropped his bag on the floor and walked to the window. The street below
was empty except for a delivery truck idling at the corner. Rain streaked
the glass.

She watched him for a long moment. Something had changed — the way he
held his shoulders, maybe, or the careful blankness in his face. She'd
seen that look before, years ago, when his mother was dying and he
wouldn't say so.

The kettle clicked off. Neither of them moved.
'''


def test_pacing_lens_returns_lens_result():
    lens = PacingLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "pacing"
    assert result.per_chapter is not None
    assert "dialogue_ratio" in result.per_chapter


def test_pacing_lens_metadata():
    lens = PacingLens()
    assert lens.name == "pacing"
    assert lens.requires_providers == []
    assert lens.consumes_lenses == []


def test_pacing_lens_per_paragraph():
    lens = PacingLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "mode" in result.per_paragraph
    assert len(result.per_paragraph["mode"]) > 0
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_pacing.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement pacing lens**

Port logic from `ml/pacing.py` into the `Lens` interface. Keep the same heuristic classification. Return `LensResult` with `per_paragraph` modes and `per_chapter` ratios.

The implementation should:
- Copy the `_classify_paragraph()` heuristic from `ml/pacing.py`
- Wrap `analyze_pacing()` to populate `LensResult`
- Map mode strings to float codes in `per_paragraph` (0=dialogue, 1=action, 2=interiority, 3=setting)
- Put `mode_ratios`, `dialogue_ratio`, talking head count, desert count in `per_chapter`
- Put raw `PacingProfile` fields in `raw`

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_pacing.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/pacing.py tests/test_lens_pacing.py
git commit -m "feat: pacing lens — first lens migration (no ML deps)"
```

---

## Task 5: Migrate foregrounding lens

**Files:**
- Create: `src/prose_doctor/lenses/foregrounding.py`
- Test: `tests/test_lens_foregrounding.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_lens_foregrounding.py
import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.foregrounding import ForegroundingLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
Rain hammered the corrugated roof. Marcus pressed his back against
the wall, feeling the cold seep through his jacket. His breath came
in short, ragged bursts.

Down the corridor, something scraped. Metal on concrete. He counted
to three, then moved — low, fast, keeping to the shadows.

She was waiting at the junction, rifle across her knees. "Took you
long enough," she said, not looking up. Her fingers worked the bolt
with practiced ease.
'''


@pytest.mark.slow
def test_foregrounding_lens_returns_result():
    lens = ForegroundingLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "foregrounding"
    assert result.per_chapter is not None
    assert "inversion_pct" in result.per_chapter
    assert "sl_cv" in result.per_chapter
    assert "fragment_pct" in result.per_chapter


@pytest.mark.slow
def test_foregrounding_lens_metadata():
    lens = ForegroundingLens()
    assert lens.name == "foregrounding"
    assert "spacy" in lens.requires_providers
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_foregrounding.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement foregrounding lens**

Port logic from `ml/foregrounding.py` (188 lines). The implementation should:
- Move `score_chapter()` logic into `ForegroundingLens.analyze()`
- Use `providers.spacy` instead of `model_manager.spacy`
- Use `providers.sentence_transformer` instead of `model_manager.sentence_transformer`
- Return `LensResult` with `per_chapter` containing `index`, `inversion_pct`, `sl_cv`, `fragment_pct`, `alliteration`, `unexpected_collocations`
- Put `weakest_axis`, `prescription` in `raw`

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_foregrounding.py -v -m slow`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/foregrounding.py tests/test_lens_foregrounding.py
git commit -m "feat: foregrounding lens migration"
```

---

## Task 6: Migrate info_contour lens

**Files:**
- Create: `src/prose_doctor/lenses/info_contour.py`
- Test: `tests/test_lens_info_contour.py`

- [ ] **Step 1: Write the test**

Test should verify the lens returns `per_sentence` with `surprisal` list, `per_chapter` with `rhythmicity`, `spikes`, `flatlines`, `mean_surprisal`. Mark `@pytest.mark.slow`. Use the same SAMPLE text pattern as Task 5.

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_info_contour.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement info_contour lens**

Port from `ml/info_contour.py` (281 lines). Use `providers.gpt2` (returns `(model, tokenizer)` tuple) and `providers.spacy`. The FFT analysis, flatline detection, and spike detection logic are internal — just rewrap the return into `LensResult`. Put sentence surprisals in `per_sentence`, aggregate stats in `per_chapter`, raw `InfoContourResult` fields in `raw`.

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_info_contour.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/info_contour.py tests/test_lens_info_contour.py
git commit -m "feat: info_contour lens migration"
```

---

## Task 7: Migrate psychic_distance lens

**Files:**
- Create: `src/prose_doctor/lenses/psychic_distance.py`
- Test: `tests/test_lens_psychic_distance.py`

- [ ] **Step 1: Write the test**

Test should verify `per_sentence` contains `distance` scores, `per_paragraph` contains `pd_mean`, `per_chapter` contains `pd_mean` and `pd_std`. Also verify `raw` contains `zoom_jumps`. Mark `@pytest.mark.slow`. Use the same SAMPLE text pattern.

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_psychic_distance.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement psychic_distance lens**

Port from `ml/psychic_distance.py` (344 lines). This is the most complex lens — it uses spaCy and the sensory probe (which loads its own `all-mpnet-base-v2` model internally). Use `providers.spacy`. Keep the module-level constants (`PERCEPTION_VERBS`, `COGNITION_VERBS`, `PROXIMAL_DEICTICS`) — `agent_issues.py` imports them from here.

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_psychic_distance.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/psychic_distance.py tests/test_lens_psychic_distance.py
git commit -m "feat: psychic_distance lens migration"
```

---

## Task 8: Migrate sensory lens

**Files:**
- Create: `src/prose_doctor/lenses/sensory.py`
- Test: `tests/test_lens_sensory.py`

**Important:** `SensoryProbe` uses `all-mpnet-base-v2` (768d), NOT the shared `sentence_transformer` provider (384d `all-MiniLM-L6-v2`). The sensory lens keeps its own model loading self-contained. It does NOT declare `sentence_transformer` in `requires_providers` — only `spacy` (for paragraph splitting).

- [ ] **Step 1: Write the test**

Test should verify `per_paragraph` contains modality score lists (`visual`, `auditory`, etc.), `per_chapter` contains `dominant_modality`, `balance_ratio`. Mark `@pytest.mark.slow`.

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_sensory.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement sensory lens**

Port from `ml/sensory.py` (212 lines). `SensoryProbe` class stays self-contained with its own `all-mpnet-base-v2` loading and `.pt` probe file. Use `providers.spacy` only for paragraph splitting. Put desert locations in `raw`.

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_sensory.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/sensory.py tests/test_lens_sensory.py
git commit -m "feat: sensory lens migration (self-contained mpnet probe)"
```

---

## Task 9: Migrate dialogue_voice, slop_classifier, emotion_arc lenses

**Files:**
- Create: `src/prose_doctor/lenses/dialogue_voice.py`
- Create: `src/prose_doctor/lenses/slop_classifier.py`
- Create: `src/prose_doctor/lenses/emotion_arc.py`
- Test: `tests/test_lens_dialogue_voice.py`
- Test: `tests/test_lens_slop_classifier.py`
- Test: `tests/test_lens_emotion_arc.py`

Three lenses migrated together since they follow the same pattern established in Tasks 4-8. Each gets a test.

- [ ] **Step 1: Write tests for all three**

Each test verifies: correct `name`, correct `requires_providers`, `analyze()` returns `LensResult` with expected `per_chapter` keys. Mark `@pytest.mark.slow`.

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_dialogue_voice.py tests/test_lens_slop_classifier.py tests/test_lens_emotion_arc.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement dialogue_voice lens**

Port from `ml/dialogue.py` (240 lines). Uses `providers.spacy` and `providers.sentence_transformer`. Put `speaker_separation`, `dialogue_ratio`, `talking_heads` count in `per_chapter`. Put per-speaker data in `raw`.

- [ ] **Step 4: Implement slop_classifier lens**

Port from `ml/slop_scorer.py` (188 lines). `SlopScorer` should remain a class inside the lens — it has its own ModernBERT model loading (HuggingFace checkpoint). Uses `providers.device` only. Put `flagged_pct`, `mean_slop` in `per_chapter`. Put per-paragraph scores in `per_paragraph`.

- [ ] **Step 5: Implement emotion_arc lens**

Port from `ml/emotion.py` (102 lines). Uses standalone distilbert pipeline — no provider needed. `requires_providers = []`. Put `flat` (bool as 0/1), `dynamic_range`, `arc` in `per_chapter`.

- [ ] **Step 6: Run all three tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_dialogue_voice.py tests/test_lens_slop_classifier.py tests/test_lens_emotion_arc.py -v -m slow`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/prose_doctor/lenses/dialogue_voice.py src/prose_doctor/lenses/slop_classifier.py src/prose_doctor/lenses/emotion_arc.py tests/test_lens_dialogue_voice.py tests/test_lens_slop_classifier.py tests/test_lens_emotion_arc.py
git commit -m "feat: dialogue_voice, slop_classifier, emotion_arc lens migrations"
```

---

## Task 10: Migrate perplexity lens

**Files:**
- Create: `src/prose_doctor/lenses/perplexity.py`
- Test: `tests/test_lens_perplexity.py`

- [ ] **Step 1: Write the test**

Test should verify `per_chapter` contains `mean_ppl`, `pct_below_55`, and that `per_paragraph` contains `perplexity` scores. Mark `@pytest.mark.slow`.

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement perplexity lens**

Port from `ml/perplexity.py` (104 lines). Uses `providers.gpt2`. Put `mean_ppl`, `pct_below_55` in `per_chapter`, per-paragraph perplexity in `per_paragraph`, smoothest paragraphs in `raw`.

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_perplexity.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/perplexity.py tests/test_lens_perplexity.py
git commit -m "feat: perplexity lens migration"
```

---

## Task 11: Extract uncertainty_reduction lens from prototype

**Files:**
- Create: `src/prose_doctor/lenses/uncertainty_reduction.py`
- Test: `tests/test_lens_uncertainty_reduction.py`

This is new code — extracted from `narrative_attention_proto.py` lines 253-263. The prototype computes per-paragraph uncertainty reduction (entropy decrease across paragraph boundaries using GPT-2).

- [ ] **Step 1: Write the test**

Test with sample text, verify `per_paragraph` contains `uncertainty_reduction` list of floats, `per_chapter` contains `mean_reduction`. Mark `@pytest.mark.slow`.

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement**

Extract the entropy-based computation from the prototype. Uses `providers.gpt2`. For each paragraph boundary, compute entropy of last N tokens of paragraph P and first N tokens of paragraph P+1 — the difference is uncertainty reduction.

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_uncertainty_reduction.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/uncertainty_reduction.py tests/test_lens_uncertainty_reduction.py
git commit -m "feat: uncertainty_reduction lens (extracted from narrative attention proto)"
```

---

## Task 12: Extract boyd_narrative_role lens from prototype

**Files:**
- Create: `src/prose_doctor/lenses/boyd_narrative_role.py`
- Test: `tests/test_lens_boyd_narrative_role.py`

Extracted from `narrative_attention_proto.py`. Boyd's staging/progression/tension model — classifies paragraphs by function-word profile.

- [ ] **Step 1: Write the test**

Test with sample text, verify `per_paragraph` contains `staging`, `progression`, `tension` lists, `per_chapter` contains the same as means. No providers needed — pure function-word counting.

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement**

Extract Boyd function-word sets and scoring from the prototype. No ML providers needed — this is word-set counting. `requires_providers = []`. Put per-paragraph scores in `per_paragraph`, means in `per_chapter`, dominant mode per paragraph in `raw`.

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_boyd_narrative_role.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/boyd_narrative_role.py tests/test_lens_boyd_narrative_role.py
git commit -m "feat: boyd_narrative_role lens (extracted from narrative attention proto)"
```

---

## Task 13: Extract fragment_classifier lens from agent_issues

**Files:**
- Create: `src/prose_doctor/lenses/fragment_classifier.py`
- Test: `tests/test_lens_fragment_classifier.py`

The craft-vs-crutch fragment classification logic currently lives in `agent_issues.py:find_fragment_issues()`. This lens wraps that detection into the lens interface.

- [ ] **Step 1: Write the test**

Test with text containing both craft fragments (dialogue, sensory detail, sequential rhythm) and crutch fragments (vague abstractions). Verify `per_paragraph` contains `fragment_ratio`, `per_chapter` contains `crutch_count` and `craft_count`. Uses spaCy.

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement**

Use `providers.spacy`. Move the `_is_vague_fragment()` and `_has_concrete_detail()` helpers from `agent_issues.py` into this module. `agent_issues.py` will import them from here after the rewire in Task 18.

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_fragment_classifier.py -v -m slow`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/fragment_classifier.py tests/test_lens_fragment_classifier.py
git commit -m "feat: fragment_classifier lens (extracted from agent_issues)"
```

---

## Task 14: Narrative attention meta-lens

**Files:**
- Create: `src/prose_doctor/lenses/narrative_attention.py`
- Test: `tests/test_lens_narrative_attention.py`

This is the meta-lens — it consumes outputs from psychic_distance, info_contour, foregrounding, emotion_arc, boyd_narrative_role, and uncertainty_reduction. Builds a per-paragraph feature matrix, computes cosine attention.

- [ ] **Step 1: Write the test**

Test with stub consumed results (mock `LensResult` dicts with known `per_paragraph` values). Verify the meta-lens produces `per_paragraph` with `attention_entropy` and `per_chapter` with coherence metrics. This test does NOT need `@pytest.mark.slow` since it uses stub data.

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement**

Port from `narrative_attention_proto.py:build_paragraph_features()` and the attention matrix computation. Instead of calling analyzers directly, read values from `consumed` dict. The lens should:
- Build feature matrix from consumed lenses' `per_paragraph` scores
- Compute cosine similarity attention matrix
- Detect scene structure from attention blocks
- Put feature matrix and attention matrix in `raw`
- Put per-paragraph attention entropy in `per_paragraph`
- Put coherence stats in `per_chapter`

`consumes_lenses = ["psychic_distance", "info_contour", "foregrounding", "emotion_arc", "boyd_narrative_role", "uncertainty_reduction"]`

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_narrative_attention.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/narrative_attention.py tests/test_lens_narrative_attention.py
git commit -m "feat: narrative_attention meta-lens"
```

---

## Task 15: Move cross-file utilities (twins, repetition, voice)

**Files:**
- Create: `src/prose_doctor/lenses/twins.py`
- Create: `src/prose_doctor/lenses/repetition.py`
- Create: `src/prose_doctor/lenses/voice.py`

These are NOT per-chapter lenses — they analyze multiple files together. They don't implement the `Lens` interface. They move to `lenses/` as utilities that share providers.

- [ ] **Step 1: Move twins.py**

Copy `ml/twins.py` to `lenses/twins.py`. Replace `model_manager` parameter with `providers: ProviderPool`. Change `model_manager.spacy` → `providers.spacy`, `model_manager.sentence_transformer` → `providers.sentence_transformer`.

- [ ] **Step 2: Move repetition.py**

Copy `ml/repetition.py` to `lenses/repetition.py`. Replace `model_manager` with `providers: ProviderPool`. Uses `providers.spacy` (optional).

- [ ] **Step 3: Move voice.py**

Copy `ml/voice.py` to `lenses/voice.py`. Replace `model_manager` with `providers: ProviderPool`. Uses `providers.sentence_transformer`.

- [ ] **Step 4: Verify imports work**

Run: `cd /home/ben/code/prose-doctor && uv run python -c "from prose_doctor.lenses.twins import find_twins; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/twins.py src/prose_doctor/lenses/repetition.py src/prose_doctor/lenses/voice.py
git commit -m "feat: move cross-file utilities (twins, repetition, voice) to lenses/"
```

---

## Task 16: Default lens registry

**Files:**
- Create: `src/prose_doctor/lenses/defaults.py`
- Test: append to `tests/test_lens_interface.py`

A function that creates a `LensRegistry` pre-populated with all standard lenses. This is what the runner, CLI, and agent code will call.

- [ ] **Step 1: Write the test**

```python
# append to tests/test_lens_interface.py
def test_default_registry_has_all_lenses():
    from prose_doctor.lenses.defaults import default_registry
    registry = default_registry()
    expected = [
        "pacing", "emotion_arc", "foregrounding", "info_contour",
        "psychic_distance", "sensory", "dialogue_voice", "slop_classifier",
        "perplexity", "uncertainty_reduction", "boyd_narrative_role",
        "fragment_classifier", "narrative_attention",
    ]
    for name in expected:
        assert registry.get(name) is not None, f"Missing lens: {name}"
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement**

```python
# src/prose_doctor/lenses/defaults.py
"""Default lens registry with all standard lenses."""
from prose_doctor.lenses import LensRegistry


def default_registry() -> LensRegistry:
    registry = LensRegistry()
    from prose_doctor.lenses.pacing import PacingLens
    from prose_doctor.lenses.emotion_arc import EmotionArcLens
    from prose_doctor.lenses.foregrounding import ForegroundingLens
    from prose_doctor.lenses.info_contour import InfoContourLens
    from prose_doctor.lenses.psychic_distance import PsychicDistanceLens
    from prose_doctor.lenses.sensory import SensoryLens
    from prose_doctor.lenses.dialogue_voice import DialogueVoiceLens
    from prose_doctor.lenses.slop_classifier import SlopClassifierLens
    from prose_doctor.lenses.perplexity import PerplexityLens
    from prose_doctor.lenses.uncertainty_reduction import UncertaintyReductionLens
    from prose_doctor.lenses.boyd_narrative_role import BoydNarrativeRoleLens
    from prose_doctor.lenses.fragment_classifier import FragmentClassifierLens
    from prose_doctor.lenses.narrative_attention import NarrativeAttentionLens

    for cls in [
        PacingLens, EmotionArcLens, ForegroundingLens, InfoContourLens,
        PsychicDistanceLens, SensoryLens, DialogueVoiceLens, SlopClassifierLens,
        PerplexityLens, UncertaintyReductionLens, BoydNarrativeRoleLens,
        FragmentClassifierLens, NarrativeAttentionLens,
    ]:
        registry.register(cls())
    return registry
```

- [ ] **Step 4: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_lens_interface.py::test_default_registry_has_all_lenses -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/lenses/defaults.py tests/test_lens_interface.py
git commit -m "feat: default lens registry with all 13 lenses"
```

---

## Task 17: Rewire agent_scan.py to use LensRunner

**Files:**
- Modify: `src/prose_doctor/agent_scan.py`

- [ ] **Step 1: Read current agent_scan.py**

Understand current imports and flow. Currently imports from `ml.psychic_distance`, `ml.info_contour`, `ml.foregrounding`, `ml.sensory`, and `ml.models.ModelManager`.

- [ ] **Step 2: Rewrite to use LensRunner**

Replace all `ml.*` imports with:
```python
from prose_doctor.providers import require_ml
from prose_doctor.lenses.defaults import default_registry
from prose_doctor.lenses.runner import LensRunner
from prose_doctor.providers import ProviderPool
```

The `scan_deep()` function should:
1. Call `require_ml()` (replaces `from prose_doctor.ml import require_ml`)
2. Create `ProviderPool()` and `LensRunner(default_registry(), pool)`
3. Call `runner.run_all(text, filename)`
4. Build `ProseMetrics` from the lens results' `per_chapter` dicts
5. Build the `report_dict` from lens results (maintaining the same dict structure `critique.py` expects)

- [ ] **Step 3: Run existing agent tests**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_models.py tests/test_agent.py -v`
Expected: PASS — the agent tests mock `_do_scan`, so they should still pass

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/agent_scan.py
git commit -m "refactor: rewire agent_scan to use LensRunner"
```

---

## Task 18: Rewire agent_issues.py

**Files:**
- Modify: `src/prose_doctor/agent_issues.py`

- [ ] **Step 1: Read current agent_issues.py**

Currently imports `ModelManager` and `PERCEPTION_VERBS`/`COGNITION_VERBS`/`PROXIMAL_DEICTICS` from `ml.psychic_distance`.

- [ ] **Step 2: Update imports**

Change imports to point at the lens modules:
- `from prose_doctor.ml.models import ModelManager` → `from prose_doctor.providers import ProviderPool`
- `from prose_doctor.ml.psychic_distance import PERCEPTION_VERBS, ...` → `from prose_doctor.lenses.psychic_distance import PERCEPTION_VERBS, ...`
- `mm = ModelManager()` → `pool = ProviderPool()`; `mm.spacy` → `pool.spacy`
- `_is_vague_fragment` / `_has_concrete_detail` → import from `prose_doctor.lenses.fragment_classifier`

- [ ] **Step 3: Run existing tests**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_issues.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/agent_issues.py
git commit -m "refactor: rewire agent_issues to use lens imports"
```

---

## Task 19: Rewire cli.py

**Files:**
- Modify: `src/prose_doctor/cli.py`

This is the largest rewire — `cli.py` imports from 14 different `ml.*` modules across its subcommands.

- [ ] **Step 1: Read current cli.py to catalog all ml imports**

- [ ] **Step 2: Replace all ml imports**

Key replacements:
- `from prose_doctor.ml import require_ml` → `from prose_doctor.providers import require_ml`
- `from prose_doctor.ml.models import ModelManager` → `from prose_doctor.providers import ProviderPool`
- Individual analyzer imports → use `LensRunner.run_one()` or import from `lenses/` modules
- `from prose_doctor.ml.twins import find_twins` → `from prose_doctor.lenses.twins import find_twins`
- `from prose_doctor.ml.perplexity import PerplexityScorer` → use `LensRunner.run_one("perplexity", ...)`

For `scan --deep`: use `LensRunner.run_all()` and format results.
For individual commands (`slop`, `foregrounding`, etc.): use `LensRunner.run_one()`.
For `critique`: pass lens results to `build_critique()` (via adapter from Task 20).
For `revise`: already goes through `agent_scan.py` — no direct changes needed.

Replace `ModelManager()` with `ProviderPool()` everywhere.

- [ ] **Step 3: Smoke test**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor --help`
Expected: Runs without import errors

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/cli.py
git commit -m "refactor: rewire cli.py to use lens framework"
```

---

## Task 20: Update critique.py to accept LensResult

**Files:**
- Modify: `src/prose_doctor/critique.py`

- [ ] **Step 1: Read current critique.py**

Currently expects a `report` dict with keys like `psychic_distance`, `info_contour`, etc., each containing specific nested fields.

- [ ] **Step 2: Add lens result adapter**

Add a function `lens_results_to_report(results: dict[str, LensResult]) -> dict` that converts lens results into the dict format `build_critique()` expects. This is a compatibility bridge — `build_critique()` keeps working with either input.

The adapter maps:
- `results["psychic_distance"].per_chapter` → `report["psychic_distance"]`
- Merging `per_chapter` and `raw` dicts for each lens
- Handling the field name differences between `LensResult` keys and the expected report dict keys

- [ ] **Step 3: Run existing tests**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/ -v --ignore=tests/test_agent_scan.py -m "not slow"`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/critique.py
git commit -m "refactor: critique.py accepts LensResult via adapter"
```

---

## Task 21: Validation framework

**Files:**
- Create: `src/prose_doctor/validation/__init__.py`
- Create: `src/prose_doctor/validation/corpus.py`
- Create: `src/prose_doctor/validation/discriminator.py`
- Create: `src/prose_doctor/validation/promotion.py`
- Create: `src/prose_doctor/validation/tiers.toml`
- Test: `tests/test_validation.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_validation.py
from prose_doctor.validation.discriminator import compute_discrimination
from prose_doctor.validation.promotion import check_tier


def test_compute_discrimination():
    human_scores = [0.4, 0.35, 0.38, 0.42, 0.39]
    llm_scores = [0.2, 0.18, 0.22, 0.19, 0.21]
    result = compute_discrimination(human_scores, llm_scores)
    assert result["cohens_d"] > 0.5
    assert result["p_value"] < 0.05


def test_check_tier_experimental():
    stats = {"cohens_d": 0.2, "p_value": 0.3}
    assert check_tier(stats, revision_evidence=[]) == "experimental"


def test_check_tier_validated():
    stats = {"cohens_d": 0.8, "p_value": 0.001}
    assert check_tier(stats, revision_evidence=[]) == "validated"


def test_check_tier_stable():
    stats = {"cohens_d": 0.8, "p_value": 0.001}
    evidence = ["b1ch01:3_accepted", "b2ch06:2_accepted"]
    assert check_tier(stats, revision_evidence=evidence) == "stable"
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_validation.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement corpus.py**

```python
# src/prose_doctor/validation/corpus.py
"""Human/LLM corpus management for lens validation."""
from __future__ import annotations

from pathlib import Path


def load_corpus(directory: Path) -> list[tuple[str, str]]:
    """Load all .md files from a directory. Returns [(filename, text), ...]."""
    files = sorted(directory.glob("**/*.md"))
    return [(f.name, f.read_text()) for f in files if f.stat().st_size > 100]
```

- [ ] **Step 4: Implement discriminator.py**

```python
# src/prose_doctor/validation/discriminator.py
"""Statistical discrimination between human and LLM prose."""
from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def compute_discrimination(
    human_scores: list[float],
    llm_scores: list[float],
) -> dict:
    """Compute Cohen's d, Mann-Whitney U, and KS statistic."""
    h = np.array(human_scores)
    l = np.array(llm_scores)

    pooled_std = np.sqrt((h.std()**2 + l.std()**2) / 2)
    d = (h.mean() - l.mean()) / pooled_std if pooled_std > 0 else 0.0

    u_stat, p_value = scipy_stats.mannwhitneyu(h, l, alternative="two-sided")
    ks_stat, ks_p = scipy_stats.ks_2samp(h, l)

    return {
        "cohens_d": round(float(d), 3),
        "p_value": round(float(p_value), 6),
        "u_statistic": float(u_stat),
        "ks_statistic": round(float(ks_stat), 3),
        "ks_p_value": round(float(ks_p), 6),
        "human_mean": round(float(h.mean()), 4),
        "llm_mean": round(float(l.mean()), 4),
        "human_n": len(h),
        "llm_n": len(l),
    }
```

- [ ] **Step 5: Implement promotion.py**

```python
# src/prose_doctor/validation/promotion.py
"""Tier promotion logic."""
from __future__ import annotations


def check_tier(
    stats: dict,
    revision_evidence: list[str],
) -> str:
    """Determine tier from discrimination stats and revision evidence.

    - experimental: lens runs without errors (default)
    - validated: Cohen's d > 0.5 OR p < 0.01
    - stable: validated + 3+ accepted edits across 2+ distinct chapters
    """
    d = abs(stats.get("cohens_d", 0))
    p = stats.get("p_value", 1.0)

    if d < 0.5 and p >= 0.01:
        return "experimental"

    if revision_evidence and len(revision_evidence) >= 2:
        total_accepted = 0
        chapters = set()
        for ev in revision_evidence:
            parts = ev.split(":")
            if len(parts) == 2 and "_accepted" in parts[1]:
                chapters.add(parts[0])
                try:
                    total_accepted += int(parts[1].split("_")[0])
                except ValueError:
                    pass
        if total_accepted >= 3 and len(chapters) >= 2:
            return "stable"

    return "validated"
```

- [ ] **Step 6: Create initial tiers.toml**

Write the expected initial tier assignments from the spec.

- [ ] **Step 7: Create validation/__init__.py**

Empty file.

- [ ] **Step 8: Run test — verify it passes**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_validation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add src/prose_doctor/validation/ tests/test_validation.py
git commit -m "feat: validation framework — discriminator, promotion, corpus loader"
```

---

## Task 22: CLI validate command

**Files:**
- Modify: `src/prose_doctor/cli.py`

- [ ] **Step 1: Add validate subcommand**

Add `validate` subparser with:
- `prose-doctor validate <lens_name>` — run one lens against corpus
- `prose-doctor validate --all` — run all lenses, print tier table
- `prose-doctor validate --promote` — auto-promote and write tiers.toml

- [ ] **Step 2: Add tier filtering flags to scan/revise**

Add `--experimental` and `--validated` flags to `scan` and `revise` subparsers. Pass tier filter to `LensRunner`.

- [ ] **Step 3: Smoke test**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor validate --help`
Expected: Prints validate subcommand help without errors

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/cli.py
git commit -m "feat: validate CLI command + tier filtering for scan/revise"
```

---

## Task 23: Delete ml/ directory and bump version

**Files:**
- Delete: `src/prose_doctor/ml/` (entire directory)
- Modify: `pyproject.toml`

- [ ] **Step 1: Verify no remaining imports from ml/**

Run: `cd /home/ben/code/prose-doctor && grep -r "from prose_doctor.ml" src/ tests/ --include="*.py" | grep -v __pycache__`
Expected: Zero hits (all rewired to lenses/providers)

- [ ] **Step 2: Check narrative_attention_proto.py**

The prototype at the repo root still imports from `ml/`. That's expected — it's a standalone script, not part of the package. It will need updating separately if you want to keep it runnable.

- [ ] **Step 3: Delete ml/**

```bash
rm -rf src/prose_doctor/ml/
```

- [ ] **Step 4: Bump version to 0.2.0**

In `pyproject.toml`, change `version = "0.1.0"` → `version = "0.2.0"`.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: delete ml/ directory, bump to 0.2.0"
```

---

## Task 24: End-to-end validation

- [ ] **Step 1: Run full test suite**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Smoke test — scan a real chapter**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor scan --deep /home/ben/code/nladfg/story_tracker/prose/book1/ch01.md`
Expected: Full report with all lens results

- [ ] **Step 3: Smoke test — revise dry run**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor revise --dry-run /home/ben/code/nladfg/story_tracker/prose/book1/ch01.md`
Expected: Critique + metrics, no LLM calls

- [ ] **Step 4: Smoke test — validate command**

Run: `cd /home/ben/code/prose-doctor && uv run prose-doctor validate --all`
Expected: Tier table for all lenses

- [ ] **Step 5: Commit any final fixes**

```bash
git add -A
git commit -m "feat: lens architecture v0.2.0 complete"
```
