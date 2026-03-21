# New Lens Integration into Revision System

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the four new lenses (concreteness, discourse_relations, situation_shifts, referential_cohesion) into the revision agent so it can detect and fix AI-typical prose patterns, using empirical baselines derived from 3,313 AI novels and validated against published human fiction.

**Architecture:** Add new metrics to `ProseMetrics` with baselines from corpus analysis. Add issue finders in `agent_issues.py` that locate specific paragraphs to fix. Expand `agent_scan.py` to run the new lenses. Each issue finder gives actionable, paragraph-level revision guidance.

**Tech Stack:** Existing lens framework, spaCy, no new dependencies.

**Prior context:** Empirical analysis of 3,313 AI novel chapters (Novelist dataset) and published human fiction established these discrimination baselines:

| Metric | AI mean (n=3313) | Human range (7 novels) | Cohen's d range | Direction |
|--------|------------------|----------------------|-----------------|-----------|
| `relation_entropy` | 0.433 ± 0.123 | 0.62–0.82 | 1.5–2.7 | human higher |
| `implicit_ratio` | 0.937 ± 0.022 | 0.86–0.90 | 1.4–2.5 | human lower |
| `causal_ratio` | 0.003 ± 0.005 | 0.008–0.024 | 0.8–2.1 | human higher |
| `concreteness_mean` | 3.127 ± 0.145 | 2.76–3.04 | 0.5–2.4 | human lower |
| `abstractness_ratio` | 0.192 ± 0.053 | 0.23–0.34 | 0.5–2.2 | human higher |
| `total_shifts` | 51.6 ± 15.9 | 62–140 | 0.5–3.4 | human higher |
| `subject_churn` | 0.661 ± 0.072 | 0.54–0.65 | 0.4–1.9 | human lower |

Key insight: `relation_entropy` is the single most reliable discriminator — every tested human author separates from AI with d > 1.4. Discourse relations and concreteness are the highest-value additions to the revision loop.

---

## File Map

### Modified files

| File | Change |
|------|--------|
| `src/prose_doctor/agent_models.py` | Add 4 new metrics to `ProseMetrics` and `BASELINES` |
| `src/prose_doctor/analyzers/doctor.py` | Add new lens fields to `ChapterHealth` dataclass + `to_dict()` |
| `src/prose_doctor/agent_scan.py` | Add new lenses to `_FULL_LENSES`, attach results to report dict |
| `src/prose_doctor/agent_issues.py` | Add 3 new issue finders + register in `METRIC_FINDERS` |
| `tests/test_agent_models.py` | Update tests for new metrics |
| `tests/test_agent_issues.py` | Add tests for new issue finders |

**Note:** `referential_cohesion` is wired into the scan for data collection but has no `ProseMetrics` field or issue finder yet — discrimination is moderate (d=0.4–1.9) and the lens needs coreference improvements before it's actionable. Deferred to a future plan.

---

## Task 1: Add new metrics to ProseMetrics

**Files:**
- Modify: `src/prose_doctor/agent_models.py`
- Test: `tests/test_agent_models.py`

- [ ] **Step 1: Update BASELINES and ProseMetrics**

Add four new metrics with baselines derived from corpus analysis. The baseline is the human-fiction target value. Direction indicates which side of the baseline is "better" (i.e., more human-like).

In `src/prose_doctor/agent_models.py`, add to `BASELINES`:

```python
# Discourse — human prose uses diverse, explicit connectives
"dr_entropy":       (0.65, "higher"),   # human range 0.62-0.82, AI mean 0.43
"dr_implicit":      (0.90, "lower"),    # human range 0.86-0.90, AI mean 0.94

# Concreteness — human prose is more abstract than AI
"cn_abstract":      (0.27, "higher"),   # human range 0.23-0.34, AI mean 0.19

# Situation shifts — normalized to per-paragraph rate
"ss_shift_rate":    (1.5, "higher"),    # human ~2 shifts/para, AI ~1 shift/para
```

Add corresponding fields to `ProseMetrics`:

```python
class ProseMetrics(BaseModel):
    # ... existing fields ...
    dr_entropy: float = 0.0
    dr_implicit: float = 1.0
    cn_abstract: float = 0.0
    ss_shift_rate: float = 0.0
```

Default values are the "worst case" (most AI-like) so that metrics degrade gracefully when lenses are unavailable.

- [ ] **Step 2: Run existing tests — verify they fail**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_models.py -v`
Expected: FAIL — new fields not matched in test fixtures

- [ ] **Step 3: Update test fixtures**

Update `test_prose_metrics_total_distance_at_baseline` to include new fields at their baseline values. Update `test_prose_metrics_total_distance_above_baseline` and `test_prose_metrics_distances` with the new fields.

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/agent_models.py tests/test_agent_models.py
git commit -m "feat: add discourse, concreteness, shift metrics to ProseMetrics

Baselines from empirical analysis of 3,313 AI novels vs human fiction:
- dr_entropy: discourse relation diversity (human 0.65, AI 0.43)
- dr_implicit: implicit relation ratio (human 0.90, AI 0.94)
- cn_abstract: abstractness ratio (human 0.27, AI 0.19)
- ss_shift_rate: situation shifts per paragraph (human 1.5, AI 1.0)"
```

---

## Task 2: Add new lens fields to ChapterHealth

**Files:**
- Modify: `src/prose_doctor/analyzers/doctor.py`

The `ChapterHealth` dataclass has a **hand-written** `to_dict()` method that explicitly lists every field. Both the dataclass fields AND the `to_dict()` method must be updated, or the new lens data will silently be dropped.

- [ ] **Step 1: Add fields to ChapterHealth dataclass**

After the existing `pacing` field (line 49), add:

```python
discourse_relations: dict | None = None
concreteness: dict | None = None
situation_shifts: dict | None = None
referential_cohesion: dict | None = None
```

- [ ] **Step 2: Update to_dict()**

Add to the return dict in `to_dict()` (after the `"pacing"` entry):

```python
"discourse_relations": self.discourse_relations,
"concreteness": self.concreteness,
"situation_shifts": self.situation_shifts,
"referential_cohesion": self.referential_cohesion,
```

- [ ] **Step 3: Verify existing tests pass**

```bash
cd /home/ben/code/prose-doctor && uv run pytest tests/test_analyzers.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/prose_doctor/analyzers/doctor.py
git commit -m "feat: add new lens fields to ChapterHealth + to_dict()"
```

---

## Task 3: Wire new lenses into agent_scan

**Files:**
- Modify: `src/prose_doctor/agent_scan.py`

- [ ] **Step 1: Add new lenses to scan sets**

Add to `_FULL_LENSES`:

```python
_FULL_LENSES = _METRIC_LENSES | {
    "sensory",
    "discourse_relations",
    "concreteness",
    "situation_shifts",
    "referential_cohesion",
}
```

Also add the discourse and concreteness lenses to `_METRIC_LENSES` since they're cheap (pure spaCy + dict lookup) and feed directly into ProseMetrics:

```python
_METRIC_LENSES = {
    "psychic_distance", "info_contour", "foregrounding",
    "discourse_relations", "concreteness", "situation_shifts",
}
```

- [ ] **Step 2: Attach new lens results to report dict**

After the existing `sensory_res` block, add:

```python
dr_res = results.get("discourse_relations")
if dr_res:
    pc = dr_res.per_chapter or {}
    report.discourse_relations = {
        "relation_entropy": pc.get("relation_entropy", 0),
        "implicit_ratio": pc.get("implicit_ratio", 1.0),
        "causal_ratio": pc.get("causal_ratio", 0),
        "contrastive_ratio": pc.get("contrastive_ratio", 0),
        "additive_ratio": pc.get("additive_ratio", 0),
        "temporal_ratio": pc.get("temporal_ratio", 0),
        "additive_only_zones": int(pc.get("additive_only_zones", 0)),
    }

cn_res = results.get("concreteness")
if cn_res:
    pc = cn_res.per_chapter or {}
    report.concreteness = {
        "concreteness_mean": pc.get("concreteness_mean", 3.0),
        "abstractness_ratio": pc.get("abstractness_ratio", 0),
        "vague_noun_density": pc.get("vague_noun_density", 0),
    }

ss_res = results.get("situation_shifts")
if ss_res:
    pc = ss_res.per_chapter or {}
    report.situation_shifts = {
        "total_shifts": int(pc.get("total_shifts", 0)),
        "time_shifts": int(pc.get("time_shifts", 0)),
        "space_shifts": int(pc.get("space_shifts", 0)),
        "actor_shifts": int(pc.get("actor_shifts", 0)),
        "disorientation_score": pc.get("disorientation_score", 0),
    }

rc_res = results.get("referential_cohesion")
if rc_res:
    pc = rc_res.per_chapter or {}
    report.referential_cohesion = {
        "coherence_score": pc.get("coherence_score", 0),
        "subject_churn": pc.get("subject_churn", 0),
        "entity_count": int(pc.get("entity_count", 0)),
    }
```

- [ ] **Step 3: Wire new metrics into ProseMetrics construction**

Update the `ProseMetrics(...)` constructor call to include the new fields:

```python
dr_dict = report_dict.get("discourse_relations") or {}
cn_dict = report_dict.get("concreteness") or {}
ss_dict = report_dict.get("situation_shifts") or {}

# Compute shift rate (shifts per paragraph)
from prose_doctor.text import split_paragraphs
n_paras = max(len(split_paragraphs(text)), 1)
total_shifts = ss_dict.get("total_shifts", 0)

metrics = ProseMetrics(
    # ... existing fields ...
    dr_entropy=dr_dict.get("relation_entropy", 0),
    dr_implicit=dr_dict.get("implicit_ratio", 1.0),
    cn_abstract=cn_dict.get("abstractness_ratio", 0),
    ss_shift_rate=round(total_shifts / n_paras, 3),
)
```

- [ ] **Step 4: Verify scan works end-to-end**

```bash
cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent.py -v
```

Expected: PASS (existing agent tests should still work since new fields have defaults)

- [ ] **Step 5: Commit**

```bash
git add src/prose_doctor/agent_scan.py
git commit -m "feat: wire discourse, concreteness, shift lenses into agent scan"
```

---

## Task 4: Add discourse relation issue finder

**Files:**
- Modify: `src/prose_doctor/agent_issues.py`
- Test: `tests/test_agent_issues.py`

This is the highest-value issue finder — discourse entropy is the strongest universal discriminator between human and AI prose.

- [ ] **Step 1: Write the test**

```python
def test_find_discourse_issues():
    """Additive-only prose should produce discourse issues."""
    text = (
        "Marcus walked down the corridor. And the walls were grey. And the "
        "floor was concrete. And the lights hummed overhead.\n\n"
        "He reached the end of the hall. And there was a door. And the door "
        "was locked. And he tried the handle again.\n\n"
        "The room beyond was small. And it was empty. And the window was "
        "boarded up. And dust covered every surface."
    )
    report = {"discourse_relations": {"relation_entropy": 0.15, "implicit_ratio": 0.95}}
    issues = find_discourse_issues(text, report)
    assert len(issues) > 0
    assert any("connective" in i.reason.lower() or "implicit" in i.reason.lower() for i in issues)
```

- [ ] **Step 2: Run test — verify it fails**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_issues.py::test_find_discourse_issues -v`
Expected: FAIL — function not defined

- [ ] **Step 3: Implement find_discourse_issues**

```python
def find_discourse_issues(text: str, report: dict) -> list[Issue]:
    """Find paragraphs with monotonous discourse relations.

    Targets two patterns empirically shown to separate AI from human prose:
    1. Long runs of implicit relations (no connectives at all)
    2. Additive-only zones (only "and" connectives, no causal/contrastive)

    Prescriptions focus on adding causal ("because", "so") and contrastive
    ("but", "however") connectives — the types most underused by AI.
    """
    pool = ProviderPool()
    nlp = pool.spacy

    paragraphs = split_paragraphs(text)
    dr = report.get("discourse_relations") or {}
    entropy = dr.get("relation_entropy", 1.0)
    implicit = dr.get("implicit_ratio", 0.0)

    # Only flag if chapter-level metrics are in AI territory.
    # Gate requires BOTH to be acceptable (conjunctive) — if either metric
    # looks AI-like, we run the finder. This avoids false negatives where
    # one metric masks problems in the other.
    if entropy > 0.55 and implicit < 0.92:
        return []

    from prose_doctor.lenses.discourse_relations import _classify_sentence

    issues = []
    consecutive_implicit = 0

    for pi, para in enumerate(paragraphs):
        doc = nlp(para)
        sents = list(doc.sents)
        para_relations = []

        for sent in sents:
            rel_name, _, evidence = _classify_sentence(sent.text)
            para_relations.append(rel_name)

        # Count implicit sentences in this paragraph
        n_implicit = sum(1 for r in para_relations if r == "implicit")
        n_additive = sum(1 for r in para_relations if r == "additive")
        n_total = len(para_relations)

        if n_total < 2:
            consecutive_implicit = 0
            continue

        all_implicit = n_implicit == n_total
        all_additive_or_implicit = all(r in ("implicit", "additive") for r in para_relations)

        if all_implicit:
            consecutive_implicit += 1
        else:
            consecutive_implicit = 0

        # Flag: 3+ consecutive all-implicit paragraphs
        if consecutive_implicit >= 3:
            first_sent = sents[0].text.strip()[:150] if sents else ""
            ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
            ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=first_sent,
                context_before=ctx_before,
                context_after=ctx_after,
                reason=(
                    f"implicit relation run ({consecutive_implicit} paragraphs with no connectives) — "
                    f"add a causal ('because', 'since') or contrastive ('but', 'however') "
                    f"connective to show how sentences relate"
                ),
                preserve=False,
            ))

        # Flag: additive-only paragraph (all "and" connectives)
        elif all_additive_or_implicit and n_additive >= 2:
            first_sent = sents[0].text.strip()[:150] if sents else ""
            ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
            ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=first_sent,
                context_before=ctx_before,
                context_after=ctx_after,
                reason=(
                    f"additive-only paragraph ({n_additive} 'and' connectives, no causal or contrastive) — "
                    f"replace some 'and' with 'because', 'so', 'but', or 'although' "
                    f"to show logical relationships"
                ),
                preserve=False,
            ))

    return issues[:15]
```

- [ ] **Step 4: Register in METRIC_FINDERS**

Add to the `METRIC_FINDERS` dict:

```python
"dr_entropy": find_discourse_issues,
"dr_implicit": find_discourse_issues,
```

- [ ] **Step 5: Add negative test (healthy prose returns no issues)**

```python
def test_find_discourse_issues_healthy():
    """Prose with good entropy and low implicit ratio should have no issues."""
    text = (
        "Marcus waited because the guard hadn't passed yet. Although the "
        "corridor looked clear, he knew better.\n\n"
        "Then he moved, fast and low. But the floor creaked under his weight. "
        "Consequently, the guard turned.\n\n"
        "Meanwhile, Elena watched from the rooftop. She could see the whole "
        "courtyard, so she tracked his progress."
    )
    report = {"discourse_relations": {"relation_entropy": 0.70, "implicit_ratio": 0.85}}
    issues = find_discourse_issues(text, report)
    assert len(issues) == 0
```

- [ ] **Step 6: Run tests — verify they pass**

Run: `cd /home/ben/code/prose-doctor && uv run pytest tests/test_agent_issues.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/prose_doctor/agent_issues.py tests/test_agent_issues.py
git commit -m "feat: discourse relation issue finder — targets implicit runs and additive-only zones"
```

---

## Task 5: Add concreteness issue finder

**Files:**
- Modify: `src/prose_doctor/agent_issues.py`
- Test: `tests/test_agent_issues.py`

Key insight from data: AI prose is too concrete (mean 3.13), human prose is more abstract (mean 2.86–3.04). The issue finder should flag excessively concrete passages that lack reflection, and also flag vague noun usage.

- [ ] **Step 1: Write the test**

```python
def test_find_concreteness_issues_flags_no_abstraction():
    """Relentlessly concrete prose should get flagged."""
    text = (
        "He pressed his back against the brick wall. His fingers found "
        "the crack in the mortar, cold and damp. The flashlight in his left "
        "hand was dead weight now, batteries drained hours ago.\n\n"
        "She set the coffee mug on the counter. The ceramic clinked against "
        "the granite. Outside, rain hammered the tin roof.\n\n"
        "He grabbed the door handle. The metal was cold. He twisted it. "
        "The hinges squeaked. The hallway stretched out before him."
    )
    report = {"concreteness": {
        "concreteness_mean": 3.4,
        "abstractness_ratio": 0.05,
        "vague_noun_density": 0.0,
    }}
    issues = find_concreteness_issues(text, report)
    assert len(issues) > 0
    assert any("abstract" in i.reason.lower() or "reflect" in i.reason.lower() for i in issues)
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement find_concreteness_issues**

```python
def find_concreteness_issues(text: str, report: dict) -> list[Issue]:
    """Find passages that are excessively concrete or use vague nouns.

    Two complementary signals:
    1. Chapters with very low abstractness_ratio (< 0.15) need moments of
       reflection, interpretation, or generalization. AI prose tends to
       stay relentlessly concrete.
    2. Vague nouns ("thing", "stuff", "something") that could be replaced
       with specific terms.
    """
    pool = ProviderPool()
    nlp = pool.spacy

    paragraphs = split_paragraphs(text)
    cn = report.get("concreteness") or {}
    abstract_ratio = cn.get("abstractness_ratio", 0.5)
    vague_density = cn.get("vague_noun_density", 0)

    issues = []

    from prose_doctor.lenses.concreteness import VAGUE_NOUNS

    # Issue type 1: Vague nouns
    if vague_density > 0.5:
        for pi, para in enumerate(paragraphs):
            doc = nlp(para)
            for sent in doc.sents:
                for token in sent:
                    if token.lemma_.lower() in VAGUE_NOUNS:
                        sent_text = sent.text.strip()[:150]
                        ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
                        ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
                        issues.append(Issue(
                            paragraph_idx=pi,
                            sentence_text=sent_text,
                            context_before=ctx_before,
                            context_after=ctx_after,
                            reason=(
                                f"vague noun '{token.text}' — replace with a specific noun "
                                f"(what thing? what kind? what exactly?)"
                            ),
                            preserve=False,
                        ))
                        break  # one per sentence

    # Issue type 2: No abstraction — flag longest concrete-only stretches
    if abstract_ratio < 0.15:
        from prose_doctor.lenses.concreteness import _load_norms
        norms = _load_norms()

        concrete_run = 0
        for pi, para in enumerate(paragraphs):
            doc = nlp(para)
            scores = []
            for token in doc:
                if not token.is_alpha or token.is_stop or len(token.text) <= 2:
                    continue
                word = token.lemma_.lower()
                if word in norms:
                    scores.append(norms[word])

            para_mean = sum(scores) / len(scores) if scores else 3.0
            if para_mean > 3.2:
                concrete_run += 1
            else:
                concrete_run = 0

            if concrete_run >= 4:
                sent_text = paragraphs[pi][:150]
                ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
                ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
                issues.append(Issue(
                    paragraph_idx=pi,
                    sentence_text=sent_text,
                    context_before=ctx_before,
                    context_after=ctx_after,
                    reason=(
                        f"concrete run ({concrete_run} paragraphs with no abstraction) — "
                        f"add a moment of reflection, interpretation, memory, or opinion "
                        f"to break the sensory surface"
                    ),
                    preserve=False,
                ))

    return issues[:15]
```

- [ ] **Step 4: Register in METRIC_FINDERS**

```python
"cn_abstract": find_concreteness_issues,
```

- [ ] **Step 5: Run tests — verify they pass**

- [ ] **Step 6: Commit**

```bash
git add src/prose_doctor/agent_issues.py tests/test_agent_issues.py
git commit -m "feat: concreteness issue finder — flags vague nouns and missing abstraction"
```

---

## Task 6: Add situation shift issue finder

**Files:**
- Modify: `src/prose_doctor/agent_issues.py`
- Test: `tests/test_agent_issues.py`

- [ ] **Step 1: Write the test**

```python
def test_find_shift_issues_flags_static_scene():
    """A long scene with no shifts should get flagged."""
    # 6 paragraphs, same time/place/character
    text = "\n\n".join([
        "Marcus sat at the table. He stared at the map.",
        "He traced the route with his finger. The line ran north.",
        "He leaned back in the chair. The map was wrong.",
        "He checked the coordinates again. Still wrong.",
        "He pulled out a second map. This one was older.",
        "He compared the two. The discrepancy was clear.",
    ])
    report = {"situation_shifts": {"total_shifts": 0, "time_shifts": 0, "space_shifts": 0, "actor_shifts": 0}}
    issues = find_shift_issues(text, report)
    assert len(issues) > 0
    assert any("static" in i.reason.lower() or "shift" in i.reason.lower() for i in issues)
```

- [ ] **Step 2: Run test — verify it fails**

- [ ] **Step 3: Implement find_shift_issues**

```python
def find_shift_issues(text: str, report: dict) -> list[Issue]:
    """Find long static stretches that lack time, space, or character shifts.

    Human prose averages ~1.5-2 shifts per paragraph. AI prose averages ~1.
    Flag runs of 5+ paragraphs with no shifts — these are the most static
    stretches where a cut, time skip, or new character would add dynamism.
    """
    paragraphs = split_paragraphs(text)
    ss = report.get("situation_shifts") or {}
    total = ss.get("total_shifts", 999)
    n_paras = max(len(paragraphs), 1)
    shift_rate = total / n_paras

    # Only flag if shift rate is below human baseline
    if shift_rate > 1.2:
        return []

    pool = ProviderPool()
    nlp = pool.spacy

    from prose_doctor.lenses.situation_shifts import (
        _has_temporal_markers, _get_first_subject,
    )

    issues = []
    no_shift_run = 0
    prev_subject = None

    for pi, para in enumerate(paragraphs):
        doc = nlp(para)
        cur_subject = _get_first_subject(doc)
        has_time = _has_temporal_markers(para)

        has_any_shift = has_time  # time marker = shift
        if cur_subject and prev_subject and cur_subject != prev_subject:
            if cur_subject not in ("he", "she", "they", "it") and prev_subject not in ("he", "she", "they", "it"):
                has_any_shift = True

        if has_any_shift:
            no_shift_run = 0
        else:
            no_shift_run += 1

        if no_shift_run >= 5:
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=paragraphs[pi][:150],
                context_before=paragraphs[pi - 1][:100] if pi > 0 else "",
                context_after=paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else "",
                reason=(
                    f"static scene ({no_shift_run} paragraphs, same time/place/character) — "
                    f"consider a time reference, location change, new character entering, "
                    f"or a scene cut"
                ),
                preserve=False,
            ))

        prev_subject = cur_subject

    return issues[:10]
```

- [ ] **Step 4: Register in METRIC_FINDERS**

```python
"ss_shift_rate": find_shift_issues,
```

- [ ] **Step 5: Run tests — verify they pass**

- [ ] **Step 6: Commit**

```bash
git add src/prose_doctor/agent_issues.py tests/test_agent_issues.py
git commit -m "feat: situation shift issue finder — flags static scenes lacking transitions"
```

---

## Task 7: Full integration test

**Files:**
- No new files — run existing tests and a manual validation

- [ ] **Step 1: Run full test suite**

```bash
cd /home/ben/code/prose-doctor && uv run pytest tests/ -v
```

Expected: ALL PASS

- [ ] **Step 2: End-to-end validation**

Run a scan with the new lenses to verify they produce data:

```bash
cd /home/ben/code/prose-doctor && uv run python -c "
from prose_doctor.agent_scan import scan_deep
sample = list(Path('corpus/baselines').glob('*.md'))[0]
metrics, report = scan_deep(sample.read_text())
print(f'Total distance: {metrics.total_distance}')
print(f'Discourse entropy: {metrics.dr_entropy}')
print(f'Abstractness: {metrics.cn_abstract}')
print(f'Shift rate: {metrics.ss_shift_rate}')
print(f'Implicit ratio: {metrics.dr_implicit}')
print(f'Worst metric: {metrics.worst_metric}')
print(f'Distances: {metrics.distances()}')

# Verify issue finders work
from prose_doctor.agent_issues import find_issues
issues = find_issues('dr_entropy', sample.read_text(), report)
print(f'\nDiscourse issues: {len(issues)}')
for i in issues[:3]:
    print(f'  [{i.paragraph_idx}] {i.reason}')
"
```

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "chore: integration fixes for new lens metrics"
```
