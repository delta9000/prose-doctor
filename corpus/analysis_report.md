# prose-doctor Metric Analysis Report

## 1. Methodology

### Datasets

| Group | Label | Chapters | Words | Source |
|-------|-------|----------|-------|--------|
| Human baselines | `human` | 35 | 140,000 | Doctorow (10), Watts (10), Shelley (5), Conrad (5), Dickens (5) |
| LLM corpus | `llm` | 10 | 18,859 | DeepSeek v3.2 (3), Hermes-4 70B (2), Qwen3 30B (2), Mistral Small Creative (2), Llama-4 Scout (1) |
| Book 1 | `book1` | 13 | 50,247 | DeepSeek-generated post-apocalyptic novel |

### Critical confounds

1. **Paragraph splitting failure on baselines.** Every human chapter reads as a single paragraph (paragraphs=1). The raw files contain no blank-line paragraph breaks — the full ~4000 words are one continuous block. LLM and Book 1 files have normal paragraph structure (30–170 paragraphs per chapter). This means any metric computed *per-paragraph* (perplexity distribution, psychic distance curves, emotion arcs) is comparing a single giant paragraph against many small ones. The effect is severe:
   - `pct_below_55` for humans is binary: either 0% or 100%, depending on whether the single-paragraph mean falls above or below 55. No intermediate values exist. Among LLM and Book 1 data, this metric shows a continuous distribution.
   - `emotion_std` and `emotion_dynamic_range` are exactly 0.0 for all 35 human chapters. With one paragraph, there is no variance to measure. These metrics are **not broken for detection** — they accidentally become perfect classifiers — but they tell us nothing about whether human prose actually has flatter emotion arcs.
   - `pd_zoom_jumps` is artificially low for humans (mean 0.91) because there's only one paragraph to jump between. Book 1 averages 7.15 jumps across ~100 paragraphs.
   - `paragraph_means` lists contain exactly 1 entry for all human chapters. Per-paragraph psychic distance curves are unavailable for the baselines.

2. **Word count mismatch.** All human chapters are truncated to exactly 4000 words. LLM corpus chapters range from 657 to 2,864 words (mean 1,886). Book 1 chapters range from 1,468 to 5,287 words (mean 3,865). Metrics correlated with word count (|r| > 0.3): `emotion_std` (r=-0.43), `emotion_dynamic_range` (r=-0.46), `ic_rhythmicity` (r=-0.65), `ic_flatlines` (r=+0.36), `ic_spikes` (r=+0.62).

3. **Small LLM sample.** Only 10 chapters from 5 models (1–3 each). High within-group variance makes effect sizes unstable.

4. **Genre mismatch.** Human corpus spans YA contemporary (Doctorow), hard sci-fi (Watts), Gothic (Shelley), literary (Conrad, Dickens). LLM and Book 1 are all genre fiction responding to specific prompts. Some "discriminators" may be separating genre/era, not human-vs-LLM.

### What to trust

Metrics computed at the **whole-chapter** level without paragraph decomposition are comparable across groups: `mean_ppl`, foregrounding axes (`fg_index`, `fg_alliteration`, `fg_inversion`, `fg_sent_len_cv`, `fg_fragment`), `ic_mean_surprisal`, `ic_cv_surprisal`, sensory scores, `vocab_crutch_count`, `vocab_crutch_excess`, `pattern_hits`, `density_items`. These are the focus of this analysis.

---

## 2. Strong Discriminators (|Cohen's d| > 0.8, human vs LLM)

Excluding confounded metrics (paragraph-dependent or word-count-driven), the following reliably separate human from LLM prose:

### 2.1. Perplexity (`mean_ppl`) — d = -1.36

| Group | Mean | Std | Median | P25 | P75 |
|-------|------|-----|--------|-----|-----|
| Human | 44.1 | 16.9 | 38.7 | 33.7 | 57.0 |
| LLM | 78.5 | 44.4 | 74.0 | 51.3 | 113.1 |
| Book 1 | 123.1 | 29.0 | 117.5 | 106.3 | 141.1 |

LLM prose is substantially higher perplexity than human baselines. Book 1 is higher still — z = +4.69 vs human, z = +1.01 vs LLM. However, there is significant within-LLM variance: Hermes-4 averages 18.0 (lower than any human author), while DeepSeek v3.2 averages 130.4. Perplexity is computed against GPT-2; it partly measures vocabulary rarity and syntactic complexity, not just "predictability to a language model."

Per-author human breakdown:
- Doctorow: 28.4 (low — accessible YA prose)
- Dickens: 40.4
- Shelley: 41.2
- Conrad: 49.3
- Watts: 60.4 (high — dense hard sci-fi)

### 2.2. Inversion percentage (`fg_inversion`) — d = +1.07

| Group | Mean | Std |
|-------|------|-----|
| Human | 41.4 | 12.0 |
| LLM | 27.6 | 16.1 |
| Book 1 | 16.0 | 6.7 |

Human prose inverts more (subject-verb order deviations, fronted adverbials, etc.). This is a genuine stylistic signal. Book 1 is notably low — z = -2.13 vs human, below even the LLM corpus mean.

Per-author: Shelley leads at 61.9%, Doctorow/Conrad/Dickens cluster around 41–44%, Watts is lowest among humans at 27.6%. The Watts-LLM overlap suggests this metric partly captures prose formality/complexity rather than human-ness per se.

### 2.3. Sentence length CV (`fg_sent_len_cv`) — d = +0.95

| Group | Mean | Std |
|-------|------|-----|
| Human | 0.71 | 0.11 |
| LLM | 0.60 | 0.10 |
| Book 1 | 0.67 | 0.06 |

Human prose has more varied sentence lengths. Book 1 sits between the groups (z = -0.33 vs human). This is a clean, non-confounded metric.

Per-author: Dickens leads at 0.89, Conrad 0.74, Watts 0.71, Doctorow 0.67, Shelley 0.57. Hermes-4 is lowest at 0.45.

### 2.4. Vocabulary crutch count (`vocab_crutch_count`) — d = -1.56

| Group | Mean | Std |
|-------|------|-----|
| Human | 42.5 | 10.8 |
| LLM | 63.2 | 20.1 |
| Book 1 | 66.9 | 9.1 |

LLM prose repeats more unique words beyond their "budget." Book 1 is solidly in LLM territory (z = +2.27 vs human). Note: this is partly word-count-normalized (budget scales with chapter length), but human chapters are all 4000 words while LLM chapters average only 1886 words — meaning LLMs hit more crutches in *less* text.

### 2.5. Pattern hits (`pattern_hits`) — d = -1.72

| Group | Mean | Std |
|-------|------|-----|
| Human | 3.4 | 3.0 |
| LLM | 19.2 | 19.3 |
| Book 1 | 16.5 | 11.8 |

Formulaic constructions (began/started to, etc.) are far more common in LLM prose. Human variance is low. Book 1 z = +4.43 vs human. This is one of the cleanest discriminators, though LLM variance is very high (Hermes-4 averages 34.5 hits, Llama-4 only 4.0).

### 2.6. Density over budget (`density_items`) — d = -1.66

| Group | Mean | Std |
|-------|------|-----|
| Human | 0.03 | 0.17 |
| LLM | 0.70 | 0.82 |
| Book 1 | 1.69 | 0.95 |

Nearly zero for humans, nonzero for LLMs, highest for Book 1 (z = +9.82 vs human). Only 1 of 35 human chapters triggers any density-over-budget flag. Book 1 triggers it in 12 of 13 chapters.

### 2.7. Info contour spikes (`ic_spikes`) — d = +1.81

| Group | Mean | Std |
|-------|------|-----|
| Human | 8.1 | 1.9 |
| LLM | 4.6 | 2.0 |
| Book 1 | 9.7 | 0.9 |

Human prose has more information spikes. Interestingly, Book 1 *exceeds* the human mean here — it has more surprisal spikes than even the human baselines. This may be correlated with word count (r = +0.62).

### 2.8. Psychic distance mean (`pd_mean`) — d = +1.79

| Group | Mean | Std |
|-------|------|-----|
| Human | 0.33 | 0.02 |
| LLM | 0.30 | 0.02 |
| Book 1 | 0.28 | 0.01 |

Human prose sits at slightly higher psychic distance (more "establishing shot" / narrative distance). Book 1 is the lowest of all groups — z = -3.46 vs human. Given that the human chapters are single-paragraph blocks, this may be distorted — the classifier sees one huge chunk and assigns a different distance profile than it would for paragraph-segmented text. **Treat with caution.**

### 2.9. Total issues — d = -0.92

| Group | Mean | Std |
|-------|------|-----|
| Human | 141.5 | 44.1 |
| LLM | 190.8 | 79.4 |
| Book 1 | 315.9 | 74.9 |

This is a composite count (vocab crutches + pattern hits + density + others). Book 1 is an extreme outlier — z = +3.95 vs human. However, it's a sum of other metrics, not an independent signal.

---

## 3. Weak/Broken Discriminators

### 3.1. Emotion metrics (`emotion_std`, `emotion_dynamic_range`) — d = -5.93 / -7.23

These show *perfect* separation because human chapters have exactly one paragraph, yielding std=0 and dynamic_range=0 by definition. **Completely confounded by paragraph splitting failure.** Not usable until baselines are re-processed with proper paragraph breaks.

### 3.2. Smooth percentage (`pct_below_55`) — d = +0.43

For humans, this is binary: either 0% or 100%. Distribution:
- Human: 12 chapters at 0%, 23 at 100%, 0 in between
- LLM: 2 at 100%, 0 at 0%, 8 in between (continuous range 2.6–73.7%)
- Book 1: 1 at 0%, 0 at 100%, 12 in between (range 8.1–30.8%)

The bimodality in humans is **real but an artifact**: with one paragraph per chapter, there's only one perplexity value, which is either above or below 55. The metric is not inherently broken, but the single-paragraph problem makes the human values meaningless for comparison.

### 3.3. Sensory scores (all 6 modalities + balance + deserts)

All sensory metrics show near-zero variance. Cohen's d values: visual 0.36, auditory 0.68, haptic 0.33, olfactory 0.48, gustatory 0.84, interoceptive 0.62, balance 0.15, deserts 0.00.

The scores cluster in extremely narrow bands:
- Visual: 0.44–0.47 across all groups
- Balance: 0.99 everywhere
- Deserts: 0 everywhere

The classifier appears to be producing near-constant outputs regardless of input. Within-human CV is < 0.025 for all sensory metrics. These metrics are effectively non-functional as discriminators or style measures. The gustatory d=0.84 is technically "strong" but the actual difference is 0.42 vs 0.41 — an absolute gap of 0.01 on a 0–1 scale.

### 3.4. Foregrounding index (`fg_index`) — d = 0.67

The composite index is a medium discriminator but has significant overlap. Human mean 7.38, LLM mean 7.05. The individual axes (inversion, sent_len_cv) are more informative than the composite.

### 3.5. Alliteration (`fg_alliteration`) — d = 0.01

Zero discriminative power. Human mean 21.3, LLM mean 21.3. Alliteration rate is identical across human and LLM prose.

### 3.6. Fragment percentage (`fg_fragment`) — d = -0.29

Weak discriminator. High within-human variance (CV = 0.59) driven by Watts (17.1%) vs Shelley (3.5%). This is more of a style dimension than a human/LLM dimension.

### 3.7. Info contour CV (`ic_cv_surprisal`) — d = -0.17

No discriminative power. All groups cluster around 0.19–0.20.

### 3.8. Psychic distance zoom jumps (`pd_zoom_jumps`) — d = -2.07

Technically a strong discriminator but **confounded by paragraph count**. With 1 paragraph, humans max out at 1 jump. With 100+ paragraphs, Book 1 averages 7.15 jumps. Not comparable.

---

## 4. Style Dimensions (high within-human variance)

These metrics vary substantially among human authors, making them useful for characterizing *style* rather than detecting AI generation.

### 4.1. Fragment percentage (`fg_fragment`) — CV = 0.59

| Author | Mean | Range |
|--------|------|-------|
| Watts | 17.1 | 8.9–21.2 |
| Doctorow | 7.8 | 5.0–14.4 |
| Conrad | 7.4 | 4.6–8.9 |
| Dickens | 7.3 | 4.2–12.4 |
| Shelley | 3.5 | 2.8–4.7 |

Sentence fragment rate captures prose rhythm. Watts's choppy hard-sci-fi style fragments heavily; Shelley's 19th-century formal prose almost never does. Book 1 at 17.1% matches Watts exactly.

### 4.2. Perplexity (`mean_ppl`) — CV = 0.38

Perplexity is both a discriminator and a style dimension. Among humans, Doctorow (28.4) and Watts (60.4) span a 2x range. The metric captures vocabulary complexity, syntactic novelty, and era-specific language.

### 4.3. Inversion percentage (`fg_inversion`) — CV = 0.29

Shelley (61.9%) inverts far more than Watts (27.6%). This captures syntactic formality and period-appropriate prose style.

### 4.4. Info contour rhythmicity (`ic_rhythmicity`) — CV = 0.31

| Author | Mean | Range |
|--------|------|-------|
| Dickens | 0.17 | 0.10–0.22 |
| Shelley | 0.16 | 0.13–0.19 |
| Conrad | 0.12 | 0.09–0.14 |
| Doctorow | 0.11 | 0.08–0.15 |
| Watts | 0.10 | 0.07–0.12 |

Higher rhythmicity = more periodic/regular information flow. 19th-century prose is more rhythmic; modern genre prose is less so. LLM corpus is notably *more* rhythmic (0.19) than any human author. Book 1 is the *least* rhythmic of all groups (0.09). This is a potential quality signal: rhythmic ≈ monotonous pacing.

### 4.5. Sentence length CV (`fg_sent_len_cv`) — CV = 0.16

Dickens (0.89) has extremely varied sentence lengths; Shelley (0.57) is more uniform. Moderate style dimension.

### 4.6. Pattern hits — CV = 0.87

High human variance, but the range (1–11 for humans, 4–34 for LLMs) still separates the groups. Watts averages 5.7 pattern hits — higher than other human authors, overlapping with the low end of LLMs.

---

## 5. Redundant Metrics

Correlation analysis (Pearson r, all 58 samples) identifies the following clusters:

### Cluster 1: Perplexity-Inversion-Distance nexus
- `mean_ppl` x `fg_inversion`: r = -0.78
- `mean_ppl` x `pd_mean`: r = -0.80
- `mean_ppl` x `ic_mean_surprisal`: r = +0.85
- `fg_inversion` x `pd_mean`: r = +0.82
- `fg_inversion` x `ic_mean_surprisal`: r = -0.73
- `pd_mean` x `ic_mean_surprisal`: r = -0.78
- `pd_mean` x `emotion_std`: r = -0.79

These five metrics are measuring overlapping aspects of the same underlying dimension: prose that is syntactically complex/inverted tends to be higher perplexity, higher psychic distance, and lower information surprisal. **Recommendation: keep `mean_ppl` and `fg_inversion` as representatives; treat `pd_mean`, `ic_mean_surprisal`, and `emotion_std` as partially redundant.**

### Cluster 2: Foregrounding composite
- `fg_index` x `fg_sent_len_cv`: r = +0.80
- `fg_fragment` x `fg_inversion`: r = -0.84

The composite `fg_index` is dominated by `fg_sent_len_cv`. Fragment rate and inversion rate are strongly anti-correlated — prose that fragments more also inverts less. **Recommendation: report `fg_inversion` and `fg_sent_len_cv` separately rather than using the composite index.**

### Cluster 3: Issue counts
- `vocab_crutch_count` x `pattern_hits`: r = +0.45
- `vocab_crutch_count` x `emotion_std`: r = +0.63
- `pattern_hits` x `emotion_std`: r = +0.60

These are moderately correlated. Each captures a distinct failure mode (repetition, formulaic construction, paragraph-level emotion uniformity), so keeping all three is justified despite correlation.

### Not redundant (r < 0.4 between them)
- `fg_alliteration` vs all other metrics — independent
- `ic_cv_surprisal` vs most metrics — independent
- `ic_rhythmicity` vs most metrics — independent (and anti-correlated with word count)
- `sensory_balance` — independent but non-functional (near-constant)

---

## 6. Book 1 Diagnosis

Book 1 (13 chapters of DeepSeek-generated prose) shows the following profile relative to human baselines and the LLM corpus:

### Outlier vs human (|z| > 2)

| Metric | Book 1 | z vs human | z vs LLM | Interpretation |
|--------|--------|------------|----------|----------------|
| `mean_ppl` | 123.1 | +4.69 | +1.01 | Far higher perplexity than humans; above even the LLM corpus mean |
| `fg_inversion` | 16.0 | -2.13 | -0.72 | Very low syntactic inversion; even lower than LLM average |
| `pd_mean` | 0.28 | -3.46 | -1.20 | Lowest psychic distance of all groups (but confounded by paragraph structure) |
| `ic_mean_surprisal` | 5.26 | +2.40 | +0.94 | High information density (correlated with perplexity) |
| `sensory_interoceptive` | 0.28 | -3.14 | -0.33 | Slightly below human baseline (tiny absolute gap: 0.01) |
| `vocab_crutch_count` | 66.9 | +2.27 | +0.18 | Matches LLM corpus; well above human |
| `pattern_hits` | 16.5 | +4.43 | -0.14 | Matches LLM corpus; far above human |

### Outlier vs LLM

| Metric | Book 1 | z vs LLM | Interpretation |
|--------|--------|----------|----------------|
| `ic_rhythmicity` | 0.09 | -2.01 | Less rhythmic/more chaotic pacing than LLM corpus |

### Within normal range for both groups

| Metric | Book 1 | Assessment |
|--------|--------|------------|
| `fg_index` | 7.35 | Human-typical |
| `fg_sent_len_cv` | 0.67 | Between groups, closer to human |
| `fg_alliteration` | 23.9 | Slightly above both (not significant) |
| `ic_cv_surprisal` | 0.19 | Dead center of all groups |
| `sensory_balance` | 0.99 | Same as everyone |

### Summary

Book 1's most distinctive signature:
1. **Very high perplexity** (123.1) — 4.7 standard deviations above human mean. This may reflect DeepSeek's tendency toward unusual word choices and elaborate constructions.
2. **Very low inversion** (16.0%) — the lowest of all three groups. Prose is syntactically straightforward (subject-verb-object) with little syntactic variety.
3. **High pattern hits** (16.5) and **vocab crutches** (66.9) — typical LLM repetition signatures.
4. **High density over budget** (1.69) — the worst of all three groups. Prose is overloaded with descriptive density.
5. **Low rhythmicity** (0.09) — actually *better* than the LLM corpus (which is too rhythmic/monotonous). Book 1's pacing is more varied, closer to Doctorow/Watts.
6. **High fragment rate** (17.1%) — matches Watts-level choppiness, well above the LLM corpus.

The overall picture: Book 1 reads as recognizably LLM-generated on repetition metrics (crutches, patterns, density) but has better-than-average pacing variety and sentence rhythm for LLM prose. Its biggest weaknesses compared to human prose are syntactic monotony (low inversion) and word repetition.

---

## 7. Recommendations

### 7.1. Metrics worth using in critique prompts

**Tier 1 — Strong, non-confounded discriminators:**

| Metric | Human reference | What to tell the LLM |
|--------|----------------|---------------------|
| `fg_inversion` | 41% (range 28–62%) | "Invert at least 30% of clauses — front adverbials, use passive voice for emphasis, vary subject-verb order." |
| `fg_sent_len_cv` | 0.71 (range 0.57–0.89) | "Sentence length coefficient of variation should be above 0.65. Mix very short fragments with long complex sentences." |
| `pattern_hits` | 3.4 per 4k words (range 0–11) | "Eliminate formulaic constructions: 'began to', 'started to', 'couldn't help but'. Target < 5 per chapter." |
| `vocab_crutch_count` | 42 per 4k words | "Reduce repeated words. If a non-character word appears > 5 times per 4k words, find synonyms or restructure." |
| `density_items` | 0 (almost never triggered in human prose) | "If a passage is flagged for density-over-budget, thin the descriptive load. Human prose rarely overloads." |

**Tier 2 — Useful but partially confounded or redundant:**

| Metric | Human reference | Notes |
|--------|----------------|-------|
| `mean_ppl` | 44 (range 17–80) | Varies enormously by author/genre. Doctorow=28, Watts=60. Not useful as a single threshold. |
| `ic_rhythmicity` | 0.12 (range 0.07–0.22) | LLMs tend too high (0.19). Target < 0.15 for genre fiction. |
| `ic_spikes` | 8.1 per chapter | Correlated with word count. Use directionally — more spikes = better. |
| `fg_fragment` | 9.7% (range 3–21%) | Style-dependent. Match target author's style. |

### 7.2. Metrics to drop or fix

| Metric | Action |
|--------|--------|
| `emotion_std`, `emotion_dynamic_range` | **Fix**: re-process baselines with paragraph breaks. Currently unusable for comparison. |
| `pct_below_55` | **Fix**: same paragraph-splitting issue makes human values meaningless. |
| `pd_zoom_jumps` | **Fix**: confounded by paragraph count. |
| `sensory_*` (all 6 modalities + balance + deserts) | **Drop or rework**: classifier outputs near-constant values regardless of input. The model is not actually measuring sensory content. |
| `fg_alliteration` | **Drop from detection**: zero discriminative power (d=0.01). May still be useful as a style dimension if the underlying measurement is validated. |
| `ic_cv_surprisal` | **Drop**: no discriminative power, low style variance. |
| `fg_index` (composite) | **Drop in favor of components**: `fg_inversion` and `fg_sent_len_cv` are more informative individually. |

### 7.3. Reference values for Book 1 critique

For the specific case of Book 1, the critique prompt should focus on:

1. **Inversion deficit** — Book 1 is at 16%, human floor is 28% (Watts). Push toward 30%+.
2. **Pattern hit excess** — Book 1 averages 16.5 per chapter vs human 3.4. Target < 8.
3. **Density overload** — 1.69 items per chapter vs near-zero human. Thin descriptions.
4. **Vocabulary crutches** — 66.9 per chapter vs human 42.5. Reduce repetition.
5. **Sentence length variety** — Book 1 at 0.67 is adequate but could improve toward 0.71+.

Book 1's strengths relative to other LLM prose: low rhythmicity (good pacing variety), high fragment rate (choppy style reads as intentional), high info spikes (avoids monotonous information flow).

### 7.4. Baseline reprocessing recommendation

The single most impactful improvement to this analysis would be re-processing the human baseline files with proper paragraph breaks inserted. The current single-paragraph format invalidates all paragraph-level metrics (emotion, psychic distance curves, perplexity distribution, zoom jumps). Until this is fixed, approximately 40% of the analyzer's output is not comparable across groups.
