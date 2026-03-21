# Experimental Design: Validating prose-doctor's Claims

## What We Claim

1. Computational psychic distance is measurable and discriminates human from LLM prose
2. Six metrics reliably separate human from LLM fiction
3. LLM prose is characteristically "too regular, too external, too fragmentary"
4. A critique-and-revise loop measurably improves LLM prose on structural dimensions
5. Structural improvements (inversion, fragments) are achievable; information-theoretic improvements (rhythmicity, surprisal) are not
6. Prompt engineering informed by metrics produces better first drafts over successive stories
7. Ending register shift is measurable and LLMs fail to produce it

## What We Need to Prove

### Experiment 1: Metric Validation — Do Our Metrics Measure What We Think?

**Question:** Are our six discriminators measuring literary craft features or just surface statistics that happen to correlate with authorship?

**Design:**
- Take 10 passages from published human fiction that a writing workshop would critique as BAD (purple prose, telling-not-showing, padding)
- Take 10 passages from LLM fiction that a workshop would praise as GOOD (specific, grounded, earned)
- Score both with all metrics
- If the metrics flag the bad human prose and pass the good LLM prose, they're measuring craft. If they flag all LLM and pass all human regardless of quality, they're measuring provenance, not craft.

**Data needed:** Curated examples with human quality judgments. Ask 3-5 fiction writers to rate 40 passages (20 human, 20 LLM, blinded) on a 1-5 quality scale. Correlate our metrics with their ratings.

**Success criterion:** Pearson r > 0.4 between our composite metric and human quality ratings, regardless of authorship.

### Experiment 2: Psychic Distance — Ground Truth Validation

**Question:** Does our psychic distance scorer agree with human annotators on Gardner's scale?

**Design:**
- Select 100 sentences spanning all five Gardner levels:
  1. "It was winter of the year 1853. A large man stepped out of a doorway." (distant)
  2. "Henry J. Warburton had never liked the month of December." (named character)
  3. "Henry hated December." (close third)
  4. "God how he hated these damn December nights." (very close)
  5. "Cold. Knee-deep snow. Fingers numb. What the hell am I doing here?" (interior)
- Have 5 annotators rate each sentence on 1-5 distance scale
- Score each with our psychic_distance.score_sentence()
- Compute correlation

**Data needed:** Sentence corpus with human distance annotations. Can be constructed from existing fiction — sample sentences that clearly represent each level.

**Success criterion:** Spearman r > 0.6 between our score and human annotations. Inter-annotator agreement (Krippendorff's alpha) > 0.7.

### Experiment 3: Discriminator Power — Expanded Baselines

**Question:** Do our six discriminators hold up with more data and more diverse authors?

**Current weakness:** Our baselines are 5 authors, 50 chapters. Three are 19th century. The LLM corpus is 10 chapters from 5 models. This is underpowered.

**Design:**
- Expand human baseline to 200+ chapters from 20+ authors:
  - 10 contemporary (CC-licensed or newly PD: Doctorow, Watts, Faulkner 1929-30, Hemingway 1929, Hammett 1930, plus BCS stories)
  - 10 classic (Austen, Dickens, Conrad, Shelley, Brontë, Eliot, Hardy, Twain, Woolf, James)
- Expand LLM corpus to 50+ chapters from 10+ models
- Re-run all metrics, compute Cohen's d with confidence intervals
- Bootstrap 95% CIs on the effect sizes
- Report which metrics survive at d > 0.8 with the expanded data

**Success criterion:** At least 4 of our 6 discriminators maintain d > 0.8 with 95% CI not crossing zero.

### Experiment 4: The Revision Loop — Controlled Improvement Study

**Question:** Does the critique-and-revise loop produce measurable and consistent improvement?

**Current evidence:** 3 stories, each revised once. Metrics moved toward baseline in 4-7 of 8 cases. Small N, no controls.

**Design:**
- Generate 20 stories from diverse prompts (varied genre, POV, register)
- Split into 4 groups of 5:
  - Group A: Revise with full prose-doctor critique
  - Group B: Revise with generic "improve the prose" instruction (no specific metrics)
  - Group C: Revise with only the structural prescriptions (inversion, fragments) — no information-theoretic ones
  - Group D: No revision (control)
- Score all 80 texts (20 originals + 60 revisions) on all metrics
- Also get human quality ratings (blinded, 3-5 raters, 1-5 scale)

**Hypotheses:**
- H1: Group A improves more than Group B on structural metrics
- H2: Group A does NOT improve more than Group B on information-theoretic metrics (rhythmicity, surprisal)
- H3: Group C improves as much as Group A on structural metrics (the information-theoretic prescriptions add nothing)
- H4: Human quality ratings improve for Group A but NOT for Group D

**Success criterion:** Statistically significant improvement (p < 0.05, paired t-test) in Group A vs Group D on at least 3 metrics. H2 and H3 confirmed would demonstrate the structural vs information-theoretic boundary.

### Experiment 5: Prompt Engineering Effect — Does Knowledge Transfer?

**Question:** Do generation prompts informed by prose-doctor metrics produce better first drafts than uninformed prompts?

**Current evidence:** Anecdotal — Grave Shift (prompt 3) had better metrics than Ablation (prompt 1). But confounded by different stories, different complexity, small N.

**Design:**
- Use 10 identical story premises
- For each premise, generate with two prompts:
  - Prompt A: bare premise only ("write a 3000 word story about X")
  - Prompt B: premise + metric-informed technique instructions ("40%+ inversions, fragments only at impact, vary density, shift ending register")
- Same model, same temperature, same parameters
- Score all 20 stories
- Paired comparison on each metric

**Success criterion:** Prompt B produces statistically better scores on at least 4 of 6 discriminators (paired Wilcoxon signed-rank, p < 0.05).

### Experiment 6: Ending Register — Validation at Scale

**Question:** Is the ending register shift a real human/LLM discriminator or an artifact of our small sample?

**Design:**
- Collect 50 human chapter endings (actual chapter breaks, not word-count chunks)
- Collect 50 LLM chapter endings
- Compute both mean shift and range for the final 15% vs middle 60%
- Compare distributions

**Current weakness:** Our human baselines were arbitrary 4000-word chunks, not real chapter breaks. This confounds the finding.

**Data needed:** Human fiction with actual chapter structure preserved. Standard Ebooks are chapter-delimited — parse the XHTML to extract real chapters.

**Success criterion:** Cohen's d > 0.5 between human and LLM ending range distributions.

### Experiment 7: Sensory Probe — Is It Actually Useful?

**Question:** Does the sensory profiler provide actionable information?

**Current evidence:** The profiler produces scores, but the baseline comparison showed human and LLM prose have identical sensory balance (0.992 both). If it can't discriminate, what's it for?

**Design:**
- Run profiler at paragraph level (not chapter level) on 20 human and 20 LLM chapters
- Check if variance WITHIN chapters differs (humans might vary more per-paragraph even if chapter means are identical)
- Check if sensory deserts correlate with human quality ratings (readers prefer chapters without long sensory gaps)
- If neither holds, the profiler is not useful and should be documented as such

**Success criterion:** Either (a) within-chapter sensory variance discriminates (d > 0.5) or (b) sensory desert count correlates with quality ratings (r > 0.3). If neither, report negative result.

### Experiment 8: Retexture Quality — Does Cydonia Help?

**Question:** Do the retexture fragments actually improve prose when incorporated?

**Design:**
- Take 20 identified flat passages
- For each, create three versions:
  - Original (no change)
  - Retextured with Cydonia fragments (shaped to fit)
  - Retextured by a general-purpose LLM (same instruction, no Cydonia)
- Have 5 human raters rank the three versions (blinded) on vividness, specificity, and fit-with-context

**Success criterion:** Cydonia-retextured passages ranked first > 50% of the time. Or if not, document that general-purpose LLM retexture is equally effective (which would simplify the pipeline).

## Data Requirements Summary

| Experiment | Human texts needed | LLM texts needed | Human raters needed | Effort |
|-----------|-------------------|------------------|--------------------|----|
| 1. Metric validation | 10 bad passages | 10 good passages | 3-5 raters, 40 ratings each | Medium |
| 2. Psychic distance | 100 annotated sentences | — | 5 raters, 100 ratings each | Medium |
| 3. Discriminator power | 200 chapters, 20 authors | 50 chapters, 10 models | — | High (data collection) |
| 4. Revision loop | — | 20 stories × 4 conditions | 3-5 raters, 80 ratings each | High |
| 5. Prompt engineering | — | 20 stories (10 pairs) | — | Low |
| 6. Ending register | 50 real chapters | 50 chapters | — | Medium |
| 7. Sensory probe | 20 chapters | 20 chapters | 3-5 raters | Medium |
| 8. Retexture quality | — | 20 passages × 3 versions | 5 raters, 60 rankings | Medium |

## Priority Order

1. **Experiment 5** (prompt engineering) — lowest effort, highest immediate value, answers "does this actually help?"
2. **Experiment 2** (psychic distance validation) — validates our most novel claim
3. **Experiment 4** (revision loop) — the core value proposition, needs the most design but proves the pipeline
4. **Experiment 3** (expanded baselines) — strengthens all other findings, can run unattended
5. **Experiment 1** (metric validation) — requires human raters but answers the key question
6. **Experiment 6** (ending register) — quick to run once we have real chapter data
7. **Experiment 7** (sensory probe) — might be a negative result, still worth reporting
8. **Experiment 8** (retexture quality) — nice to have, not essential for the paper
