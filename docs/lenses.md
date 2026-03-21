# Lens Guide

This document explains what each prose-doctor lens measures, why it is useful, how to use it in this repo, what research backs it, and where the lens is still heuristic or experimental.

## How the lens system works

A lens is a focused analytical perspective over prose. Every lens returns a `LensResult` with some combination of sentence-, paragraph-, and chapter-level signals. The registry lives in `src/prose_doctor/lenses/defaults.py`, the shared output shape lives in `src/prose_doctor/lenses/__init__.py`, and the current maturity assignments live in `src/prose_doctor/validation/tiers.toml`.

The repo distinguishes three maturity tiers:

- `experimental`: the lens runs and measures something, but it is not yet proven useful for revision. May be consumed by meta-lenses.
- `validated`: the lens has good discrimination stats (Cohen's d ≥ 0.5 or p < 0.01) AND revision evidence showing edits guided by it are accepted — proof it meaningfully improves text.
- `stable`: rock solid. Good discrimination, extensive revision evidence (3+ accepted edits across 2+ chapters).

The promotion rules are documented in `src/prose_doctor/validation/promotion.py`, and the current evidence summary is in `docs/specs/2026-03-20-lens-architecture.md`.

## Current access surface

The registry contains more lenses than the CLI currently exposes directly.

- `prose-doctor scan --deep` currently reports: slop scoring, foregrounding, perplexity, emotion arc, psychic distance, information contour, sensory, dialogue voice, and pacing
- Dedicated CLI commands exist for: `index`, `classify`, `distance`, `contour`, `sensory`, and `validate`
- Registry-only lenses are best accessed programmatically today: `boyd_narrative_role`, `fragment_classifier`, `narrative_attention`, and `uncertainty_reduction`

Programmatic access looks like this:

```python
from prose_doctor.lenses.defaults import default_registry
from prose_doctor.lenses.runner import LensRunner
from prose_doctor.providers import ProviderPool

runner = LensRunner(default_registry(), ProviderPool())
result = runner.run_one("boyd_narrative_role", text, "chapter.md")
print(result.per_chapter)
```

## Lens Summary

| Lens | Tier | Best current access | Primary use |
| --- | --- | --- | --- |
| `pacing` | experimental | `scan --deep` | scene-balance problems, talking heads, action deserts |
| `emotion_arc` | experimental | `scan --deep` | flat emotional trajectories across a chapter |
| `foregrounding` | stable | `index`, `scan --deep` | literary texture and stylistic markedness |
| `info_contour` | stable | `contour`, `scan --deep` | information-density rhythm and flatlines |
| `psychic_distance` | stable | `distance`, `scan --deep` | narrative zoom and abrupt viewpoint jumps |
| `sensory` | validated | `sensory`, `scan --deep` | sensory balance and sensory deserts |
| `dialogue_voice` | validated | `scan --deep` | character voice separation and talking heads |
| `slop_classifier` | stable | `classify`, `scan --deep` | project-specific LLM-prose pattern detection |
| `perplexity` | validated | `scan --deep` | overly smooth, overly predictable prose |
| `uncertainty_reduction` | experimental | Python, `validate` | paragraph-boundary uncertainty shifts |
| `boyd_narrative_role` | validated | Python, `validate` | staging vs progression vs tension balance |
| `fragment_classifier` | stable | Python, `validate` | deliberate fragments vs filler fragments |
| `narrative_attention` | validated | Python, `validate` | cross-lens structural coherence |
| `concreteness` | validated | Python, `validate` | semantic vagueness and abstract-concrete balance |
| `referential_cohesion` | experimental | Python, `validate` | entity continuity and referent churn |
| `situation_shifts` | experimental | Python, `validate` | time/space/actor transitions at paragraph boundaries |
| `discourse_relations` | experimental | Python, `validate` | connective diversity and additive-only zones |

---

## Lens Reference

### `psychic_distance` (`stable`)

**What it is.** Estimates narrative zoom level per sentence and paragraph. Combines pronoun types, perception verbs (saw, felt, heard), cognition verbs (thought, realized, wondered), deictic markers (this/here vs that/there), tense, and sensory concreteness (via a trained embedding probe) to derive `pd_mean`, `pd_std`, a qualitative label, and `zoom_jumps`.

**Why it is useful.** This is a viewpoint-control lens. It catches prose that slips between camera distance and interior access without the writer noticing, or chapters that feel emotionally detached even though the scene content should feel intimate.

**How to use it.**
- Run `prose-doctor distance chapter.md` or `scan --deep`
- Read `pd_mean` as the average zoom level; `zoom_jumps` is the higher-value revision signal
- Fix accidental shifts first; only then decide whether the chapter should overall move closer or farther

**Research backing.**
- **Gardner (1984)**, *The Art of Fiction* — defines psychic distance as "the distance the reader feels between himself and the events of the story," proposing a 5-level continuum from establishing shot to deep interiority.
- **Genette (1980)**, *Narrative Discourse* — formalizes focalization (zero, internal, external). Distinguishes "who sees" from "who speaks," which maps onto the close/distant axis this lens measures.
- **Stockwell (2002)**, *Cognitive Poetics* — Chapter 4 applies deictic shift theory (DST) to narrative, showing how proximal/distal deictics track the reader's psychological position. Directly informs the deictic scoring component.
- **Duchan, Bruder & Hewitt (1995)**, *Deixis in Narrative* — foundational collection on DST, establishing the "push/pop" model of deictic fields in narrative.
- Perception/cognition verb density as a marker of interiority is well-established in narratology (Genette's internal focalization) and supported by corpus studies of free indirect discourse. Experimental work shows narrative POV materially affects reader alignment and perspective-taking ([Holler, Salem & Weskott 2017](https://doaj.org/article/31dfc84d890e4bf2bda11a14bf519f5e)).
- **Project-internal validation:** Cohen's d = 1.78 discriminating human vs LLM prose on pd_mean. Human baseline: pd_mean = 0.336, pd_std = 0.093 (50-chapter corpus).

**Caveats.**
- Gardner's scale is craft pedagogy, not an empirically derived taxonomy. Omniscient narration, second person, and experimental forms may not map cleanly onto the 5-level model.
- The scoring weights (0.20 pronoun, 0.25 perception, 0.10 deictic, 0.15 sensory, 0.10 tense, 0.20 interiority) are heuristic, calibrated against close-third-person fiction.
- Dialogue content is not separated from narration, so quoted speech can distort paragraph-level scores.
- The zoom-jump detector assumes smooth transitions are desirable, which may not hold for modernist or fragmentary styles.
- No published computational validation of Gardner's psychic distance scale exists. This lens is novel.

---

### `info_contour` (`stable`)

**What it is.** Uses GPT-2 sentence-level surprisal to estimate how information density rises and falls through a chapter. Reports `mean_surprisal`, `cv_surprisal`, `dominant_period`, `rhythmicity`, `spectral_entropy`, plus `flatlines` and `spikes`.

**Why it is useful.** Catches prose that is too even in its informational load. Chapters can fail not because every sentence is bad, but because the cadence of surprise never changes. LLM prose tends toward Uniform Information Density; human prose does not.

**How to use it.**
- Run `prose-doctor contour chapter.md` or `scan --deep`
- Treat `flatlines` as the highest-priority signal
- Use `rhythmicity` and `spectral_entropy` comparatively across chapters, not as absolute craft scores
- Inspect `spike_details` in `raw` if a chapter feels jagged rather than flat

**Research backing.**
- **Hale (2001)**, "A Probabilistic Earley Parser as a Psycholinguistic Model," *NAACL* — establishes surprisal theory: processing difficulty at a word is proportional to −log P(w|context).
- **Levy (2008)**, "Expectation-Based Syntactic Comprehension," *Cognition* 106(3), DOI: [10.1016/j.cognition.2007.05.006](https://doi.org/10.1016/j.cognition.2007.05.006) — extensive empirical validation that surprisal predicts reading times across syntactic structures.
- **Levy & Jaeger (2007)**, "Speakers Optimize Information Density Through Syntactic Reduction," *NeurIPS 20* — proposes the Uniform Information Density (UID) hypothesis. LLM prose tends toward UID; fiction writing intentionally violates it for dramatic effect.
- **Tsipidi et al. (2024)**, "Surprise! Uniform Information Density Isn't the Whole Story," *EMNLP 2024*, [aclanthology.org/2024.emnlp-main.1047](https://aclanthology.org/2024.emnlp-main.1047/) — shows information rate fluctuates in structured, predictable ways in natural discourse. Proposes the Structured Context Hypothesis. The direct inspiration for the FFT rhythmicity analysis.
- GPT-2 surprisal correlates with human reading times (Wilcox et al. 2020, "On the Predictive Power of Neural Language Models," *CogSci 2020*).
- **Project-internal validation:** Cohen's d = −1.52 on rhythmicity. LLM prose has higher rhythmicity (more periodic, less varied) than human prose. Human baseline: rhythmicity = 0.129, spikes = 7.7/chapter, flatlines = 3.1/chapter.

**Caveats.**
- GPT-2 is a 2019 model. Its surprisal reflects WebText, not modern prose norms. Archaic vocabulary or domain-specific terms produce artificially high surprisal.
- FFT assumes stationarity; surprisal contours in fiction are inherently non-stationary (climax vs exposition have different base rates).
- Tsipidi et al. (2024) show that position within a document is a significant predictor of surprisal, which this lens does not control for.
- Short texts (<500 words) lack sufficient data for reliable FFT analysis.
- The FFT rhythmicity analysis is a project-specific extension of surprisal theory — the surprisal foundation is stronger than the exact periodicity heuristic.

---

### `foregrounding` (`stable`)

**What it is.** Measures stylistic markedness across five axes: alliteration density, syntactic inversion percentage, sentence-length variation (CV), fragment ratio, and unexpected collocations (semantically distant word pairs in syntactic dependency). Returns a composite `index`, the weakest axis, and a revision prescription.

**Why it is useful.** The best single lens for chapters that feel competent but bland. It turns "the prose is flat" into specific levers: rhythm, syntax, fragments, sound patterning, and lexical surprise.

**How to use it.**
- Run `prose-doctor index chapter.md` for a focused view, or `scan --deep`
- Read the composite `index`, then the weakest axis
- Use axis scores to pick one stylistic revision pass at a time instead of trying to increase every score at once

**Research backing.**
- **Shklovsky (1917)**, "Art as Technique" — introduces defamiliarization (ostranenie): art's purpose is to make the familiar strange, restoring perception.
- **Mukarovský (1964)**, "Standard Language and Poetic Language" — defines foregrounding as systematic deviation from the linguistic norm. Distinguishes automatization (backgrounded, habitual language) from foregrounding (language that draws attention).
- **van Peer (1986)**, *Stylistics and Psychology* — first empirical validation. Reader ratings of "strikingness" correlate significantly with independently coded foregrounding density (p < 0.01).
- **Miall & Kuiken (1994)**, "Foregrounding, Defamiliarization, and Affect," *Poetics* 22, pp. 389–407 — four studies showing foregrounded segments produce longer reading times, higher strikingness ratings, and higher affect ratings, independent of literary training. Effect sizes moderate to large.
- Sentence-length variation as a text quality measure has parallels in Coh-Metrix (McNamara et al. 2014, *Automated Evaluation of Text and Discourse with Coh-Metrix*).
- **Project-internal validation:** Cohen's d = 1.34 on inversion_pct. Human baseline: inversion_pct = 44.2%, sl_cv = 0.706, fragment_pct = 6.7%.

**Caveats.**
- The five-axis decomposition is novel to prose-doctor. Each axis individually has support in stylistics, but their weighted combination (alliteration 0.15, inversion 0.15, sl_cv 0.25, fragments 0.15, collocations 0.30) is heuristic, not empirically derived.
- The composite index assumes more foregrounding = better prose. Excessive foregrounding can be as problematic as too little (purple prose).
- Inversion detection relies on spaCy dependency parsing (~95% accuracy). Fragment ratio treats all <5-token sentences as fragments, catching false positives from dialogue tags.
- Unexpected collocations depend on sentence-transformer embedding space geometry — words that are semantically distant in embedding space may not feel "unexpected" to readers.

---

### `sensory` (`validated`)

**What it is.** Scores text across six perceptual modalities: visual, auditory, haptic, olfactory, gustatory, and interoceptive. Reports chapter-level modality means, a `balance_ratio`, strongest and weakest modalities, and `deserts` where sensory signal drops for too long. Uses a trained linear probe on `all-mpnet-base-v2` (768d) sentence embeddings — NOT the shared sentence-transformer provider.

**Why it is useful.** Catches prose that is visually competent but bodily thin — chapters that describe what can be seen while neglecting sound, touch, smell, taste, and internal sensation.

**How to use it.**
- Run `prose-doctor sensory chapter.md` or `scan --deep`
- Prioritize `deserts` over small differences in chapter-level averages
- Look for a weak modality that matches the scene type
- Use the prescription to add one missing channel rather than trying to maximize all six

**Research backing.**
- **Lynott et al. (2020)**, "The Lancaster Sensorimotor Norms," *Behavior Research Methods* 52, pp. 1271–1291, DOI: [10.3758/s13428-019-01316-z](https://doi.org/10.3758/s13428-019-01316-z) — human-rated sensory modality norms for 39,707 words across 6 perceptual and 5 action dimensions, from 3,500 participants. Freely available at [osf.io/7emr6](https://osf.io/7emr6/). Conceptual basis for the 6-modality decomposition. Visual modality dominates (mean 2.79/5), with olfactory (0.73) and gustatory (0.81) reliably weakest.
- **Winter (2019)**, *Sensory Linguistics*, John Benjamins — comprehensive treatment of how language encodes perceptual experience; documents the "visual bias" in language.
- **Lacey, Stilla & Sathian (2012)**, "Metaphorically Feeling," *Brain and Language* 120(3), DOI: [10.1016/j.bandl.2011.12.016](https://doi.org/10.1016/j.bandl.2011.12.016) — reading sensory language activates corresponding cortical areas.
- The probe is a data-free distillation: 768d embeddings → 96d hidden → 6d sigmoid, trained on pseudo-labels from embedding direction vectors. No direct validation against Lancaster norms.

**Caveats.**
- The probe was trained on pseudo-labels (embedding direction vectors), not human sensory judgments. Agreement with Lancaster norms is assumed but not measured.
- Word-level scoring loses phrase-level sensory information. "The crack of a rifle" is auditory at the phrase level but may not score highly from individual word embeddings.
- Interoception (internal body states) is the least validated modality — Lynott et al. (2020) note it was a novel addition with less theoretical grounding.
- Loads its own `all-mpnet-base-v2` model separately from the shared 384d provider, requiring additional GPU memory.

---

### `dialogue_voice` (`validated`)

**What it is.** Extracts dialogue lines, heuristically groups them by speaker, embeds each speaker's lines, and estimates whether the cast sounds distinct. Reports `speaker_separation`, `dialogue_ratio`, `talking_heads_count`, speaker similarities, and a prescription if everyone sounds alike.

**Why it is useful.** Catches scenes where every speaker shares the same cadence, vocabulary, and sentence shape — a signature of LLM-generated dialogue. Also flags dialogue-heavy sections with no embodied interruption.

**How to use it.**
- Run it through `scan --deep`
- Use `speaker_separation` comparatively across scenes with multiple named speakers
- Inspect `speaker_similarities` to see which pair of characters is collapsing together
- Treat `talking_heads_count` as a staging problem, not a dialogue problem

**Research backing.**
- **Burrows (1987)**, *Computation into Criticism*, Oxford University Press — pioneering computational stylistics work demonstrating that characters in Austen's novels have statistically distinguishable word-frequency profiles.
- **Rybicki & Eder (2011)**, "Deeper Delta across Genres and Languages," *Literary and Linguistic Computing* 26(3) — extends Burrows's Delta method, showing function-word frequencies discriminate styles across languages and genres.
- **Reimers & Gurevych (2019)**, "Sentence-BERT," *EMNLP 2019*, [aclanthology.org/D19-1410](https://aclanthology.org/D19-1410/) — the embedding method for dialogue vectorization.
- **Bamman, Underwood & Smith (2014)**, "A Bayesian Mixed Effects Model of Literary Character," *ACL 2014* — computational model for characterizing literary characters through their language.

**Caveats.**
- Speaker attribution is regex-based (said/asked/replied + proper noun). Misses unattributed dialogue, non-standard tags, and multi-speaker paragraphs. More principled alternatives exist (BookNLP, booknlp/booknlp, Apache 2.0).
- Embedding-based separation conflates topic with voice. Characters discussing different subjects appear more distinct than they actually are stylistically.
- The separation threshold (< 0.15 = "all same voice") is heuristic, not empirically derived.
- Requires at least 2 speakers with 3+ attributed lines each.

---

### `slop_classifier` (`stable`)

**What it is.** A supervised prose-pattern detector combining a ModernBERT classifier (fine-tuned on 13K paragraphs from 8 LLM models) with rule-based pattern overrides. Returns paragraph-level probabilities, predicted classes, and rule hits.

**Why it is useful.** The most direct lens for the specific failure modes prose-doctor cares about: thesis statements, narrator glosses, generic emotion naming, padding, dead figurative language, and related LLM-style habits.

**How to use it.**
- Run `prose-doctor classify chapter.md` or `scan --deep`
- Read the predicted class and supporting rule hits together
- Use it as revision triage, not proof-of-authorship detection

**Research backing.**
- **Devlin et al. (2019)**, "BERT," *NAACL 2019*, [aclanthology.org/N19-1423](https://aclanthology.org/N19-1423/) — foundational encoder-only transformer architecture. ModernBERT is a 2024 update with RoPE embeddings and 8192-token context.
- **Warner et al. (2024)**, "ModernBERT," Answer.AI / LightOn.AI, [huggingface.co/answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) — the specific encoder model, trained on 2 trillion tokens.
- **Gehrmann, Strobelt & Rush (2019)**, "GLTR," *ACL 2019 Demos*, [aclanthology.org/P19-3019](https://aclanthology.org/P19-3019/) — related AI text detection using GPT-2 probability distributions. Improved human detection from 54% to 72%.
- The broader premise that machine-generated text is detectable via model-based methods is consistent with generated-text detection literature ([Mitchell et al. 2023](https://proceedings.mlr.press/v202/mitchell23a.html), "DetectGPT," *ICML 2023*).
- Model card: [dt9k/prose-doctor-slop-classifier](https://huggingface.co/dt9k/prose-doctor-slop-classifier).

**Caveats.**
- This lens is best read as a project-taxonomy classifier, not a universal human-vs-AI detector.
- Trained on outputs from 8 specific LLM models — may not generalize to new architectures or fine-tuned models.
- "Slop" is a subjective, community-defined concept. The taxonomy encodes one author's aesthetic judgments.
- Context-aware scoring (prev/current/next paragraph) requires sequential paragraph ordering; shuffled paragraphs lose signal.
- May flag non-native English prose or formulaic genre fiction (romance, thriller) as AI-generated.

---

### `perplexity` (`validated`)

**What it is.** Runs GPT-2 over paragraphs and reports `mean_ppl`, `pct_below_55`, and the `smoothest_paragraphs`.

**Why it is useful.** A coarse measure of how predictable the wording is to a language model. Low perplexity is a warning sign for over-smoothed prose: lines that are fluent but too easy, too templated, or too unsurprising.

**How to use it.**
- Run it through `scan --deep`
- Use `mean_ppl` for whole-chapter comparison
- Use `smoothest_paragraphs` for concrete revision targets
- Do not treat high perplexity as automatically good; nonsense is also surprising

**Research backing.**
- **Radford et al. (2019)**, "Language Models are Unsupervised Multitask Learners," OpenAI Technical Report, [link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 model paper.
- **Gehrmann, Strobelt & Rush (2019)**, "GLTR," *ACL 2019* — demonstrates that low perplexity under a language model is a signal of machine-generated text.
- **Mitchell et al. (2023)**, "DetectGPT," *ICML 2023* — more sophisticated approach but same principle: model-generated text occupies low-perplexity regions.
- **Holtzman et al. (2020)**, "The Curious Case of Neural Text Degeneration," *ICLR 2020* — shows maximizing likelihood too aggressively produces bland, repetitive output diverging from human text distributions.

**Caveats.**
- GPT-2 perplexity is a proxy, not a direct quality measure. Low perplexity can indicate either AI generation or highly conventional human prose.
- As models improve, the perplexity gap between human and AI text narrows.
- Genre-dependent: formulaic genre fiction naturally has lower perplexity.
- GPT-2's training data (2019 web text) is increasingly outdated as a baseline.

---

### `pacing` (`experimental`)

**What it is.** Classifies each paragraph into one dominant mode — `dialogue`, `action`, `interiority`, or `setting` — then reports mode ratios, long runs, `talking_heads`, `action_deserts`, and `interiority_gaps`.

**Why it is useful.** A scene-balance lens. Useful when a chapter feels static, too talky, or emotionally under-internalized, but you don't know whether the problem is lack of grounding, lack of motion, or lack of access to the character's head.

**How to use it.**
- Run it through `scan --deep`
- Trust the run and gap flags more than the exact ratios
- Use `talking_heads` to find stretches needing gesture, blocking, or sensory grounding
- Treat thresholds as diagnostics, not universal target counts

**Research backing.**
- **Genette (1980)**, *Narrative Discourse* — defines narrative speed as the relationship between discourse time and story time. The dialogue/action/interiority/setting decomposition is a pragmatic simplification of Genette's summary/scene/pause/ellipsis framework.
- **Brunner (2013)**, "Automatic Recognition of Speech, Thought, and Writing Representation," *Literary and Linguistic Computing* 28(4), DOI: [10.1093/llc/fqt024](https://doi.org/10.1093/llc/fqt024) — achieved F1 = 0.87 for direct speech detection, 0.40 for free indirect discourse, demonstrating feasibility and difficulty of automated mode classification.
- **Jockers (2013)**, *Macroanalysis* — computational approaches to literary analysis at scale.

**Caveats.**
- This is explicitly experimental. Interiority detection needs threshold rework — in batch testing it produced zero hits.
- The keyword-based classifier has no learned component. Action verbs are a closed set of ~32 words.
- Priority order (dialogue > interiority > action > setting) means paragraphs with both dialogue and action are always classified as dialogue.
- No handling of scene breaks. Mode transitions across scene breaks are treated the same as within-scene transitions.
- No published paragraph-level pacing distributions for English fiction exist.

---

### `emotion_arc` (`experimental`)

**What it is.** Samples paragraphs across a chapter, scores them with a DistilBERT sentiment model (SST-2), and summarizes as `dynamic_range`, `mean_sentiment`, `peaks`, `valleys`, and a `flat` flag.

**Why it is useful.** A coarse pass for chapters whose emotional motion is too even. Detects chapters that stay tonally flat from opening to close even when plot events are changing.

**How to use it.**
- Run it through `scan --deep`
- Look first at `flat` and `dynamic_range`
- Treat the arc string as a shape summary, not a plot summary
- Use it comparatively across drafts or chapters, not as a stand-alone verdict

**Research backing.**
- **Reagan et al. (2016)**, "The Emotional Arcs of Stories Are Dominated by Six Basic Shapes," *EPJ Data Science* 5(1), DOI: [10.1140/epjds/s13688-016-0093-1](https://doi.org/10.1140/epjds/s13688-016-0093-1) — analyzed 1,327 Project Gutenberg texts, identified six core arc shapes. Validated via matrix decomposition, supervised learning, and unsupervised clustering. Most-downloaded stories exhibit "Man in a Hole" and "Rags to Riches" arcs.
- **Socher et al. (2013)**, "Recursive Deep Models for Semantic Compositionality," *EMNLP 2013* — introduces SST with 215,154 phrase-level sentiment annotations. SST-2 is the training data for the DistilBERT model this lens uses.
- **Sanh et al. (2019)**, "DistilBERT," *NeurIPS Workshop* — 97% of BERT's understanding at 40% smaller, 60% faster. 91.3% accuracy on SST-2.

**Caveats.**
- SST-2 was trained on movie reviews, not narrative prose. Prose like "The walls were grey and the air smelled of rust" may not align with review sentiment.
- Binary positive/negative classification collapses emotional nuance. Fear, anger, sadness, and disgust all map to "negative."
- Sampling every 3rd paragraph loses fine-grained arc structure.
- The intensity mapping treats high-confidence negative sentiment as high intensity, conflating valence with arousal.
- Orthogonal feature in this project (max |r| = 0.19 with other metrics) — limited discrimination power on its own.

---

### `uncertainty_reduction` (`experimental`)

**What it is.** Measures how the entropy of GPT-2's next-token distribution changes at paragraph boundaries. Positive values mean the new paragraph resolves uncertainty; negative values mean it opens more possibilities.

**Why it is useful.** An exploratory discourse lens for paragraph transitions. Useful when you want to know whether transitions are closing questions too quickly, never resolving them, or oscillating arbitrarily.

**How to use it.**
- Access through Python or `prose-doctor validate uncertainty_reduction`
- Compare paragraph-to-paragraph shifts rather than chasing a single ideal chapter mean
- Use as a secondary diagnostic alongside `info_contour`, not a replacement

**Research backing.**
- **Wilmot & Keller (2020)**, "Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation," *ACL 2020*, [aclanthology.org/2020.acl-main.161](https://aclanthology.org/2020.acl-main.161/) — core paper. Finds uncertainty reduction over neural story representations is the best predictor of human suspense judgments, achieving near-human accuracy. Outperforms backward-looking surprise.
- **Hale (2006)**, "Uncertainty About the Rest of the Sentence," *Cognitive Science* 30(4), DOI: [10.1207/s15516709cog0000_64](https://doi.org/10.1207/s15516709cog0000_64) — entropy reduction as a model of incremental comprehension.
- **Frank (2013)**, "Uncertainty Reduction as a Measure of Cognitive Load," *Topics in Cognitive Science* 5(3), DOI: [10.1111/tops.12025](https://doi.org/10.1111/tops.12025) — validates entropy reduction as predictor of cognitive processing effort above and beyond simple surprisal.
- Open-source: Wilmot's **Story-Untangling**, [github.com/dwlmt/Story-Untangling](https://github.com/dwlmt/Story-Untangling).

**Caveats.**
- The original model (Wilmot & Keller) uses a hierarchical language model. Adapting to off-the-shelf GPT-2 is an approximation.
- Paragraph-boundary measurement loses within-paragraph suspense dynamics.
- GPT-2's entropy reflects model uncertainty, not necessarily reader uncertainty.
- Context window of ~50 tokens per boundary is a heuristic.
- Project-internal: Cohen's d = 0.31 — weak signal. Experimental tier.

---

### `boyd_narrative_role` (`validated`)

**What it is.** Scores each paragraph for `staging`, `progression`, and `tension` using lightweight function-word sets and reports the dominant mode by paragraph.

- **Staging**: spatial/temporal words, articles, prepositions (the, a, in, on, at, through, before, after, during)
- **Progression**: causal/sequential connectives, action adverbs (then, so, because, suddenly, began, started, continued)
- **Tension**: negation, intensifiers, contrast markers (not, never, but, however, despite, against, without)

**Why it is useful.** Diagnoses scenes that over-stage without advancing, advance without building pressure, or stay tense without enough spatial and causal orientation.

**How to use it.**
- Access through Python or `prose-doctor validate boyd_narrative_role`
- Compare the dominant mode sequence across a chapter
- Ask structural questions: where is the scene setting the board, where is it moving, and where is it tightening pressure?

**Research backing.**
- **Boyd, Blackburn & Pennebaker (2020)**, "The Narrative Arc: Revealing Core Narrative Structures Through Text Analysis," *Science Advances* 6(32), DOI: [10.1126/sciadv.aba2196](https://doi.org/10.1126/sciadv.aba2196) — core paper. Analyzed ~40,000 traditional and ~20,000 nontraditional narratives using LIWC function-word categories. Found three primary processes (staging, progression, cognitive tension) following a consistent arc: staging high at beginning, progression peaks mid-story, tension peaks at climax.
- **Pennebaker et al. (2015)**, "The Development and Psychometric Properties of LIWC2015" — the word-counting tool underlying Boyd's analysis. LIWC categories are psychologically validated for function words.
- **Tausczik & Pennebaker (2010)**, "The Psychological Meaning of Words," *JLSP* 29(1), DOI: [10.1177/0261927X09351676](https://doi.org/10.1177/0261927X09351676) — validates function words as reliable markers of cognitive and social processes.
- Project-internal: ANOVA F = 4.39** confirms scene differentiation on staging.

**Caveats.**
- Pure word-counting ignores syntax and semantics. "Before the war" (staging) vs "never before" (tension) get classified identically.
- Boyd et al.'s three-factor model was derived from chapter/segment-level analysis. Paragraph-level classification operates at a finer granularity than validated.
- Function-word sets are hand-curated. Coverage and boundaries are debatable.
- The arc is most consistent in traditional narrative forms; experimental or non-linear narratives may not follow the expected pattern.
- Open alternative: [Arc of Narrative](https://www.arcofnarrative.com/) (Boyd's group) implements the narrative arc analysis.

---

### `fragment_classifier` (`stable`)

**What it is.** Finds very short sentences and classifies them as likely `craft` or likely `crutch`. Reports `craft_count`, `crutch_count`, `fragment_pct`, and the reasons behind each decision.

- **Craft signals**: part of a 3+ fragment sequence (rhythmic), contains concrete sensory detail, dialogue fragments, em-dash constructions, anaphoric echoes
- **Crutch signals**: vague abstractions ("Something shifted," "The horizon shimmered"), no named characters or body-part words, generic atmospheric sentences

**Why it is useful.** Not all fragments are bad. This lens separates deliberate staccato effect, montage, and echo from generic filler fragments that LLMs produce by default.

**How to use it.**
- Access through Python or `prose-doctor validate fragment_classifier`
- Read the `raw["fragments"]` reasons before revising
- Preserve fragments doing rhythmic or dramatic work
- Cut or rewrite isolated vague fragments first

**Research backing.**
- **Tufte (2006)**, *Artful Sentences: Syntax as Style*, Graphics Press — catalogs sentence fragments as deliberate stylistic devices in published prose. The primary craft reference for understanding when fragments work.
- **Lanham (2003)**, *Analyzing Prose*, 2nd ed., Continuum — prose style analysis framework including sentence architecture classification.
- **Miall & Kuiken (1994)** — foregrounded (marked) syntax increases salience and affect.
- **Feng, Banerjee & Choi (2012)**, "Characterizing Stylistic Elements in Syntactic Structure," *EMNLP 2012*, [aclanthology.org/D12-1139](https://aclanthology.org/D12-1139/) — syntactic patterning carries stylistic signal.
- No published computational validation of the craft-vs-crutch distinction exists. The classification is a project heuristic, currently treated as stable based on revision evidence.

**Caveats.**
- Binary craft/crutch oversimplifies a spectrum. Fragments can be partially intentional, contextually appropriate but stylistically weak, or ambiguous.
- Context dependence is extreme. "Gone." after a character death is craft; the same word mid-paragraph is likely an error.
- "Concrete detail" heuristic uses body-part word lists and named character detection — may miss other forms of specificity.

---

### `narrative_attention` (`validated`)

**What it is.** A meta-lens. Does not read raw text directly. Consumes other lens outputs (`psychic_distance`, `info_contour`, `foregrounding`, `emotion_arc`, `boyd_narrative_role`, `uncertainty_reduction`), builds a per-paragraph feature matrix, computes cosine-similarity "attention" across paragraphs, and summarizes with `coherence` and per-paragraph `attention_entropy`.

**Why it is useful.** A structural integration lens. Useful when no single metric looks catastrophic, but the chapter still feels as if its internal signals drift, reset, or fail to lock together.

**How to use it.**
- Access through Python or `prose-doctor validate narrative_attention`
- Run only after its dependencies are available
- Read low `coherence` as a prompt to inspect scene transitions and paragraph ordering
- Use `attention_entropy` to find paragraphs whose structural role is unusually diffuse

**Research backing.**
- **Vaswani et al. (2017)**, "Attention Is All You Need," *NeurIPS 2017*, [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) — self-attention mechanism. The narrative attention matrix is a simplified version: cosine similarity without learned Q/K/V projections.
- **van Dalen-Oskam (2023)**, *The Riddle of Literary Quality*, Amsterdam University Press, [open access](https://library.oapen.org/handle/20.500.12657/63705) — large-scale computational literary quality analysis using reader surveys (14,000 readers) and textual features. Establishes that measurable textual features correlate with perceived literary quality. The multi-feature approach of the attention lens follows this paradigm.
- Discourse coherence: **Barzilay & Lapata (2008)** and **Feng & Hirst (2014)** show local coherence can be modeled from cross-sentence structural patterns rather than only word overlap.
- Project-internal validation: Human vs LLM discrimination Mann-Whitney U = 0, p < 0.0001 on attention entropy. LLM attention entropy 7.29–7.50 vs human 6.45–7.14.

**Caveats.**
- The 12-feature vector is hand-selected after correlation pruning. Different feature sets might produce different discrimination results.
- Cosine attention is a geometric measure, not a learned mechanism. It does not learn which features matter most for structural coherence.
- Quality depends entirely on the consumed lenses — noisy inputs produce a noisy matrix.
- Feature normalization (z-score) means a single outlier paragraph can distort the entire matrix.
- The coherence metric (mean of adjacent-diagonal attention) assumes linear narrative structure — non-linear narratives may score poorly.
- All validation was performed on a single author's fiction corpus. Generalization to other styles, genres, and languages is unknown.

---

### `concreteness` (`validated`)

**What it is.** Scores prose on the concrete-abstract spectrum using Brysbaert et al. (2014) concreteness norms for known words (~40K, CC-BY) and direction-vector projection in mpnet-base-v2 embedding space for OOV words. Reports `concreteness_mean`, `abstractness_ratio`, `vague_noun_density`, and per-sentence/paragraph scores.

**Why it is useful.** Catches prose that drifts into semantic vagueness — "something shifted in the dynamic between them" — without the writer noticing. Concrete prose grounds the reader; abstract prose floats. The vague-noun detector flags words like "thing," "stuff," "aspect" that are semantically empty without qualification.

**How to use it.**
- Access through Python or `prose-doctor validate concreteness`
- Treat `vague_noun_density` as the highest-priority revision signal
- Use `concreteness_mean` comparatively across chapters — fiction typically sits 2.5–3.5
- `abstractness_ratio` flags chapters where most sentences score below 2.5

**Research backing.**
- **Brysbaert, Warriner & Kuperman (2014)**, "Concreteness ratings for 40,000 generally known English word lemmas," *Behavior Research Methods* 46, 904-911, DOI: [10.3758/s13428-013-0403-5](https://doi.org/10.3758/s13428-013-0403-5) — the norm set. 4,000+ participant ratings on a 1–5 concreteness scale. CC-BY licensed.
- **Paivio (1991)**, *Dual Coding Theory* — concrete words activate both verbal and imagistic representations, making them more memorable and vivid.
- **Snefjella & Kuperman (2016)**, "It's all in the delivery: Effects of context valence, concreteness, and affordances on visual word processing," *Cognition* — concreteness effects in processing.
- Direction-vector validation: r = 0.69 against Brysbaert norms on 5,000-word sample using all-mpnet-base-v2 embeddings.

**Caveats.**
- Brysbaert norms are lemma-level, context-free ratings. "Bank" gets one score whether it means a riverbank or a financial institution.
- The OOV embedding fallback is less reliable than norm lookup (r ≈ 0.69 vs ground truth).
- Concreteness is not a proxy for quality. Abstract prose is not inherently bad — it depends on genre and intent.
- Vague-noun detection is a word list, not a semantic analysis. "The weight of the pack" is concrete despite "weight" appearing abstract.

---

### `referential_cohesion` (`experimental`)

**What it is.** Builds an entity grid from spaCy dependency parsing — tracking each entity's grammatical role (subject, object, other, absent) across sentences — and scores transition probabilities. Also builds an entity co-occurrence graph via networkx for structural analysis.

**Why it is useful.** Catches prose where entities appear from nowhere, disappear without resolution, or churn so fast the reader loses track. The entity-grid model is a well-established measure of local coherence in computational linguistics.

**How to use it.**
- Access through Python or `prose-doctor validate referential_cohesion`
- `coherence_score` is the primary signal — higher means smoother entity transitions
- `subject_churn` flags passages where new subjects keep appearing
- `entity_continuity` per paragraph shows where referential threads break

**Research backing.**
- **Barzilay & Lapata (2008)**, "Modeling Local Coherence: An Entity-Based Approach," *Computational Linguistics* 34(1), [aclanthology.org/J08-1001](https://aclanthology.org/J08-1001/) — core paper. Shows entity grids predict text coherence with high accuracy. S→S and S→O transitions mark coherent text; —→S transitions mark incoherent text.
- **Graesser et al. (2011)**, "Coh-Metrix: Providing Multilevel Analyses of Text Characteristics," *Educational Researcher* 40(5), DOI: [10.3102/0013189X11413260](https://doi.org/10.3102/0013189X11413260) — referential cohesion as a key dimension of text readability.
- **Grosz, Joshi & Weinstein (1995)**, "Centering: A Framework for Modeling the Local Coherence of Discourse," *Computational Linguistics* 21(2) — centering theory, the theoretical basis for entity-grid models.

**Caveats.**
- Pronoun resolution uses a naive nearest-entity heuristic, not a trained coreference resolver.
- en_core_web_sm NER misses many character references, especially pronouns and epithets.
- The transition-scoring weights are heuristic, not learned from a coherence corpus.
- Entity matching is exact-string on lemmatized heads — "the old man" and "Marcus" won't match without coreference.

---

### `situation_shifts` (`experimental`)

**What it is.** Detects time, space, and actor shifts at paragraph boundaries using spaCy NER, morphological tense analysis, temporal marker word lists, motion verbs, and subject tracking. Based on the event-indexing model.

**Why it is useful.** Catches disorienting scene transitions — jumps in time without temporal grounding, location changes without spatial markers, character switches without introduction. The event-indexing model predicts that ungrounded shifts increase cognitive load and reduce comprehension.

**How to use it.**
- Access through Python or `prose-doctor validate situation_shifts`
- `total_shifts` is a scene-complexity measure, not a quality score
- `disorientation_score` is the revision signal — shifts detected only from tense change (no explicit markers) are the hardest for readers to follow
- Per-paragraph flags show exactly where each shift type occurs

**Research backing.**
- **Zwaan, Langston & Graesser (1995)**, "The Construction of Situation Models in Narrative Comprehension: An Event-Indexing Model," *Psychological Science* 6(5), DOI: [10.1111/j.1467-9280.1995.tb00513.x](https://doi.org/10.1111/j.1467-9280.1995.tb00513.x) — core paper. Readers track five situational dimensions (time, space, protagonist, causation, intentionality). Shifts on any dimension increase processing time.
- **Zwaan & Radvansky (1998)**, "Situation Models in Language Comprehension and Memory," *Psychological Bulletin* 123(2) — comprehensive review of situation model theory.
- **Rinck & Weber (2003)**, "Who When Where: An Experimental Test of the Event-Indexing Model," *Memory & Cognition* 31(8) — experimental validation showing readers monitor all five dimensions simultaneously.

**Caveats.**
- Only three of five event-indexing dimensions are tracked (time, space, actor). Causation and intentionality are not implemented.
- Temporal detection relies on word lists and tense morphology — indirect time references ("years had passed since") may be missed.
- Space shift detection depends on NER quality. en_core_web_sm misses many fictional locations.
- Actor shift detection uses grammatical subject only. Pronoun subjects are not flagged as shifts to avoid false positives.

---

### `discourse_relations` (`experimental`)

**What it is.** Classifies inter-sentence discourse relations as causal, contrastive, temporal, additive, or implicit by detecting connective words and phrases. Reports relation entropy, per-paragraph diversity, and additive-only zones.

**Why it is useful.** Catches prose that relies too heavily on additive ("and... and... and...") or implicit connections. Diverse connective usage signals stronger argumentative and narrative structure. Additive-only zones suggest list-like prose without logical progression.

**How to use it.**
- Access through Python or `prose-doctor validate discourse_relations`
- `relation_entropy` is the primary diversity measure — higher is more varied
- `additive_only_zones` flags runs of 2+ paragraphs with only additive/implicit relations
- Per-paragraph `relation_diversity` shows which paragraphs are structurally monotonous

**Research backing.**
- **Sanders & Noordman (2000)**, "The Role of Coherence Relations and Their Linguistic Markers in Text Processing," *Discourse Processes* 29(1), DOI: [10.1207/S15326950dp2901_3](https://doi.org/10.1207/S15326950dp2901_3) — core paper. Shows coherence relations affect processing speed and text recall. Causal and contrastive relations produce deeper processing than additive relations.
- **Koornneef & Sanders (2013)**, "Establishing Coherence Relations in Discourse: The Influence of Implicit Causality and Connectives on Pronoun Resolution," *Language and Cognitive Processes* 28(8), DOI: [10.1080/01690965.2012.699076](https://doi.org/10.1080/01690965.2012.699076) — connectives guide reader expectations and affect comprehension.
- **Knott & Dale (1994)**, "Using Linguistic Phenomena to Motivate a Set of Coherence Relations," *Discourse Processes* 18(1) — taxonomy of discourse relations based on linguistic markers.
- **Prasad et al. (2008)**, "The Penn Discourse TreeBank 2.0," *LREC* — large-scale annotated discourse relation corpus. The four-way classification (causal, contrastive, temporal, additive) aligns with PDTB's top-level senses.

**Caveats.**
- Connective detection is word-list based, not syntactic. "Since" is classified as causal even when temporal ("since Tuesday").
- Multi-word connectives ("on the other hand") are checked only at sentence start.
- Implicit relations (no connective) are the majority in most prose — the lens can only characterize explicit connective usage.
- "And" at sentence start in fiction is often a stylistic choice (biblical cadence, stream of consciousness), not an additive-only deficiency.

---

## Cross-Cutting References

### NLP Infrastructure
- **spaCy** — Honnibal & Montani (2017), [spacy.io](https://spacy.io). Used by: psychic_distance, info_contour, foregrounding, dialogue_voice, fragment_classifier, pacing, concreteness, referential_cohesion, situation_shifts, discourse_relations.
- **sentence-transformers** — Reimers & Gurevych (2019), "Sentence-BERT," *EMNLP*, [aclanthology.org/D19-1410](https://aclanthology.org/D19-1410/). Used by: foregrounding, dialogue_voice, sensory (separate model).
- **Hugging Face Transformers** — Wolf et al. (2020), *EMNLP Demos*, [aclanthology.org/2020.emnlp-demos.6](https://aclanthology.org/2020.emnlp-demos.6/). Used by: info_contour, perplexity, uncertainty_reduction (GPT-2), slop_classifier (ModernBERT), emotion_arc (DistilBERT).

### Literary Theory Framework
- Gardner (1984), *The Art of Fiction* — psychic distance scale
- Genette (1980), *Narrative Discourse* — focalization, narrative speed
- Shklovsky (1917), "Art as Technique" — defamiliarization
- Mukarovský (1964), "Standard Language and Poetic Language" — foregrounding theory
- Boyd (2009), *On the Origin of Stories* — evolutionary narrative theory
- Zwaan, Langston & Graesser (1995), "Event-Indexing Model" — situation model tracking
- Barzilay & Lapata (2008), "Entity-Based Coherence" — entity-grid local coherence
- Brysbaert, Warriner & Kuperman (2014), "Concreteness Ratings" — semantic concreteness norms

### Computational Literary Studies
- van Dalen-Oskam (2023), *The Riddle of Literary Quality* — multi-feature quality analysis
- Boyd, Blackburn & Pennebaker (2020), "The Narrative Arc," *Science Advances* — function-word narrative structure
- Reagan et al. (2016), "Emotional Arcs," *EPJ Data Science* — sentiment-based story shapes
- Underwood (2019), *Distant Horizons* — computational approaches to literary change
- Piper (2018), *Enumerations* — quantitative literary analysis methodology
