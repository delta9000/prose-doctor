# Lens Citations and Evidence

Academic citations, validation evidence, open-source references, and known limitations for each prose-doctor analytical lens.

Last updated: 2026-03-21

---

## 1. Psychic Distance

Sentence-level narrative zoom scoring based on pronoun types, perception/cognition verbs, deictic markers, sensory concreteness (via trained probe), and tense.

### Foundational Work

- Gardner, J. (1984). *The Art of Fiction: Notes on Craft for Young Writers.* Alfred A. Knopf.
  - Defines psychic distance as "the distance that the reader feels between himself and the events of the story." Proposes a 5-level continuum from establishing shot ("It was winter of the year 1853") to deep interiority ("She could not stand it").

- Genette, G. (1980). *Narrative Discourse: An Essay in Method.* Trans. Jane E. Lewin. Cornell University Press.
  - Formalizes focalization (zero, internal, external) as a narratological framework. Distinguishes "who sees" from "who speaks," which maps onto the close/distant axis this lens measures.

- Stockwell, P. (2002). *Cognitive Poetics: An Introduction.* Routledge.
  - Chapter 4 applies deictic shift theory (DST) to narrative, describing how proximal/distal deictics ("this/here" vs "that/there") track the reader's psychological position relative to the narrative. Directly informs the deictic scoring component.

- Duchan, J., Bruder, G., & Hewitt, L. (Eds.) (1995). *Deixis in Narrative: A Cognitive Science Perspective.* Lawrence Erlbaum.
  - Foundational collection on deictic shift theory. Establishes the "push/pop" model of deictic fields in narrative, where entering a character's perspective is a deictic push.

### Validation Evidence

- No published computational validation of Gardner's psychic distance scale specifically exists. The concept comes from craft pedagogy, not empirical research.
- Deictic markers as indicators of narrative perspective have been validated in cognitive poetics experiments (Stockwell 2002, Chapter 4), though not with the specific weighted formula used here.
- Perception/cognition verb density as a marker of interiority is well-established in narratology (Genette's internal focalization) and supported by corpus studies of free indirect discourse.

### Open-Source Implementations

- **spaCy** -- https://spacy.io -- POS tagging, dependency parsing, morphological features used for pronoun detection, verb tense extraction, and sentence segmentation.
- **Lancaster Sensorimotor Norms** -- https://osf.io/7emr6/ -- The sensory concreteness scoring component draws on this dataset conceptually, though the lens uses a trained neural probe rather than direct lookup.

### Baseline Data

- No published corpus-level psychic distance statistics exist for human fiction. The lens's scoring weights (0.20 pronoun, 0.25 perception, 0.10 deictic, 0.15 sensory, 0.10 tense, 0.20 interiority) are heuristic, calibrated against the developer's reading of close-third-person fiction.

### Known Limitations

- Gardner's scale is prescriptive craft advice, not empirically derived. Different literary traditions (omniscient narration, second person, experimental) may not map cleanly onto the 5-level model.
- The weighted combination assumes independence of features, but pronoun type and perception verbs are correlated (first person + cognition verbs co-occur in interior monologue).
- The zoom-jump detection assumes smooth transitions are desirable, which may not hold for modernist or fragmentary styles.
- Dialogue content is not separated from narration, so quoted speech can distort paragraph-level distance scores.

---

## 2. Information Contour

GPT-2 sentence-level surprisal scoring with FFT spectral analysis for rhythm detection, flatline detection, and spike detection.

### Foundational Work

- Hale, J. (2001). "A Probabilistic Earley Parser as a Psycholinguistic Model." *Proceedings of NAACL 2001*, pp. 1-8.
  - Introduces surprisal theory: processing difficulty at a word is proportional to its negative log probability given context. Foundational for using LM perplexity as a proxy for information density.

- Levy, R. (2008). "Expectation-based Syntactic Comprehension." *Cognition*, 106(3), pp. 1126-1177. https://doi.org/10.1016/j.cognition.2007.05.006
  - Extends surprisal theory with extensive empirical validation. Shows surprisal predicts reading times across diverse syntactic constructions.

- Levy, R. & Jaeger, T.F. (2007). "Speakers Optimize Information Density Through Syntactic Reduction." *Advances in Neural Information Processing Systems 20*.
  - Proposes the Uniform Information Density (UID) hypothesis: speakers modulate production to maintain near-constant information rate. LLM prose tends toward UID; human prose does not.

- Tsipidi, E., Nowak, F., Cotterell, R., Wilcox, E., Giulianelli, M., & Warstadt, A. (2024). "Surprise! Uniform Information Density Isn't the Whole Story: Predicting Surprisal Contours in Long-form Discourse." *Proceedings of EMNLP 2024*. https://aclanthology.org/2024.emnlp-main.1047/
  - Directly relevant. Shows information rate fluctuates in structured, predictable ways in natural discourse. Proposes the Structured Context Hypothesis: speakers modulate information rate based on hierarchical discourse structure. The position-residualized surprisal concept from this paper informs the narrative_attention lens's surprisal_residual feature.

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report*. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - GPT-2 model paper. The lens uses GPT-2 as its surprisal estimator.

### Validation Evidence

- Tsipidi et al. (2024) validate that surprisal contours in long-form text are non-uniform and structurally predictable. Their discourse-position predictors explain significant variance in surprisal beyond UID.
- The UID hypothesis (Levy & Jaeger 2007) has been validated across multiple languages and modalities, with consistent finding that speakers distribute information to avoid peaks and valleys -- but fiction writing intentionally violates this for dramatic effect.
- Flatline detection (low coefficient of variation in surprisal) as an LLM fingerprint is empirically supported by the observation that LLMs are trained to minimize perplexity, producing smoother information contours than human prose.

### Open-Source Implementations

- **Hugging Face Transformers** -- https://github.com/huggingface/transformers -- GPT-2 model loading and inference.
- **Hugging Face perplexity guide** -- https://huggingface.co/docs/transformers/perplexity -- Canonical reference for computing perplexity with fixed-length models.
- **GLTR** -- http://gltr.io/ -- Gehrmann, Strobelt, & Rush (ACL 2019). Statistical detection of generated text using GPT-2 token probabilities. Related but visual rather than spectral.

### Baseline Data

- GPT-2 perplexity on WebText (the training distribution) averages ~20-30 per token. Fiction text typically scores higher (40-80+) due to creative vocabulary.
- No published FFT spectral analysis baselines for surprisal contours in fiction exist. The rhythmicity and spectral entropy thresholds (0.6 = strongly rhythmic, 0.35 = moderate) are empirically calibrated.

### Known Limitations

- GPT-2 is a 2019 model. Its surprisal estimates reflect its training distribution (WebText), not modern prose norms. Archaic vocabulary or domain-specific terms produce artificially high surprisal.
- Sentence-level surprisal averages across tokens, losing within-sentence information structure.
- FFT assumes stationarity; surprisal contours in fiction are inherently non-stationary (climax vs exposition have different base rates).
- The flatline detection threshold (CV < 0.08) is heuristic. Short chapters may produce false flatline signals.
- Tsipidi et al. (2024) show that position within a document is a significant predictor of surprisal, which this lens does not control for.

---

## 3. Foregrounding

5-axis literary texture measurement: alliteration, syntactic inversion, sentence-length coefficient of variation, fragment ratio, and unexpected collocations (semantically distant word pairs in proximity).

### Foundational Work

- Shklovsky, V. (1917). "Art as Technique." In L. Lemon & M. Reis (Eds.), *Russian Formalist Criticism: Four Essays* (1965), pp. 3-24.
  - Introduces defamiliarization (ostranenie): art's purpose is to make the familiar strange, restoring perception. The theoretical basis for measuring textural density as a quality signal.

- Mukarovsky, J. (1964). "Standard Language and Poetic Language." In P. Garvin (Ed.), *A Prague School Reader on Esthetics, Literary Structure, and Style*, pp. 17-30.
  - Defines foregrounding as systematic deviation from the linguistic norm. Distinguishes automatization (backgrounded, habitual language) from foregrounding (language that draws attention to itself).

- van Peer, W. (1986). *Stylistics and Psychology: Investigations of Foregrounding.* Croom Helm.
  - First empirical validation. Using six poems, showed that reader ratings of "strikingness" correlate significantly with independently coded foregrounding density. Established that foregrounding effects are measurable and cross-reader consistent.

- Miall, D.S. & Kuiken, D. (1994). "Foregrounding, Defamiliarization, and Affect: Response to Literary Stories." *Poetics*, 22, pp. 389-407.
  - Four studies showing foregrounded segments produce (a) longer reading times, (b) higher strikingness ratings, (c) higher affect ratings, independent of literary training. Effect sizes moderate to large. Validates foregrounding as a measurable cognitive phenomenon.

### Validation Evidence

- van Peer (1986): Reader strikingness ratings correlated significantly with foregrounding density (p < 0.01) across readers of varying literary experience.
- Miall & Kuiken (1994): Effect of foregrounding on reading time was significant across all four studies. Foregrounding-affect correlation was independent of "literary competence" (self-reported reading experience).
- The specific 5-axis decomposition (alliteration, inversion, sentence-length variation, fragments, unexpected collocations) is novel to prose-doctor. Each axis individually has support in stylistics literature, but their weighted combination is not validated.

### Open-Source Implementations

- **spaCy** -- https://spacy.io -- Dependency parsing for inversion detection (subject after ROOT verb), POS tagging for content word filtering.
- **sentence-transformers** -- https://github.com/huggingface/sentence-transformers -- Semantic embeddings for unexpected collocation detection via cosine distance between word vectors. Uses all-MiniLM-L6-v2 (384d).
  - Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP 2019*, pp. 3982-3992. https://aclanthology.org/D19-1410/

### Baseline Data

- No published computational foregrounding baselines exist for prose fiction corpora.
- Miall & Kuiken (1994) report that ~15-25% of segments in their test stories qualified as foregrounded, but this was based on expert coding, not automated measurement.
- The composite index weights (alliteration 0.15, inversion 0.15, sentence-length CV 0.25, fragments 0.15, unexpected collocations 0.30) are heuristic.

### Known Limitations

- Alliteration detection counts consecutive content words starting with the same consonant. This misses alliterative patterns separated by function words and catches coincidental adjacency.
- Inversion detection relies on dependency parsing, which has ~95% accuracy for English. Parsed inversion may differ from perceived inversion.
- Unexpected collocations depend on sentence-transformer embedding space geometry. Words that are semantically distant in embedding space may not be perceived as "unexpected" by readers (e.g., domain-specific terms).
- The composite index assumes more foregrounding = better prose. Excessive foregrounding can be as problematic as too little (purple prose).
- Fragment ratio treats all short sentences (<5 tokens) as fragments. This catches false positives from dialogue tags and short declaratives.

---

## 4. Sensory

6-modality sensory profiling (visual, auditory, haptic, olfactory, gustatory, interoceptive) using a trained linear probe on sentence embeddings.

### Foundational Work

- Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020). "The Lancaster Sensorimotor Norms: Multidimensional Measures of Perceptual and Action Strength for 40,000 English Words." *Behavior Research Methods*, 52, pp. 1271-1291. https://doi.org/10.3758/s13428-019-01316-z
  - Provides human-rated sensory modality norms for 39,707 words across 6 perceptual modalities and 5 action effectors. Data collected from 3,500 participants on Mechanical Turk using a 5-point scale. Freely available at https://osf.io/7emr6/.
  - The conceptual basis for the 6-modality decomposition. The probe was not trained directly on these norms but on pseudo-labels generated from embedding direction vectors, informed by the same modality categories.

### Validation Evidence

- Lynott et al. (2020): Sensorimotor norms predict performance on lexical decision (faster responses for words with stronger sensorimotor associations), word naming, and property verification tasks. Outperform traditional concreteness/imageability norms on multiple benchmarks.
- The specific probe architecture (768d all-mpnet-base-v2 embeddings -> 96d hidden -> 6d sigmoid output) was trained on pseudo-labels from Qwen3-Embedding-4B direction vectors and GPT-2 cloze probabilities. This is a data-free distillation approach without direct human validation.
- No published accuracy metrics for the probe against human sensory ratings are available.

### Open-Source Implementations

- **Lancaster Sensorimotor Norms** -- https://osf.io/7emr6/ -- Raw word-level norms (CC-BY license). Could be used for direct lookup as an alternative to the neural probe.
- **sentence-transformers / all-mpnet-base-v2** -- https://huggingface.co/sentence-transformers/all-mpnet-base-v2 -- The 768d embedding model underlying the probe.

### Baseline Data

- Lynott et al. (2020): Visual modality dominates English words (mean visual rating 2.79/5 across all 40K words), followed by haptic (1.97), auditory (1.63), interoceptive (1.26), gustatory (0.81), olfactory (0.73).
- In fiction prose, visual dominance is expected. Olfactory and gustatory are reliably the weakest modalities.
- Sensory desert detection uses the 15th percentile of paragraph-level max scores as threshold, calibrated per chapter.

### Known Limitations

- The probe was trained on pseudo-labels (embedding direction vectors), not human sensory judgments. Agreement with Lancaster norms is assumed but not measured.
- Word-level scoring (content words > 3 characters) loses phrase-level sensory information. "The crack of a rifle" is auditory at the phrase level but may not score highly from individual word embeddings.
- Interoception is the least validated modality. Lynott et al. (2020) note it was a novel addition to sensorimotor norming with less theoretical grounding than the traditional 5 senses.
- The probe uses all-mpnet-base-v2 (768d), which is a different model from the shared sentence_transformer provider (all-MiniLM-L6-v2, 384d). This means the sensory lens loads its own embedding model.

---

## 5. Dialogue Voice

Character voice separation via embedding-based speaker discrimination. Extracts dialogue, attributes to speakers heuristically, embeds per-character dialogue, and measures cosine distance between speaker centroids.

### Foundational Work

- Burrows, J. (1987). *Computation into Criticism: A Study of Jane Austen's Novels.* Oxford University Press.
  - Pioneering computational stylistics work demonstrating that characters in Austen's novels have statistically distinguishable word-frequency profiles. Establishes that fictional character voice is measurable.

- Rybicki, J. & Eder, M. (2011). "Deeper Delta across Genres and Languages: Do We Really Need the Most Frequent Words?" *Literary and Linguistic Computing*, 26(3), pp. 315-321.
  - Extends Burrows's Delta method for authorship/style attribution. Shows that function-word frequencies discriminate writing styles across languages and genres.

- Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP 2019*, pp. 3982-3992. https://aclanthology.org/D19-1410/
  - The embedding method used for dialogue vectorization. SBERT produces fixed-size embeddings suitable for cosine similarity comparison.

### Validation Evidence

- Burrows (1987): Successfully distinguished all major Austen characters using the 30 most frequent words. Replicated across multiple novels.
- The embedding-centroid approach used here is simpler than Burrows's method: it embeds full dialogue lines rather than counting word frequencies. No published validation of this specific approach for fictional character discrimination exists.
- The separation threshold (< 0.15 cosine distance = "all same voice") is heuristic.

### Open-Source Implementations

- **sentence-transformers** -- https://github.com/huggingface/sentence-transformers -- Embedding computation for dialogue lines.
- **Computational Stylistics Group tools** -- https://computationalstylistics.github.io/ -- R and Python tools for stylometric analysis, including Burrows's Delta. More principled for authorship attribution but heavier-weight.

### Baseline Data

- No published embedding-based voice separation baselines for fiction dialogue exist.
- Burrows (1987) reports that major Austen characters cluster distinctly in MFW space, with typical inter-character distances of 0.3-0.5 (Burrows's Delta).
- Talking-heads threshold (8 consecutive dialogue paragraphs) is a craft heuristic, not empirically derived.

### Known Limitations

- Speaker attribution is regex-based (said/asked/replied + proper noun). This misses unattributed dialogue, non-standard dialogue tags, and multi-speaker paragraphs.
- Embedding-based separation conflates topic with voice. If two characters discuss different subjects, they'll appear distinct even with identical speaking styles.
- Cosine distance between centroids averages away within-speaker variation. A character who code-switches between registers will have a blurred centroid.
- The lens requires at least 2 speakers with 3+ attributed lines each. Chapters with mostly unattributed dialogue or a single speaking character produce no voice separation score.

---

## 6. Slop Classifier

Multi-class prose quality classification using ModernBERT fine-tuned on 13K samples from 8 LLM models, with rule-based pattern overrides.

### Foundational Work

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *Proceedings of NAACL 2019*, pp. 4171-4186. https://aclanthology.org/N19-1423/
  - The foundational encoder-only transformer architecture. ModernBERT is a 2024 modernization of BERT with rotary positional embeddings (RoPE), 8192-token context, and alternating attention patterns.

- Warner, B. et al. (2024). "ModernBERT." Answer.AI / LightOn.AI. https://huggingface.co/answerdotai/ModernBERT-base
  - The specific encoder model used. Trained on 2 trillion tokens, 2-4x faster than previous BERT variants, with 8192-token context length.

- Gehrmann, S., Strobelt, H., & Rush, A.M. (2019). "GLTR: Statistical Detection and Visualization of Generated Text." *Proceedings of ACL 2019 (System Demonstrations)*, pp. 111-116. https://aclanthology.org/P19-3019/
  - Related work on AI text detection using statistical properties of LM predictions. GLTR improved human detection rate from 54% to 72%.

### Validation Evidence

- The classifier was trained on 13K paragraphs from 8 LLM models (per git log: `ece9331`). No published validation metrics are available in external literature since this is a custom fine-tuned model.
- ModernBERT achieves SOTA on GLUE benchmarks for sequence classification, establishing the base model's competence for text classification tasks.
- The rule-based pattern system (imported from `prose_doctor.patterns.rules`) provides deterministic fallback for known slop patterns, with a fixed probability override of 0.95.

### Open-Source Implementations

- **Hugging Face Transformers** -- https://github.com/huggingface/transformers -- AutoModelForSequenceClassification loading and inference.
- **The model itself** -- https://huggingface.co/dt9k/prose-doctor-slop-classifier -- The fine-tuned checkpoint (published by the project).
- **GPTZero** -- https://gptzero.me/ -- Commercial AI text detector using perplexity + burstiness. Related but not open-source.

### Baseline Data

- Human prose false positive rate and LLM prose true positive rate are not published for this specific classifier.
- General AI text detection research shows that perplexity+burstiness methods achieve 54-72% human accuracy (Gehrmann et al. 2019). Fine-tuned classifiers typically achieve 85-95% accuracy but degrade on out-of-distribution LLM outputs.

### Known Limitations

- The classifier was trained on outputs from 8 specific LLM models. It may not generalize to new models with different output distributions.
- Context-aware scoring (prev + current + next paragraph) requires sequential paragraph ordering. Shuffled or isolated paragraphs lose context signal.
- Rule-based overrides (RULE_SLOP_PROB = 0.95) are deterministic and can produce false positives on intentional stylistic choices that happen to match slop patterns.
- "Slop" is a subjective, community-defined concept. The taxonomy (CLASS_NAMES from `patterns.taxonomy`) reflects the developer's judgment about what constitutes AI-generated prose artifacts.
- The model may flag non-native English prose or formulaic genre fiction (e.g., romance, thriller) as AI-generated.

---

## 7. Pacing

Paragraph-level scene mode classification (dialogue/action/interiority/setting) using heuristic keyword matching, with run/gap detection for talking heads and action deserts.

### Foundational Work

- Genette, G. (1980). *Narrative Discourse: An Essay in Method.* Trans. Jane E. Lewin. Cornell University Press.
  - Defines narrative speed as the relationship between discourse time and story time. Distinguishes summary, scene, pause, and ellipsis as the four narrative movements. The dialogue/action/interiority/setting decomposition in this lens is a pragmatic simplification of Genette's framework.

- Brunner, A. (2013). "Automatic Recognition of Speech, Thought, and Writing Representation in German Narrative Texts." *Literary and Linguistic Computing*, 28(4), pp. 563-575. https://doi.org/10.1093/llc/fqt024
  - Computational detection of narrative modes (direct speech, indirect speech, free indirect discourse, reported speech) in fiction. Achieved F1 = 0.87 for direct speech, 0.71 for indirect, 0.40 for free indirect. Demonstrates feasibility and difficulty of automated mode classification.

- Jockers, M. (2013). *Macroanalysis: Digital Methods and Literary History.* University of Illinois Press.
  - Establishes computational approaches to large-scale literary analysis. Relevant methodology for automated pacing analysis across corpora.

### Validation Evidence

- Brunner (2013): Validated on 57,000 tokens of manually annotated German narrative texts. Direct speech detection is reliable (F1 = 0.87); free indirect discourse is much harder (F1 = 0.40).
- The heuristic approach in this lens (keyword matching for action verbs and interiority markers) has not been validated against human annotation. It is significantly simpler than Brunner's rule-based + ML approach.

### Open-Source Implementations

- **MONAPipe** -- https://aclanthology.org/2022.konvens-1.2/ -- Modular Open-Source Narrative Analysis Pipeline. German-focused but includes speech/thought detection. More sophisticated than the keyword approach used here.
- No English-language open-source pacing classifier exists at the paragraph level.

### Baseline Data

- No published paragraph-level pacing distributions for English fiction exist.
- General observations from craft literature: well-paced commercial fiction alternates modes every 3-5 paragraphs. Literary fiction tolerates longer interiority runs.
- The talking-heads threshold (6 consecutive dialogue paragraphs) and action desert threshold (15 paragraphs without action) are craft heuristics.

### Known Limitations

- The keyword-based classifier has no learned component. Action verbs ("ran", "grabbed", "pulled") are a closed set of 32 words, missing many action verbs.
- Interiority detection requires 2+ markers per paragraph, which is too strict for subtle interiority (as noted in project memory: "zero hits" in batch testing).
- The priority order (dialogue > interiority > action > setting) means paragraphs with both dialogue and action are always classified as dialogue.
- No handling of scene breaks or chapter structure. Mode transitions across scene breaks are treated the same as within-scene transitions.

---

## 8. Emotion Arc

Sentiment arc analysis using DistilBERT (fine-tuned on SST-2) to track emotional intensity across a chapter, detecting flat vs dynamic emotional contours.

### Foundational Work

- Reagan, A.J., Mitchell, L., Kiley, D., Danforth, C.M., & Dodds, P.S. (2016). "The Emotional Arcs of Stories are Dominated by Six Basic Shapes." *EPJ Data Science*, 5(1), Article 31. https://doi.org/10.1140/epjds/s13688-016-0093-1
  - Analyzes 1,327 Project Gutenberg stories using sentiment analysis. Identifies six core emotional arc shapes (Rags to Riches, Riches to Rags, Man in a Hole, Icarus, Cinderella, Oedipus). Shows particular arc shapes correlate with download popularity.

- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C.D., Ng, A., & Potts, C. (2013). "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank." *Proceedings of EMNLP 2013*, pp. 1631-1642. https://aclanthology.org/D13-1170/
  - Introduces the Stanford Sentiment Treebank (SST) with 215,154 phrase-level sentiment annotations. The SST-2 binary split is the training data for the DistilBERT model used by this lens.

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter." *EMC^2 Workshop @ NeurIPS 2019*. https://arxiv.org/abs/1910.01108
  - The model architecture. DistilBERT retains 97% of BERT's language understanding while being 40% smaller and 60% faster. Fine-tuned on SST-2, it achieves 91.3% accuracy.

### Validation Evidence

- Reagan et al. (2016): Six arc shapes validated across 1,327 stories using matrix decomposition, supervised learning, and unsupervised clustering, with consistent results across all three methods.
- DistilBERT on SST-2: 91.3% accuracy on dev set (vs BERT's 92.7%). Well-validated for sentence-level positive/negative sentiment.
- The mapping from SST-2 sentiment (positive/negative) to "emotional intensity" is a significant conceptual leap. SST-2 measures valence, not arousal or intensity.

### Open-Source Implementations

- **distilbert-base-uncased-finetuned-sst-2-english** -- https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english -- The exact model used by this lens.
- **Hugging Face pipeline API** -- Provides the `sentiment-analysis` pipeline wrapper used for inference.
- **SyuzhetR** -- https://github.com/mjockers/syuzhet -- R package by Jockers implementing sentiment-based plot arc analysis. Uses dictionary-based methods (AFINN, NRC, Stanford CoreNLP).

### Baseline Data

- Reagan et al. (2016): Emotional arcs of successful stories tend to be more dynamic (higher variance). The most downloaded stories in Project Gutenberg exhibit "Man in a Hole" (fall-rise) and "Rags to Riches" (continuous rise) arcs.
- The flat arc threshold (std < 0.15) is heuristic. No published standard deviation baselines for chapter-level sentiment in fiction exist.

### Known Limitations

- SST-2 was trained on movie reviews, not narrative prose. Sentiment of prose sentences (e.g., "The walls were grey and the air smelled of rust") may not align with movie review sentiment.
- Binary positive/negative classification collapses emotional nuance. Fear, anger, sadness, and disgust are all mapped to "negative."
- Sampling every 3rd paragraph (with minimum 10 words) loses fine-grained arc structure and may alias high-frequency emotional oscillation.
- The intensity mapping (`score` for POSITIVE, `1 - score` for NEGATIVE) treats high-confidence negative sentiment as high intensity, which conflates valence with arousal.
- Short chapters (<15 paragraphs, resulting in <5 scored points) produce unreliable arcs.

---

## 9. Perplexity (Planned)

GPT-2 perplexity scoring for detecting AI-smooth prose. Overlaps with info_contour but focused on per-paragraph scoring rather than spectral analysis.

### Foundational Work

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). "Language Models are Unsupervised Multitask Learners." *OpenAI Technical Report*. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

- Gehrmann, S., Strobelt, H., & Rush, A.M. (2019). "GLTR: Statistical Detection and Visualization of Generated Text." *Proceedings of ACL 2019 (System Demonstrations)*, pp. 111-116. https://aclanthology.org/P19-3019/
  - Uses GPT-2 token probability distributions to detect AI-generated text. Improved human detection from 54% to 72%.

### Validation Evidence

- AI-generated text generally exhibits lower perplexity (more predictable) than human text, but this signal degrades as models improve and as prose style varies.
- Known failure mode: Wikipedia-like text (factual, encyclopedic) triggers false positives because GPT-2 was trained heavily on similar content.

### Open-Source Implementations

- **Hugging Face Transformers** -- GPT-2 model and perplexity computation.
- **GLTR** -- https://github.com/HendrikStrobelt/detecting-fake-text -- Visual GPT-2 text analysis tool.

### Baseline Data

- GPT-2 perplexity on its training distribution (WebText): ~20-30. Fiction prose: typically 40-80+.
- AI-generated text perplexity varies by model and sampling temperature but is typically lower than human prose on the same GPT-2 scorer.

### Known Limitations

- Perplexity is a proxy, not a direct quality measure. Low perplexity can indicate either AI generation or highly conventional human prose.
- "Burstiness" (variation in perplexity) is a better discriminator than mean perplexity, but has high false-positive rates on non-native English writing.
- GPT-2's training data (2019 web text) is increasingly outdated as a baseline for modern prose.

---

## 10. Uncertainty Reduction (Planned)

Entropy decrease across paragraph boundaries as a measure of suspense/engagement.

### Foundational Work

- Wilmot, D. & Keller, F. (2020). "Modelling Suspense in Short Stories as Uncertainty Reduction over Neural Representation." *Proceedings of ACL 2020*, pp. 1763-1772. https://aclanthology.org/2020.acl-main.161/
  - Core paper. Compares surprise (backward-looking) vs uncertainty reduction (forward-looking) as suspense models. Finds uncertainty reduction over neural story representations is the best predictor of human suspense judgments, achieving near-human accuracy. Uses a hierarchical language model for encoding.

### Validation Evidence

- Wilmot & Keller (2020): Evaluated against short stories annotated with human suspense judgments. Uncertainty reduction significantly outperforms surprise as a suspense predictor (near human accuracy, though specific correlation values should be checked in the paper).
- Also validated on movie synopses for predicting suspenseful events.

### Open-Source Implementations

- **Story-Untangling** -- https://github.com/dwlmt/Story-Untangling -- Wilmot's implementation of story understanding and plot analysis, including uncertainty reduction computation.

### Baseline Data

- Wilmot & Keller (2020) provide baselines on their annotated short story corpus. Specific distributional statistics for uncertainty reduction in published fiction are not available.

### Known Limitations

- The original model uses a hierarchical language model specific to the paper. Adapting uncertainty reduction to off-the-shelf LMs (as planned for this lens) requires approximation.
- Paragraph-boundary measurement loses within-paragraph suspense dynamics.
- Suspense is genre-dependent. Uncertainty reduction may not capture literary tension in non-plot-driven fiction (character studies, literary fiction).

---

## 11. Boyd Narrative Role (Planned)

Function-word profile classification into staging, progression, and tension, based on Boyd et al.'s narrative arc model.

### Foundational Work

- Boyd, R.L., Blackburn, K.G., & Pennebaker, J.W. (2020). "The Narrative Arc: Revealing Core Narrative Structures Through Text Analysis." *Science Advances*, 6(32), eaba2196. https://doi.org/10.1126/sciadv.aba2196
  - Core paper. Analyzed ~40,000 traditional narratives and ~20,000 nontraditional narratives using LIWC function-word categories. Identified three primary narrative processes:
    - **Staging**: High prepositions + articles (scene-setting language)
    - **Progression**: High auxiliary verbs + adverbs + pronouns (interactional language)
    - **Tension**: High cognitive processing words (evaluative language)
  - These three dimensions follow a consistent arc: staging high at beginning, progression peaks in middle, tension peaks at climax.

- Pennebaker, J.W., Boyd, R.L., Jordan, K., & Blackburn, K. (2015). "The Development and Psychometric Properties of LIWC2015." https://repositories.lib.utexas.edu/handle/2152/31333
  - The word-counting tool underlying Boyd et al.'s analysis. LIWC categorizes words into psychologically meaningful categories including function word classes.

- Tausczik, Y.R. & Pennebaker, J.W. (2010). "The Psychological Meaning of Words: LIWC and Computerized Text Analysis Methods." *Journal of Language and Social Psychology*, 29(1), pp. 24-54. https://doi.org/10.1177/0261927X09351676
  - Validates LIWC's psychological categories, showing function words are reliable markers of cognitive and social processes.

### Validation Evidence

- Boyd et al. (2020): The staging-progression-tension arc was consistent across novels, movie scripts, and nontraditional narratives. The three-factor structure emerged from both factor analysis and independent validation across genres.
- The lens's implementation (per narrative attention prototype notes) uses function-word ratios to classify paragraphs. ANOVA results from the prototype: staging F=4.39** (scene-differentiating), confirming it captures structural variation.

### Open-Source Implementations

- **LIWC** -- https://www.liwc.app/ -- Commercial software. Not open-source, but the word categories are published in the manual.
- **Arc of Narrative** -- https://www.arcofnarrative.com/ -- Web tool from Boyd's group implementing the narrative arc analysis.
- Function-word classification can be approximated with open POS taggers (spaCy) by counting articles, prepositions, auxiliary verbs, adverbs, and cognitive process words.

### Baseline Data

- Boyd et al. (2020): Staging words comprise ~8-12% of text at narrative beginnings, declining to ~5-7% by midpoint. Progression words peak in the middle third. Cognitive tension words increase monotonically toward climax.
- These baselines are chapter-level averages from ~40,000 narratives.

### Known Limitations

- LIWC categories are English-specific. Function-word distributions differ across languages.
- The three-factor model was derived from chapter/segment-level analysis. Paragraph-level classification (as planned) operates at a finer granularity than validated.
- Staging/progression/tension overlap. A paragraph can contain elements of all three, and the classification is based on relative proportions.
- Boyd et al. note the arc is most consistent in traditional narrative forms. Experimental or non-linear narratives may not follow the expected pattern.

---

## 12. Fragment Classifier (Planned)

Distinguishing intentional craft fragments from crutch fragments (incomplete thoughts, grammatical errors).

### Foundational Work

- No single academic paper addresses the craft-vs-crutch fragment distinction. The concept draws on:
  - Tufte, V. (2006). *Artful Sentences: Syntax as Style.* Graphics Press.
    - Comprehensive analysis of how sentence structure creates stylistic effects. Covers fragments, periodic sentences, cumulative sentences, and balance. The primary craft reference for understanding when fragments work.
  - Lanham, R. (2003). *Analyzing Prose.* 2nd edition. Continuum.
    - Framework for prose style analysis including sentence architecture classification.

### Validation Evidence

- No computational validation exists for distinguishing craft fragments from error fragments. This is an open research problem.
- The distinction is inherently subjective: what constitutes a "craft" fragment depends on context, genre, and reader expectations.

### Open-Source Implementations

- No open-source implementation exists for this specific task.
- **spaCy** dependency parsing can identify sentences lacking a finite verb (a structural fragment signal), but cannot determine intentionality.

### Baseline Data

- No published fragment frequency baselines for fiction exist.
- From the foregrounding lens: fragment ratio in prose-doctor's test corpus varies from ~5% (literary fiction) to ~15% (thriller/action).

### Known Limitations

- Binary classification (craft vs crutch) oversimplifies a spectrum. Fragments can be partially intentional, contextually appropriate but stylistically weak, or ambiguous.
- Context dependence is extreme. "Gone." as a one-word paragraph after a character's death is craft. The same word as a stray fragment mid-paragraph is likely an error.
- Training data would require expert annotation with high subjectivity. Inter-annotator agreement is expected to be low.

---

## 13. Narrative Attention (Planned)

Meta-lens building a paragraph-level feature matrix from multiple lens outputs, computing a cosine attention matrix for structural fingerprinting.

### Foundational Work

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). "Attention Is All You Need." *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. https://arxiv.org/abs/1706.03762
  - Introduces the transformer self-attention mechanism. The narrative attention matrix is a simplified version: cosine similarity between paragraph feature vectors, without learned query/key/value projections.

- van Dalen-Oskam, K. (2023). *The Riddle of Literary Quality: A Computational Approach.* Amsterdam University Press. https://library.oapen.org/handle/20.500.12657/63705
  - Large-scale computational literary quality analysis using reader surveys (14,000 readers) and textual features. Establishes that measurable textual features correlate with perceived literary quality. The multi-feature approach of the narrative attention lens follows this paradigm.

- Boyd, R.L., Blackburn, K.G., & Pennebaker, J.W. (2020). "The Narrative Arc." *Science Advances*. (See Section 11.)
  - Staging, progression, and tension features are 3 of the 12 dimensions in the attention feature vector.

- Tsipidi, E. et al. (2024). "Surprise! Uniform Information Density Isn't the Whole Story." *EMNLP 2024*. (See Section 2.)
  - Position-residualized surprisal is one of the 12 features.

- Wilmot, D. & Keller, F. (2020). "Modelling Suspense as Uncertainty Reduction." *ACL 2020*. (See Section 10.)
  - Uncertainty reduction is one of the 12 features.

### Validation Evidence

- From the narrative attention prototype (per project memory):
  - **Human vs LLM discrimination**: Mann-Whitney U=0, p<0.0001 on attention entropy. LLM attention entropy 7.29-7.50 vs human 6.45-7.14.
  - **Feature validation via ANOVA**: Dialogue ratio F=56.94***, staging F=4.39** are scene-differentiating. Surprisal residual F=0.28, valence F=0.85, uncertainty reduction F=0.04 are within-scene quality features.
  - **Valence** is the most orthogonal new feature (max |r|=0.19 with existing features).
  - Tsipidi residualization: position explains only 1-2% of surprisal variance in fiction (vs more in expository text).

### Open-Source Implementations

- No comparable open-source "narrative attention matrix" implementation exists. The concept is novel to prose-doctor.
- The constituent features each have their own implementations (see respective lens sections).

### Baseline Data

- Human prose attention entropy: 6.45-7.14 (from prototype testing on the developer's fiction).
- LLM prose attention entropy: 7.29-7.50 (from prototype testing on LLM-generated fiction).
- Block diagonal score: 0.001-0.015 for human prose (weak scene boundaries in feature space). Higher for LLM prose (uniform within-scene features).

### Known Limitations

- The 12-feature vector is hand-selected after correlation pruning. Different feature sets might produce different discrimination results.
- Cosine attention between paragraph feature vectors is a geometric measure, not a learned attention mechanism. It does not learn which features matter most for structural coherence.
- The structural outlier detector (95th percentile) was found to be useless in prototype testing (95% of paragraphs flagged). The lens needs a different anomaly detection strategy.
- Pacing classifier interiority detection produces zero hits (threshold too strict), making the interiority component of the feature vector unreliable.
- All validation was performed on a single author's fiction corpus. Generalization to other styles, genres, and languages is unknown.

---

## Cross-Cutting References

These works inform multiple lenses or the overall analytical framework.

### NLP Infrastructure

- **spaCy** -- Honnibal, M. & Montani, I. (2017). "spaCy 2: Natural Language Understanding with Bloom Embeddings, Convolutional Neural Networks, and Incremental Parsing." https://spacy.io
  - Used by: psychic_distance, info_contour, foregrounding, dialogue_voice, pacing.

- **sentence-transformers** -- Reimers, N. & Gurevych, I. (2019). "Sentence-BERT." *EMNLP 2019*. https://aclanthology.org/D19-1410/
  - Used by: foregrounding (unexpected collocations), dialogue_voice (speaker discrimination), sensory (probe embeddings).

- **Hugging Face Transformers** -- Wolf, T. et al. (2020). "Transformers: State-of-the-Art Natural Language Processing." *EMNLP 2020 (System Demonstrations)*. https://aclanthology.org/2020.emnlp-demos.6/
  - Used by: info_contour (GPT-2), slop_classifier (ModernBERT), emotion_arc (DistilBERT).

### Literary Theory Framework

- Gardner, J. (1984). *The Art of Fiction.* -- Psychic distance scale.
- Genette, G. (1980). *Narrative Discourse.* -- Focalization, narrative speed.
- Shklovsky, V. (1917). "Art as Technique." -- Defamiliarization.
- Mukarovsky, J. (1964). "Standard Language and Poetic Language." -- Foregrounding theory.

### Computational Literary Studies

- van Dalen-Oskam, K. (2023). *The Riddle of Literary Quality.* -- Multi-feature quality analysis.
- Boyd et al. (2020). "The Narrative Arc." *Science Advances.* -- Function-word narrative structure.
- Reagan et al. (2016). "Emotional Arcs." *EPJ Data Science.* -- Sentiment-based story shapes.
