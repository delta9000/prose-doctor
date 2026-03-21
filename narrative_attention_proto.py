"""Prototype: Narrative attention matrix from per-paragraph metrics.

Builds a feature vector per paragraph from existing prose-doctor analyzers,
computes cosine similarity attention, and looks for structure.

Usage:
    uv run python narrative_attention_proto.py path/to/chapter.md [chapter2.md ...]
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

from prose_doctor.text import split_paragraphs, split_paragraphs_with_breaks


def build_paragraph_features(text: str, filename: str) -> dict:
    """Extract per-paragraph feature vectors from all analyzers.

    Returns dict with:
        features: (N, D) array — one row per paragraph
        feature_names: list of feature dimension names
        paragraphs: list of paragraph texts
        scene_breaks: list of paragraph indices where scene breaks occur
        positional: (N, P) array — positional encoding features
    """
    from prose_doctor.ml import require_ml
    require_ml()
    from prose_doctor.ml.models import ModelManager
    from prose_doctor.ml.psychic_distance import analyze_chapter as pd_analyze
    from prose_doctor.ml.info_contour import analyze_chapter as ic_analyze
    from prose_doctor.ml.pacing import _classify_paragraph
    from prose_doctor.ml.emotion import EmotionArcAnalyzer

    mm = ModelManager()
    nlp = mm.spacy
    paragraphs = split_paragraphs(text)
    n = len(paragraphs)

    if n < 5:
        raise ValueError(f"Too few paragraphs ({n}) for meaningful analysis")

    # --- Per-paragraph psychic distance ---
    pd_result = pd_analyze(text, filename, mm)
    pd_means = pd_result.paragraph_means
    # Pad if mismatch
    while len(pd_means) < n:
        pd_means.append(pd_means[-1] if pd_means else 0.3)

    # --- Per-paragraph surprisal ---
    ic_result = ic_analyze(text, filename, mm)
    # Map sentence surprisals to paragraphs
    sent_surprisals = ic_result.sentence_surprisals
    para_surprisals = []
    sent_idx = 0
    for para in paragraphs:
        doc = nlp(para)
        sents = [s for s in doc.sents if len(s.text.strip().split()) >= 4]
        count = len(sents)
        if count > 0 and sent_idx + count <= len(sent_surprisals):
            para_s = np.mean(sent_surprisals[sent_idx:sent_idx + count])
            sent_idx += count
        else:
            para_s = ic_result.mean_surprisal
        para_surprisals.append(float(para_s))

    # --- Per-paragraph foregrounding features ---
    para_fragment = []
    para_inversion = []
    para_sl_cv = []
    para_word_count = []
    para_dialogue_ratio = []

    for para in paragraphs:
        words = para.split()
        para_word_count.append(len(words))

        doc = nlp(para)
        sents = list(doc.sents)

        # Fragment ratio
        if sents:
            frags = sum(1 for s in sents if len(s) < 5)
            para_fragment.append(frags / len(sents))
        else:
            para_fragment.append(0.0)

        # Inversion
        inversions = 0
        total = 0
        for sent in sents:
            total += 1
            root = None
            for t in sent:
                if t.dep_ == "ROOT":
                    root = t
                if t.dep_ in ("nsubj", "nsubjpass") and root is not None and t.i > root.i:
                    inversions += 1
                    break
        para_inversion.append(inversions / max(total, 1))

        # Sentence length CV
        lengths = [len(s) for s in sents if len(s) > 2]
        if len(lengths) >= 2:
            para_sl_cv.append(float(np.std(lengths) / np.mean(lengths)))
        else:
            para_sl_cv.append(0.0)

        # Dialogue ratio
        lines = para.split('\n')
        dial_chars = sum(
            len(l) for l in lines
            if '"' in l or '\u201c' in l or '\u201d' in l
        )
        para_dialogue_ratio.append(dial_chars / max(len(para), 1))

    # --- NEW: Emotion valence (per-paragraph) ---
    print("  Computing emotion valence...", file=sys.stderr, flush=True)
    ea = EmotionArcAnalyzer(model_manager=mm)
    ea._load()
    para_valence = []
    for para in paragraphs:
        if len(para.split()) < 5:
            para_valence.append(0.5)  # neutral for tiny paras
        else:
            para_valence.append(ea._score_intensity(para))

    # --- Boyd et al. (2020): Staging / Plot Progression / Cognitive Tension ---
    # Three continuous features from function word ratios, validated across 40K narratives.
    # These capture the NARRATIVE ROLE of a paragraph independent of surface structure.
    _STAGING_WORDS = frozenset({
        # Prepositions (scene-setting, spatial/temporal grounding)
        "in", "on", "at", "to", "from", "by", "with", "into", "through",
        "across", "between", "among", "above", "below", "beneath", "behind",
        "beside", "near", "toward", "towards", "along", "around", "against",
        "over", "under", "within", "without", "upon", "during", "before",
        "after", "until", "beyond", "past", "outside", "inside",
        # Articles (establishing reference)
        "the", "a", "an",
    })
    _PROGRESSION_WORDS = frozenset({
        # Auxiliary verbs (interactional, plot-moving)
        "would", "could", "should", "might", "must", "shall", "will",
        "can", "may", "do", "does", "did", "has", "have", "had",
        "was", "were", "been", "being", "am", "is", "are",
        # Adverbs (manner, time — action modifiers)
        "then", "now", "just", "still", "already", "again", "soon",
        "quickly", "slowly", "suddenly", "finally", "immediately",
        "carefully", "quietly", "hard", "fast", "forward", "back",
    })
    _TENSION_WORDS = frozenset({
        # Cognitive process words (reasoning, evaluation, internal conflict)
        "think", "thought", "know", "knew", "believe", "believed",
        "understand", "understood", "realize", "realized", "wonder",
        "wondered", "consider", "considered", "suppose", "supposed",
        "imagine", "imagined", "remember", "remembered", "forget", "forgot",
        "decide", "decided", "hope", "hoped", "fear", "feared",
        "wish", "wished", "doubt", "doubted", "suspect", "suspected",
        "mean", "meant", "cause", "because", "reason", "why",
        "whether", "if", "unless", "although", "though", "however",
        "but", "yet", "maybe", "perhaps", "probably", "possibly",
        "should", "shouldn't", "couldn't", "wouldn't",
        "right", "wrong", "true", "false", "certain", "uncertain",
        "sure", "unsure", "possible", "impossible",
    })

    para_staging = []
    para_progression = []
    para_tension = []
    for para in paragraphs:
        words = para.lower().split()
        wc = len(words) or 1
        word_set = set(words)
        para_staging.append(len(word_set & _STAGING_WORDS) / wc)
        para_progression.append(len(word_set & _PROGRESSION_WORDS) / wc)
        para_tension.append(len(word_set & _TENSION_WORDS) / wc)

    # --- Verb energy (dynamic vs stative verbs) ---
    _DYNAMIC_VERBS = frozenset({
        "ran", "run", "walked", "grabbed", "pulled", "pushed", "threw",
        "hit", "jumped", "climbed", "fell", "opened", "closed", "turned",
        "moved", "reached", "picked", "dropped", "kicked", "punched",
        "slammed", "ripped", "cut", "shot", "fired", "swung", "caught",
        "lunged", "stumbled", "sprinted", "crawled", "lifted", "carried",
        "shattered", "cracked", "snapped", "tore", "broke", "crashed",
        "dove", "leapt", "hurled", "seized", "yanked", "wrenched",
    })
    _STATIVE_VERBS = frozenset({
        "was", "were", "is", "seemed", "appeared", "felt", "looked",
        "remained", "stayed", "stood", "sat", "lay", "had", "knew",
        "thought", "believed", "wanted", "needed", "meant", "existed",
    })
    para_verb_energy = []
    for para in paragraphs:
        doc = nlp(para)
        verbs = [t.text.lower() for t in doc if t.pos_ == "VERB"]
        if not verbs:
            para_verb_energy.append(0.5)
            continue
        dynamic = sum(1 for v in verbs if v in _DYNAMIC_VERBS)
        stative = sum(1 for v in verbs if v in _STATIVE_VERBS)
        total_classified = dynamic + stative
        if total_classified == 0:
            para_verb_energy.append(0.5)
        else:
            para_verb_energy.append(dynamic / total_classified)

    # --- NEW: Surprisal variance within paragraph ---
    para_surprisal_var = []
    sent_idx2 = 0
    for para in paragraphs:
        doc = nlp(para)
        sents = [s for s in doc.sents if len(s.text.strip().split()) >= 4]
        count = len(sents)
        if count >= 2 and sent_idx2 + count <= len(sent_surprisals):
            chunk = sent_surprisals[sent_idx2:sent_idx2 + count]
            para_surprisal_var.append(float(np.std(chunk)))
            sent_idx2 += count
        else:
            para_surprisal_var.append(0.0)
            if count > 0:
                sent_idx2 += count

    # --- Wilmot-Keller (2020): Uncertainty reduction at paragraph boundaries ---
    # Lightweight approximation: compute next-token entropy at the last token of
    # each paragraph, then measure how much it changes across boundaries.
    # High reduction = this paragraph resolves uncertainty (tension point).
    # Low reduction = filler that doesn't change what comes next.
    print("  Computing uncertainty reduction at paragraph boundaries...", file=sys.stderr, flush=True)
    import torch
    model_gpt2, tokenizer_gpt2 = mm.gpt2
    device = mm.device

    def _next_token_entropy(text_context: str, max_tokens: int = 256) -> float:
        """Compute entropy of next-token distribution given context."""
        inputs = tokenizer_gpt2(
            text_context, return_tensors="pt",
            truncation=True, max_length=max_tokens,
        )
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model_gpt2(**inputs)
            logits = outputs.logits[0, -1, :]  # last token logits
            probs = torch.softmax(logits, dim=-1)
            # Top-k entropy (full vocab entropy is dominated by noise)
            top_probs, _ = probs.topk(100)
            top_probs = top_probs / top_probs.sum()  # renormalize
            entropy = -float((top_probs * top_probs.log()).sum())
        return entropy

    para_uncertainty_reduction = []
    prev_entropy = None
    for pi, para in enumerate(paragraphs):
        # Context: last ~200 tokens of preceding text
        preceding = " ".join(paragraphs[max(0, pi-2):pi+1])
        curr_entropy = _next_token_entropy(preceding)
        if prev_entropy is not None:
            # Reduction = how much entropy decreased (positive = resolved uncertainty)
            para_uncertainty_reduction.append(prev_entropy - curr_entropy)
        else:
            para_uncertainty_reduction.append(0.0)
        prev_entropy = curr_entropy

    # --- Semantic embedding (sentence-transformer, low-dim projection) ---
    st = mm.sentence_transformer
    para_embeddings = st.encode(paragraphs, show_progress_bar=False)
    # PCA to 8 dims to keep it comparable scale to other features
    from sklearn.decomposition import PCA
    n_semantic = min(8, n - 1, para_embeddings.shape[1])
    if n_semantic >= 2:
        pca = PCA(n_components=n_semantic)
        para_semantic = pca.fit_transform(para_embeddings)
        semantic_variance = pca.explained_variance_ratio_.sum()
        print(f"  Semantic PCA: {n_semantic} dims, {semantic_variance:.1%} variance", file=sys.stderr)
    else:
        para_semantic = para_embeddings[:, :2]

    # --- Detect scene breaks ---
    items = split_paragraphs_with_breaks(text)
    scene_breaks = []
    para_counter = 0
    for item in items:
        if item is None:
            scene_breaks.append(para_counter)
        else:
            para_counter += 1

    # --- Tsipidi et al. (2024): Position-residualized surprisal ---
    # Surprisal is partially predictable from position in the discourse hierarchy.
    # The residual (observed - predicted) captures genuine surprise independent of
    # structural expectations. Regress surprisal on positional features, keep residual.

    # First compute positional features for regression
    # (scene breaks already detected above)
    pos_for_regression = np.zeros((n, 3))
    for i in range(n):
        pos_for_regression[i, 0] = i / max(n - 1, 1)  # position in chapter
        scene_idx = sum(1 for b in scene_breaks if b <= i)
        scene_start = max([b for b in scene_breaks if b <= i], default=0)
        scene_end = min([b for b in scene_breaks if b > i], default=n)
        scene_len = scene_end - scene_start
        pos_for_regression[i, 1] = (i - scene_start) / max(scene_len - 1, 1)  # position in scene
        pos_for_regression[i, 2] = scene_idx  # nesting/scene index

    raw_surprisals = np.array(para_surprisals[:n])
    # Linear regression: surprisal ~ position features
    from numpy.linalg import lstsq
    X_reg = np.column_stack([pos_for_regression, np.ones(n)])
    coeffs, _, _, _ = lstsq(X_reg, raw_surprisals, rcond=None)
    predicted_surprisal = X_reg @ coeffs
    surprisal_residual = raw_surprisals - predicted_surprisal

    r_squared = 1 - np.var(surprisal_residual) / max(np.var(raw_surprisals), 1e-10)
    print(f"  Tsipidi residualization: R²={r_squared:.3f} "
          f"(position explains {r_squared:.0%} of surprisal variance)", file=sys.stderr)

    # --- Assemble structural features ---
    # Research-backed feature set:
    # - Boyd (2020): staging, progression, tension (narrative role)
    # - Tsipidi (2024): position-residualized surprisal (genuine surprise)
    # - Wilmot-Keller (2020): uncertainty reduction (narrative consequence)
    # - Original: psychic distance, fragments, inversion, sl_cv, dialogue, valence, verb energy
    structural = np.column_stack([
        np.array(pd_means[:n]),                  # psychic distance
        np.array(surprisal_residual),            # Tsipidi: position-residualized surprisal
        np.array(para_fragment),                 # fragment ratio
        np.array(para_inversion),                # inversion ratio
        np.array(para_sl_cv),                    # sentence length variation
        np.array(para_dialogue_ratio),           # dialogue density
        np.array(para_valence),                  # emotion valence
        np.array(para_verb_energy),              # dynamic vs stative verbs
        # Boyd (2020)
        np.array(para_staging),                  # staging (prepositions + articles)
        np.array(para_progression),              # plot progression (auxiliaries + adverbs)
        np.array(para_tension),                  # cognitive tension (reasoning words)
        # Wilmot-Keller (2020)
        np.array(para_uncertainty_reduction),    # uncertainty reduction at boundaries
    ])

    structural_names = [
        "pd_mean", "surprisal_residual", "fragment_ratio", "inversion_ratio",
        "sl_cv", "dialogue_ratio", "emotion_valence", "verb_energy",
        "boyd_staging", "boyd_progression", "boyd_tension",
        "uncertainty_reduction",
    ]

    # --- Positional encoding ---
    positions = np.zeros((n, 5))
    for i in range(n):
        positions[i, 0] = i / max(n - 1, 1)  # relative position in chapter
        # Scene index
        scene_idx = sum(1 for b in scene_breaks if b <= i)
        positions[i, 1] = scene_idx
        # Position within scene
        scene_start = max([b for b in scene_breaks if b <= i], default=0)
        scene_end = min([b for b in scene_breaks if b > i], default=n)
        scene_len = scene_end - scene_start
        positions[i, 2] = (i - scene_start) / max(scene_len - 1, 1)
        # Distance to chapter end (0 = at end)
        positions[i, 3] = 1.0 - (i / max(n - 1, 1))
        # Distance to nearest scene break
        if scene_breaks:
            positions[i, 4] = min(abs(i - b) for b in scene_breaks) / max(n, 1)
        else:
            positions[i, 4] = 0.5

    pos_names = [
        "rel_position", "scene_idx", "pos_in_scene",
        "dist_to_end", "dist_to_break",
    ]

    # --- Normalize each feature to [0, 1] ---
    def normalize(arr):
        mins = arr.min(axis=0, keepdims=True)
        maxs = arr.max(axis=0, keepdims=True)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return (arr - mins) / ranges

    structural_norm = normalize(structural)
    semantic_norm = normalize(para_semantic)
    positions_norm = normalize(positions)

    # Combine: structural (weight 1.0) + semantic (weight 0.5) + positional (weight 0.3)
    # Weighting reflects: structure matters most, semantics adds context, position is auxiliary
    features = np.hstack([
        structural_norm * 1.0,
        semantic_norm * 0.5,
        positions_norm * 0.3,
    ])

    all_names = structural_names + [f"sem_{i}" for i in range(n_semantic)] + pos_names

    return {
        "features": features,
        "structural": structural_norm,
        "semantic": semantic_norm,
        "positional": positions_norm,
        "feature_names": all_names,
        "paragraphs": paragraphs,
        "scene_breaks": scene_breaks,
        "raw": {
            "pd_means": pd_means[:n],
            "surprisals": para_surprisals,
            "surprisal_residual": surprisal_residual.tolist(),
            "fragment_ratios": para_fragment,
            "inversion_ratios": para_inversion,
            "sl_cvs": para_sl_cv,
            "dialogue_ratios": para_dialogue_ratio,
            "word_counts": para_word_count,
            "valence": para_valence,
            "verb_energy": para_verb_energy,
            "staging": para_staging,
            "progression": para_progression,
            "tension": para_tension,
            "uncertainty_reduction": para_uncertainty_reduction,
        },
    }


def decompose_generics(
    structural: np.ndarray,
    feature_names: list[str],
    paragraphs: list[str],
    top_n: int = 10,
) -> list[dict]:
    """For each generic paragraph, explain WHY it's generic.

    A paragraph is generic when its features are close to the chapter mean
    across all dimensions. This function identifies which specific features
    are most "average" — those are what the revision agent should change.

    Returns a list of dicts, each with:
        paragraph_idx, text, generic_score (lower = more generic),
        most_average_features (the features closest to chapter mean),
        most_distinctive_features (what's already unique — preserve these)
    """
    n, d = structural.shape

    # Normalize to [0,1] per feature
    mins = structural.min(axis=0, keepdims=True)
    maxs = structural.max(axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (structural - mins) / ranges

    # Chapter mean per feature
    chapter_mean = normed.mean(axis=0)

    # Per-paragraph: distance from mean on each feature
    deviations = np.abs(normed - chapter_mean)  # (n, d)

    # Generic score: mean deviation across all features (low = generic)
    generic_scores = deviations.mean(axis=1)

    # Rank paragraphs by genericness
    ranking = np.argsort(generic_scores)

    results = []
    for idx in ranking[:top_n]:
        devs = deviations[idx]
        # Sort features by deviation (ascending = most average first)
        feat_order = np.argsort(devs)

        most_average = []
        for fi in feat_order[:4]:
            most_average.append({
                "feature": feature_names[fi],
                "value": round(float(normed[idx, fi]), 3),
                "chapter_mean": round(float(chapter_mean[fi]), 3),
                "deviation": round(float(devs[fi]), 4),
            })

        most_distinctive = []
        for fi in feat_order[-3:]:
            most_distinctive.append({
                "feature": feature_names[fi],
                "value": round(float(normed[idx, fi]), 3),
                "chapter_mean": round(float(chapter_mean[fi]), 3),
                "deviation": round(float(devs[fi]), 4),
            })

        results.append({
            "paragraph_idx": int(idx),
            "text": paragraphs[idx][:150],
            "generic_score": round(float(generic_scores[idx]), 4),
            "most_average_features": most_average,
            "most_distinctive_features": most_distinctive,
        })

    return results


def format_generic_decomposition(results: list[dict]) -> str:
    """Format generic decomposition for the revision agent."""
    lines = [f"## Generic Paragraphs — Feature Decomposition ({len(results)} most generic)\n"]
    lines.append("These paragraphs are structurally indistinguishable from the chapter average.")
    lines.append("Fix the MOST AVERAGE features. Preserve the DISTINCTIVE ones.\n")

    for r in results:
        lines.append(f"**[paragraph {r['paragraph_idx']}]** generic_score={r['generic_score']}")
        lines.append(f"  > {r['text']}")

        lines.append(f"  CHANGE THESE (most average):")
        for f in r["most_average_features"]:
            lines.append(f"    - {f['feature']}: {f['value']:.3f} (chapter mean: {f['chapter_mean']:.3f}, dev: {f['deviation']:.4f})")

        lines.append(f"  KEEP THESE (already distinctive):")
        for f in r["most_distinctive_features"]:
            lines.append(f"    + {f['feature']}: {f['value']:.3f} (chapter mean: {f['chapter_mean']:.3f}, dev: {f['deviation']:.4f})")

        lines.append("")

    return "\n".join(lines)


def compute_attention(features: np.ndarray) -> np.ndarray:
    """Compute cosine similarity attention matrix."""
    # Normalize rows to unit vectors
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = features / norms
    # Cosine similarity
    attn = normed @ normed.T
    return attn


def analyze_attention(
    attn: np.ndarray,
    paragraphs: list[str],
    scene_breaks: list[int],
    raw: dict,
) -> dict:
    """Extract insights from the attention matrix.

    Returns dict with:
        block_diagonal_score: how much attention clusters within scenes
        orphans: paragraphs that don't attend strongly to anything
        generics: paragraphs that attend equally to everything
        arc_profile: scene-level attention summary
        structural_outliers: paragraphs that break local pattern
    """
    n = attn.shape[0]

    # --- Block diagonal score ---
    # Compare within-scene attention to between-scene attention
    scenes = []
    prev = 0
    for b in sorted(scene_breaks):
        if b > prev:
            scenes.append((prev, b))
        prev = b
    if prev < n:
        scenes.append((prev, n))

    within_attn = []
    between_attn = []
    for start, end in scenes:
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    within_attn.append(attn[i, j])
            for j in range(n):
                if j < start or j >= end:
                    between_attn.append(attn[i, j])

    if within_attn and between_attn:
        block_score = float(np.mean(within_attn) - np.mean(between_attn))
    else:
        block_score = 0.0

    # --- Orphans: low max attention (excluding self) ---
    attn_no_self = attn.copy()
    np.fill_diagonal(attn_no_self, 0)
    max_attn = attn_no_self.max(axis=1)
    mean_max = float(np.mean(max_attn))
    std_max = float(np.std(max_attn))
    orphan_threshold = mean_max - 1.5 * std_max
    orphans = []
    for i in range(n):
        if max_attn[i] < orphan_threshold:
            orphans.append({
                "paragraph_idx": i,
                "max_attention": round(float(max_attn[i]), 3),
                "text": paragraphs[i][:100],
                "reason": "structurally orphaned — doesn't resemble any other paragraph's narrative mode",
            })

    # --- Generics: low attention variance (attends equally to everything) ---
    attn_std = attn_no_self.std(axis=1)
    mean_std = float(np.mean(attn_std))
    std_std = float(np.std(attn_std))
    generic_threshold = mean_std - 1.5 * std_std
    generics = []
    for i in range(n):
        if attn_std[i] < generic_threshold:
            generics.append({
                "paragraph_idx": i,
                "attention_std": round(float(attn_std[i]), 4),
                "text": paragraphs[i][:100],
                "reason": "structurally generic — attends equally to everything, no distinct narrative mode",
            })

    # --- Structural outliers: paragraphs that break local pattern ---
    # Compare each paragraph's features to its immediate neighbors
    outliers = []
    for i in range(1, n - 1):
        local_mean = (attn[i, i-1] + attn[i, i+1]) / 2
        if local_mean < mean_max - 2 * std_max:
            outliers.append({
                "paragraph_idx": i,
                "local_attention": round(float(local_mean), 3),
                "text": paragraphs[i][:100],
                "reason": "breaks local pattern — structurally different from neighbors",
            })

    # --- Arc profile: per-scene summary ---
    arc = []
    for si, (start, end) in enumerate(scenes):
        scene_paras = list(range(start, end))
        if not scene_paras:
            continue

        scene_pd = [raw["pd_means"][i] for i in scene_paras if i < len(raw["pd_means"])]
        scene_surp = [raw["surprisals"][i] for i in scene_paras if i < len(raw["surprisals"])]
        scene_frag = [raw["fragment_ratios"][i] for i in scene_paras if i < len(raw["fragment_ratios"])]
        scene_dial = [raw["dialogue_ratios"][i] for i in scene_paras if i < len(raw["dialogue_ratios"])]
        scene_val = [raw["valence"][i] for i in scene_paras if i < len(raw["valence"])]
        scene_energy = [raw["verb_energy"][i] for i in scene_paras if i < len(raw["verb_energy"])]
        scene_staging = [raw["staging"][i] for i in scene_paras if i < len(raw["staging"])]
        scene_progression = [raw["progression"][i] for i in scene_paras if i < len(raw["progression"])]
        scene_tension_vals = [raw["tension"][i] for i in scene_paras if i < len(raw["tension"])]
        scene_unc = [raw["uncertainty_reduction"][i] for i in scene_paras if i < len(raw["uncertainty_reduction"])]

        # Scene coherence: mean within-scene attention
        scene_attn = []
        for i in scene_paras:
            for j in scene_paras:
                if i != j:
                    scene_attn.append(attn[i, j])

        # Boyd dominant mode
        s = float(np.mean(scene_staging)) if scene_staging else 0
        p = float(np.mean(scene_progression)) if scene_progression else 0
        t = float(np.mean(scene_tension_vals)) if scene_tension_vals else 0
        boyd_mode = max({"staging": s, "progression": p, "tension": t}, key=lambda k: {"staging": s, "progression": p, "tension": t}[k])

        arc.append({
            "scene": si,
            "paragraphs": f"{start}-{end-1}",
            "length": end - start,
            "pd_mean": round(float(np.mean(scene_pd)), 3) if scene_pd else 0,
            "pd_trend": round(float(np.polyfit(range(len(scene_pd)), scene_pd, 1)[0]), 4) if len(scene_pd) >= 3 else 0,
            "surprisal_mean": round(float(np.mean(scene_surp)), 3) if scene_surp else 0,
            "fragment_density": round(float(np.mean(scene_frag)), 3) if scene_frag else 0,
            "dialogue_ratio": round(float(np.mean(scene_dial)), 3) if scene_dial else 0,
            "valence_mean": round(float(np.mean(scene_val)), 3) if scene_val else 0,
            "verb_energy": round(float(np.mean(scene_energy)), 3) if scene_energy else 0,
            "boyd_staging": round(s, 3),
            "boyd_progression": round(p, 3),
            "boyd_tension": round(t, 3),
            "boyd_mode": boyd_mode,
            "uncertainty_reduction": round(float(np.mean(scene_unc)), 4) if scene_unc else 0,
            "coherence": round(float(np.mean(scene_attn)), 3) if scene_attn else 0,
        })

    return {
        "block_diagonal_score": round(block_score, 4),
        "n_paragraphs": n,
        "n_scenes": len(scenes),
        "orphans": orphans,
        "generics": generics,
        "structural_outliers": outliers,
        "arc_profile": arc,
    }


def render_attention_ascii(attn: np.ndarray, scene_breaks: list[int], width: int = 80) -> str:
    """Render attention matrix as ASCII heatmap."""
    n = attn.shape[0]
    # Downsample if too large
    step = max(1, n // width)
    sampled = attn[::step, ::step]
    m = sampled.shape[0]

    chars = " ░▒▓█"
    lines = []
    lines.append(f"  Attention matrix ({n} paragraphs, step={step})")
    lines.append(f"  Scene breaks at: {scene_breaks}")
    lines.append("")

    for i in range(m):
        row = ""
        for j in range(m):
            v = sampled[i, j]
            idx = min(int(v * len(chars)), len(chars) - 1)
            row += chars[idx]
        # Mark scene breaks
        marker = "│" if (i * step) in scene_breaks else " "
        lines.append(f"  {marker}{row}{marker}")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python narrative_attention_proto.py chapter.md [...]")
        sys.exit(1)

    files = [Path(f) for f in sys.argv[1:] if Path(f).is_file()]
    if not files:
        print("No valid files found.")
        sys.exit(1)

    for f in files:
        print(f"\n{'=' * 70}")
        print(f"  {f.name}")
        print(f"{'=' * 70}\n")

        text = f.read_text()
        data = build_paragraph_features(text, f.name)

        # Compute attention from structural features only (no semantic)
        attn_structural = compute_attention(data["structural"])
        # Full attention (structural + semantic + positional)
        attn_full = compute_attention(data["features"])

        # Analyze both
        print("--- STRUCTURAL ONLY ---")
        analysis_s = analyze_attention(
            attn_structural, data["paragraphs"],
            data["scene_breaks"], data["raw"],
        )
        print(f"  Block diagonal score: {analysis_s['block_diagonal_score']}")
        print(f"  (>0 = scenes cluster internally, 0 = uniform, <0 = scenes anti-cluster)")
        print(f"  Scenes: {analysis_s['n_scenes']}, Paragraphs: {analysis_s['n_paragraphs']}")

        print(f"\n  Orphans ({len(analysis_s['orphans'])}):")
        for o in analysis_s["orphans"][:5]:
            print(f"    [{o['paragraph_idx']}] max_attn={o['max_attention']}: {o['text'][:80]}")

        print(f"\n  Generics ({len(analysis_s['generics'])}):")
        for g in analysis_s["generics"][:5]:
            print(f"    [{g['paragraph_idx']}] attn_std={g['attention_std']}: {g['text'][:80]}")

        print(f"\n  Structural outliers ({len(analysis_s['structural_outliers'])}):")
        for o in analysis_s["structural_outliers"][:5]:
            print(f"    [{o['paragraph_idx']}] local_attn={o['local_attention']}: {o['text'][:80]}")

        print("\n  Arc profile:")
        for scene in analysis_s["arc_profile"]:
            pd_arrow = "↗" if scene["pd_trend"] > 0.005 else ("↘" if scene["pd_trend"] < -0.005 else "→")
            print(
                f"    Scene {scene['scene']} [{scene['paragraphs']}] "
                f"({scene['length']} paras) | "
                f"pd={scene['pd_mean']}{pd_arrow} "
                f"val={scene.get('valence_mean', 0):.2f} "
                f"energy={scene.get('verb_energy', 0):.2f} "
                f"unc={scene.get('uncertainty_reduction', 0):+.3f} "
                f"coher={scene['coherence']}"
            )
            print(
                f"           Boyd: "
                f"staging={scene.get('boyd_staging', 0):.3f} "
                f"progress={scene.get('boyd_progression', 0):.3f} "
                f"tension={scene.get('boyd_tension', 0):.3f} "
                f"[{scene.get('boyd_mode', '?')}]"
            )

        print("\n--- WITH SEMANTIC + POSITIONAL ---")
        analysis_f = analyze_attention(
            attn_full, data["paragraphs"],
            data["scene_breaks"], data["raw"],
        )
        print(f"  Block diagonal score: {analysis_f['block_diagonal_score']}")
        print(f"  (compare to structural-only: {analysis_s['block_diagonal_score']})")

        if analysis_f["orphans"] != analysis_s["orphans"]:
            new_orphans = [o for o in analysis_f["orphans"]
                          if o["paragraph_idx"] not in {x["paragraph_idx"] for x in analysis_s["orphans"]}]
            resolved = [o for o in analysis_s["orphans"]
                       if o["paragraph_idx"] not in {x["paragraph_idx"] for x in analysis_f["orphans"]}]
            if new_orphans:
                print(f"\n  New orphans with semantics: {[o['paragraph_idx'] for o in new_orphans]}")
            if resolved:
                print(f"  Resolved orphans with semantics: {[o['paragraph_idx'] for o in resolved]}")

        # ASCII heatmap (structural only — cleaner signal)
        print(f"\n{render_attention_ascii(attn_structural, data['scene_breaks'])}")

        # Save full data for further analysis
        out = f.with_suffix(".attention.npz")
        np.savez(
            out,
            attn_structural=attn_structural,
            attn_full=attn_full,
            features=data["features"],
            structural=data["structural"],
            semantic=data["semantic"],
            positional=data["positional"],
        )
        print(f"\n  Saved matrices to {out}")


if __name__ == "__main__":
    main()
