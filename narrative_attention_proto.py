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

    # --- NEW: Pacing mode (one-hot: action, interiority, setting) ---
    # dialogue already captured; encode the other 3 as ratios
    para_action = []
    para_interiority = []
    para_setting = []
    for para in paragraphs:
        mode = _classify_paragraph(para)
        para_action.append(1.0 if mode == "action" else 0.0)
        para_interiority.append(1.0 if mode == "interiority" else 0.0)
        para_setting.append(1.0 if mode == "setting" else 0.0)

    # --- NEW: Verb energy (dynamic vs stative verbs) ---
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

    # --- Normalize surprisal_var by paragraph length to remove length artifact ---
    para_surprisal_var_norm = []
    for sv, wc in zip(para_surprisal_var, para_word_count):
        # Normalize: surprisal variance per sentence (not per paragraph)
        para_surprisal_var_norm.append(sv)  # already per-sentence std, no length bias

    # --- Assemble structural features ---
    # Kept: orthogonal or low-correlation features
    # Dropped: setting (r=-0.70 with dialogue), interiority (zero variance),
    #          log_word_count (r=0.73 with sl_cv, length artifact)
    structural = np.column_stack([
        np.array(pd_means[:n]),              # psychic distance — orthogonal (max|r|=0.22)
        np.array(para_surprisals[:n]),       # information density
        np.array(para_fragment),             # fragment ratio
        np.array(para_inversion),            # inversion ratio
        np.array(para_sl_cv),                # sentence length variation
        np.array(para_dialogue_ratio),       # dialogue density
        # Orthogonal additions
        np.array(para_valence),              # emotion valence — most orthogonal (max|r|=0.19)
        np.array(para_action),               # pacing: action mode
        np.array(para_verb_energy),          # dynamic vs stative verbs
        np.array(para_surprisal_var_norm),   # surprisal variation within paragraph
    ])

    structural_names = [
        "pd_mean", "surprisal", "fragment_ratio", "inversion_ratio",
        "sl_cv", "dialogue_ratio",
        "emotion_valence", "pacing_action", "verb_energy", "surprisal_var",
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
            "fragment_ratios": para_fragment,
            "inversion_ratios": para_inversion,
            "sl_cvs": para_sl_cv,
            "dialogue_ratios": para_dialogue_ratio,
            "word_counts": para_word_count,
            "valence": para_valence,
            "verb_energy": para_verb_energy,
            "surprisal_var": para_surprisal_var,
            "pacing_modes": [
                "action" if a else ("interiority" if i else ("setting" if s else "dialogue"))
                for a, i, s in zip(para_action, para_interiority, para_setting)
            ],
        },
    }


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

        # Scene coherence: mean within-scene attention
        scene_attn = []
        for i in scene_paras:
            for j in scene_paras:
                if i != j:
                    scene_attn.append(attn[i, j])

        # Pacing mode distribution within scene
        scene_modes = [raw["pacing_modes"][i] for i in scene_paras if i < len(raw["pacing_modes"])]
        from collections import Counter
        mode_dist = Counter(scene_modes)
        dominant = mode_dist.most_common(1)[0][0] if mode_dist else "?"

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
            "dominant_mode": dominant,
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
                f"surp={scene['surprisal_mean']} "
                f"frag={scene['fragment_density']:.0%} "
                f"dial={scene['dialogue_ratio']:.0%} "
                f"val={scene.get('valence_mean', 0):.2f} "
                f"energy={scene.get('verb_energy', 0):.2f} "
                f"mode={scene.get('dominant_mode', '?')} "
                f"coher={scene['coherence']}"
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
