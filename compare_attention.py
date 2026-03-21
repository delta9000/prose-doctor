"""Compare narrative attention matrices between human and LLM prose.

Tests the hypothesis that LLM prose produces flatter, more uniform attention
(less structural variety, more template-like paragraphs).

Usage:
    uv run python compare_attention.py human.md llm.md [llm2.md ...]
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, ks_2samp

from narrative_attention_proto import (
    build_paragraph_features,
    compute_attention,
    analyze_attention,
)


def attention_fingerprint(attn: np.ndarray, raw: dict, scene_breaks: list[int]) -> dict:
    """Compute a fingerprint of the attention matrix's structure."""
    n = attn.shape[0]
    np.fill_diagonal(attn, 0)

    upper = attn[np.triu_indices(n, k=1)]

    # Row-level stats
    row_max = attn.max(axis=1)
    row_std = attn.std(axis=1)
    row_mean = attn.mean(axis=1)

    # How many strong connections does each paragraph have?
    for threshold in [0.85, 0.90, 0.95]:
        strong = (attn > threshold).sum(axis=1)

    # Attention entropy per row (how spread is each paragraph's attention?)
    row_entropy = []
    for i in range(n):
        row = attn[i, :]
        row_pos = row[row > 0]
        if len(row_pos) > 1:
            p = row_pos / row_pos.sum()
            entropy = -float((p * np.log2(p)).sum())
            row_entropy.append(entropy)
        else:
            row_entropy.append(0.0)
    row_entropy = np.array(row_entropy)

    # Block diagonal score
    scenes = []
    prev = 0
    for b in sorted(scene_breaks):
        if b > prev:
            scenes.append((prev, b))
        prev = b
    if prev < n:
        scenes.append((prev, n))

    within, between = [], []
    for s, e in scenes:
        for i in range(s, e):
            for j in range(s, e):
                if i != j:
                    within.append(attn[i, j])
            for j in range(n):
                if j < s or j >= e:
                    between.append(attn[i, j])

    block_diag = float(np.mean(within) - np.mean(between)) if within and between else 0

    # Attention "temperature" — how peaked vs uniform is the attention?
    # Low temperature = sharp, focused attention (attending to specific paragraphs)
    # High temperature = diffuse, uniform attention (attending equally to everything)
    temperature = float(np.mean(row_entropy))

    # Structural variety — std of per-paragraph feature means
    # High = paragraphs differ a lot from each other
    # Low = all paragraphs look similar
    feature_stds = []
    for key in ["pd_means", "surprisals", "fragment_ratios", "valence",
                "verb_energy", "staging", "tension", "uncertainty_reduction"]:
        if key in raw and raw[key]:
            vals = np.array(raw[key])
            if len(vals) > 1:
                feature_stds.append(float(np.std(vals)))

    structural_variety = float(np.mean(feature_stds)) if feature_stds else 0

    # Orphan count (structurally unique paragraphs)
    orphan_threshold = float(np.mean(row_max) - 1.5 * np.std(row_max))
    n_orphans = int(np.sum(row_max < orphan_threshold))

    # Generic count (template paragraphs)
    generic_threshold = float(np.mean(row_std) - 1.5 * np.std(row_std))
    n_generics = int(np.sum(row_std < generic_threshold))

    return {
        "n_paragraphs": n,
        "n_scenes": len(scenes),
        "block_diagonal": round(block_diag, 4),
        "temperature": round(temperature, 3),
        "structural_variety": round(structural_variety, 4),
        "n_orphans": n_orphans,
        "n_generics": n_generics,
        "attn_mean": round(float(np.mean(upper)), 3),
        "attn_std": round(float(np.std(upper)), 3),
        "row_max_mean": round(float(np.mean(row_max)), 3),
        "row_max_std": round(float(np.std(row_max)), 3),
        "row_entropy_mean": round(float(np.mean(row_entropy)), 3),
        "row_entropy_std": round(float(np.std(row_entropy)), 3),
        # Feature-level variety
        "feature_stds": {k: round(float(np.std(raw[k])), 4)
                         for k in ["pd_means", "surprisals", "fragment_ratios",
                                   "valence", "verb_energy", "staging",
                                   "tension", "uncertainty_reduction"]
                         if k in raw and raw[k]},
    }


def compare(human_fp: dict, llm_fp: dict) -> None:
    """Print comparison between human and LLM attention fingerprints."""
    print("\n" + "=" * 70)
    print("  HUMAN vs LLM ATTENTION FINGERPRINT")
    print("=" * 70)

    metrics = [
        ("Paragraphs", "n_paragraphs", None),
        ("Scenes", "n_scenes", None),
        ("Block diagonal", "block_diagonal", "higher = scenes more distinct"),
        ("Temperature", "temperature", "higher = more uniform attention"),
        ("Structural variety", "structural_variety", "higher = more diverse paragraphs"),
        ("Orphans", "n_orphans", "structurally unique paragraphs"),
        ("Generics", "n_generics", "template-like paragraphs"),
        ("Attention mean", "attn_mean", None),
        ("Attention std", "attn_std", "higher = more selective attention"),
        ("Row max mean", "row_max_mean", "higher = stronger best-match"),
        ("Row max std", "row_max_std", "higher = more varied best-matches"),
        ("Row entropy mean", "row_entropy_mean", "higher = more diffuse attention"),
        ("Row entropy std", "row_entropy_std", "higher = more varied attention patterns"),
    ]

    print(f"\n  {'Metric':<25} {'Human':>10} {'LLM':>10} {'Delta':>10}  Direction")
    print(f"  {'-' * 75}")

    for label, key, note in metrics:
        h = human_fp[key]
        l = llm_fp[key]
        if isinstance(h, (int, float)) and isinstance(l, (int, float)):
            delta = l - h
            arrow = "→" if abs(delta) < 0.001 else ("↑" if delta > 0 else "↓")
            note_str = f"  ({note})" if note else ""
            print(f"  {label:<25} {h:>10} {l:>10} {delta:>+10.4f}  {arrow}{note_str}")

    # Per-feature variety comparison
    print(f"\n  Feature-level variety (std per feature):")
    print(f"  {'Feature':<25} {'Human':>10} {'LLM':>10} {'Delta':>10}")
    print(f"  {'-' * 55}")
    h_stds = human_fp.get("feature_stds", {})
    l_stds = llm_fp.get("feature_stds", {})
    for key in h_stds:
        if key in l_stds:
            h = h_stds[key]
            l = l_stds[key]
            delta = l - h
            arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "→")
            print(f"  {key:<25} {h:>10.4f} {l:>10.4f} {delta:>+10.4f} {arrow}")


def main():
    if len(sys.argv) < 3:
        print("Usage: uv run python compare_attention.py human.md llm.md [llm2.md ...]")
        sys.exit(1)

    human_path = Path(sys.argv[1])
    llm_paths = [Path(f) for f in sys.argv[2:]]

    print(f"\n  Human: {human_path.name}")
    for p in llm_paths:
        print(f"  LLM:   {p.name}")

    # Build human features
    print(f"\n  Processing human chapter...", file=sys.stderr)
    human_text = human_path.read_text()
    human_data = build_paragraph_features(human_text, human_path.name)
    human_attn = compute_attention(human_data["structural"])
    human_fp = attention_fingerprint(human_attn, human_data["raw"], human_data["scene_breaks"])

    for llm_path in llm_paths:
        print(f"\n  Processing LLM chapter: {llm_path.name}...", file=sys.stderr)
        llm_text = llm_path.read_text()
        llm_data = build_paragraph_features(llm_text, llm_path.name)
        llm_attn = compute_attention(llm_data["structural"])
        llm_fp = attention_fingerprint(llm_attn, llm_data["raw"], llm_data["scene_breaks"])

        compare(human_fp, llm_fp)

        # Statistical tests on the attention distributions
        human_upper = human_attn[np.triu_indices(human_attn.shape[0], k=1)]
        llm_upper = llm_attn[np.triu_indices(llm_attn.shape[0], k=1)]

        # KS test: are the attention distributions different?
        ks_stat, ks_p = ks_2samp(human_upper, llm_upper)
        print(f"\n  KS test (attention distributions): D={ks_stat:.4f}, p={ks_p:.2e}")
        if ks_p < 0.001:
            print(f"  → Distributions are significantly different (p<0.001)")
        elif ks_p < 0.05:
            print(f"  → Distributions are somewhat different (p<0.05)")
        else:
            print(f"  → Distributions are not significantly different")

        # Compare row entropy distributions (is LLM attention more uniform?)
        human_entropy = []
        llm_entropy = []
        for attn_mat, entropy_list in [(human_attn, human_entropy), (llm_attn, llm_entropy)]:
            n = attn_mat.shape[0]
            attn_copy = attn_mat.copy()
            np.fill_diagonal(attn_copy, 0)
            for i in range(n):
                row = attn_copy[i, :]
                row_pos = row[row > 0]
                if len(row_pos) > 1:
                    p = row_pos / row_pos.sum()
                    entropy_list.append(-float((p * np.log2(p)).sum()))

        u_stat, u_p = mannwhitneyu(human_entropy, llm_entropy, alternative='less')
        print(f"\n  Mann-Whitney U (is LLM entropy higher = more uniform?):")
        print(f"  human entropy: mean={np.mean(human_entropy):.3f} std={np.std(human_entropy):.3f}")
        print(f"  LLM entropy:   mean={np.mean(llm_entropy):.3f} std={np.std(llm_entropy):.3f}")
        print(f"  U={u_stat:.0f}, p={u_p:.4f}")
        if u_p < 0.05:
            print(f"  → LLM attention IS significantly more uniform (p<0.05)")
        else:
            print(f"  → No significant difference in attention uniformity")

        # Arc comparison
        print(f"\n  Arc profiles:")
        human_analysis = analyze_attention(human_attn, human_data["paragraphs"],
                                           human_data["scene_breaks"], human_data["raw"])
        llm_analysis = analyze_attention(llm_attn, llm_data["paragraphs"],
                                          llm_data["scene_breaks"], llm_data["raw"])

        print(f"  {'':>8} {'Human':>40} {'LLM':>40}")
        for hs, ls in zip(human_analysis["arc_profile"], llm_analysis["arc_profile"]):
            h_sum = f"val={hs.get('valence_mean',0):.2f} stg={hs.get('boyd_staging',0):.3f} ten={hs.get('boyd_tension',0):.3f}"
            l_sum = f"val={ls.get('valence_mean',0):.2f} stg={ls.get('boyd_staging',0):.3f} ten={ls.get('boyd_tension',0):.3f}"
            print(f"  Scene {hs['scene']:>2} {h_sum:>40} {l_sum:>40}")


if __name__ == "__main__":
    main()
