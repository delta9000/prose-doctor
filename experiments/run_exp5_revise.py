#!/usr/bin/env python3
"""Experiment 5b: Post-hoc critique + revision on the bare-prompt stories.

Takes the 10 bare-prompt stories from Experiment 5, runs prose-doctor
critique on each, sends the critique + original to an LLM for revision,
then scores the revisions. Compares bare → revised vs bare → informed.

Usage:
    uv run --with openai python experiments/run_exp5_revise.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "exp5_output"

REVISION_SYSTEM = """You are revising a short story based on a prose analysis critique.
Read the critique carefully and apply the specific prescriptions.
Preserve the plot, characters, dialogue, and tone.
Focus on prose technique, not story changes.
Output only the revised story — no commentary, no headers."""


def get_critique(filepath: Path) -> str:
    """Run prose-doctor critique on a story."""
    result = subprocess.run(
        ["uv", "run", "--with", "sentence-transformers", "--with", "torch",
         "--with", "spacy", "--with", "numpy", "--with", "transformers",
         "python", "-m", "prose_doctor", "critique", str(filepath)],
        capture_output=True, text=True, timeout=600,
    )
    return result.stdout.strip()


def revise_story(original: str, critique: str, model: str, endpoint: str, api_key: str) -> str:
    """Send original + critique to LLM for revision."""
    from openai import OpenAI
    client = OpenAI(base_url=endpoint, api_key=api_key)

    user_prompt = f"""Here is the original story:

---
{original}
---

Here is the prose analysis critique:

---
{critique}
---

Rewrite the story addressing the critique's priority issues and improvements.
Preserve the strengths listed. Keep all plot, characters, and dialogue."""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVISION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8000,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def score_story(filepath: Path) -> dict | None:
    """Run prose-doctor scan --deep --json."""
    result = subprocess.run(
        ["uv", "run", "--with", "sentence-transformers", "--with", "torch",
         "--with", "spacy", "--with", "numpy", "--with", "transformers",
         "python", "-m", "prose_doctor", "scan", "--deep", "--json", str(filepath)],
        capture_output=True, text=True, timeout=600,
    )
    try:
        return json.loads(result.stdout)["chapters"][0]
    except:
        return None


def extract_metrics(report: dict) -> dict:
    pd = report.get("psychic_distance") or {}
    fg = report.get("foregrounding") or {}
    ic = report.get("info_contour") or {}
    return {
        "pd_mean": pd.get("mean_distance", 0),
        "pd_std": pd.get("std_distance", 0),
        "fg_inversion": fg.get("inversion_pct", 0),
        "fg_sl_cv": fg.get("sentence_length_cv", 0),
        "fg_fragment": fg.get("fragment_pct", 0),
        "ic_rhythmicity": ic.get("rhythmicity", 0),
        "ic_spikes": ic.get("spikes", 0) if isinstance(ic.get("spikes"), (int, float)) else 0,
        "fg_index": fg.get("index", 0),
    }


BASELINES = {
    "pd_mean": 0.336, "pd_std": 0.093, "fg_inversion": 44.2,
    "fg_sl_cv": 0.706, "fg_fragment": 6.7, "ic_rhythmicity": 0.129,
    "ic_spikes": 7.7, "fg_index": 7.18,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--endpoint", default="http://localhost:8081/v1")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    # Phase 1: Critique and revise each bare story
    if not args.score_only:
        for i in range(10):
            bare_file = OUTPUT_DIR / f"story_{i:02d}_bare.md"
            revised_file = OUTPUT_DIR / f"story_{i:02d}_revised.md"

            if revised_file.exists():
                print(f"  {revised_file.name} exists, skipping")
                continue

            if not bare_file.exists():
                print(f"  {bare_file.name} missing, skipping")
                continue

            print(f"  Critiquing {bare_file.name}...")
            critique = get_critique(bare_file)
            if not critique:
                print(f"    Critique failed, skipping")
                continue

            print(f"  Revising → {revised_file.name}...")
            original = bare_file.read_text()
            try:
                revised = revise_story(original, critique, args.model, args.endpoint, args.api_key)
                revised_file.write_text(revised)
                print(f"    {len(revised.split())} words")
            except Exception as e:
                print(f"    FAILED: {e}")

            time.sleep(1)

    # Phase 2: Score all three conditions
    print("\nScoring all conditions...")
    results = {"bare": [], "informed": [], "revised": []}

    for i in range(10):
        for condition in ["bare", "informed", "revised"]:
            filepath = OUTPUT_DIR / f"story_{i:02d}_{condition}.md"
            if not filepath.exists():
                results[condition].append(None)
                continue
            print(f"  Scoring {filepath.name}...")
            report = score_story(filepath)
            results[condition].append(extract_metrics(report) if report else None)

    # Phase 3: Analysis
    import numpy as np

    print("\n" + "=" * 80)
    print("EXPERIMENT 5b: BARE vs INFORMED vs REVISED")
    print("=" * 80)

    metric_names = list(BASELINES.keys())

    # Collect distances to baseline
    conditions = ["bare", "informed", "revised"]
    dists = {c: {m: [] for m in metric_names} for c in conditions}
    raw = {c: {m: [] for m in metric_names} for c in conditions}

    valid = 0
    for i in range(10):
        if any(results[c][i] is None for c in conditions):
            continue
        valid += 1
        for c in conditions:
            for m in metric_names:
                val = results[c][i][m]
                raw[c][m].append(val)
                dists[c][m].append(abs(val - BASELINES[m]))

    print(f"\nValid triplets: {valid}/10")
    print(f"\n{'Metric':<18} {'Base':>6} {'Bare':>8} {'Inform':>8} {'Revised':>8}  {'Best':>8}")
    print("-" * 72)

    wins = {c: 0 for c in conditions}
    for m in metric_names:
        b_mean = np.mean(dists["bare"][m])
        i_mean = np.mean(dists["informed"][m])
        r_mean = np.mean(dists["revised"][m])
        best = min([(b_mean, "BARE"), (i_mean, "INFORM"), (r_mean, "REVISED")])[1]
        wins[{"BARE": "bare", "INFORM": "informed", "REVISED": "revised"}[best]] += 1

        print(f"  {m:<16} {BASELINES[m]:>6.2f} "
              f"{np.mean(raw['bare'][m]):>8.3f} "
              f"{np.mean(raw['informed'][m]):>8.3f} "
              f"{np.mean(raw['revised'][m]):>8.3f}  "
              f"{best:>8}")

    print(f"\nWins: Bare={wins['bare']}  Informed={wins['informed']}  Revised={wins['revised']}")

    # Wilcoxon: bare vs revised
    try:
        from scipy.stats import wilcoxon
        bare_total = [sum(dists["bare"][m][i] for m in metric_names) for i in range(valid)]
        revised_total = [sum(dists["revised"][m][i] for m in metric_names) for i in range(valid)]
        informed_total = [sum(dists["informed"][m][i] for m in metric_names) for i in range(valid)]

        print(f"\nTotal distance to baseline (lower = better):")
        print(f"  Bare:     {np.mean(bare_total):.3f}")
        print(f"  Informed: {np.mean(informed_total):.3f}")
        print(f"  Revised:  {np.mean(revised_total):.3f}")

        stat_br, p_br = wilcoxon(bare_total, revised_total)
        print(f"\nBare vs Revised: W={stat_br:.1f}, p={p_br:.4f} {'*' if p_br < 0.05 else 'ns'}")

        stat_ir, p_ir = wilcoxon(informed_total, revised_total)
        print(f"Informed vs Revised: W={stat_ir:.1f}, p={p_ir:.4f} {'*' if p_ir < 0.05 else 'ns'}")

        stat_bi, p_bi = wilcoxon(bare_total, informed_total)
        print(f"Bare vs Informed: W={stat_bi:.1f}, p={p_bi:.4f} {'*' if p_bi < 0.05 else 'ns'}")
    except ImportError:
        print("\n(scipy not available)")

    # Save
    with open(OUTPUT_DIR / "exp5b_results.json", "w") as f:
        json.dump({"valid": valid, "raw": raw, "dists": dists}, f, indent=2, default=list)
    print(f"\nSaved to {OUTPUT_DIR / 'exp5b_results.json'}")


if __name__ == "__main__":
    main()
