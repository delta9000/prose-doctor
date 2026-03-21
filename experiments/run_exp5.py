#!/usr/bin/env python3
"""Experiment 5: Does metric-informed prompting produce better first drafts?

Generates 10 stories × 2 conditions (bare vs informed prompt), scores all 20,
and compares paired metrics.

Usage:
    uv run --with openai python experiments/run_exp5.py
    uv run --with openai python experiments/run_exp5.py --model "claude-sonnet-4-20250514" --endpoint "https://api.anthropic.com/v1"
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROMPTS_FILE = Path(__file__).parent / "exp5_prompts.json"
OUTPUT_DIR = Path(__file__).parent / "exp5_output"

TECHNIQUE_SUFFIX = """

PROSE TECHNIQUE REQUIREMENTS (follow these precisely):
- 40% or more of sentences should use inverted structure: lead with a prepositional phrase, subordinate clause, or adverbial rather than subject-verb. "Through the window she watched" not "She watched through the window."
- Use sentence fragments sparingly — only at 2-3 moments of maximum impact. Merge all other fragments into full sentences.
- Vary information density: follow a dense, detail-packed paragraph with a sparse atmospheric beat. Let the reader breathe.
- Include deep interiority: the character's actual thoughts, raw and associative, not narrator summary. Not "she realized" — the realization itself.
- Include interoceptive sensory detail: heartbeat, breath, gut feeling, muscle tension, temperature on skin.
- The ending (final 3-5 paragraphs) must shift register — either zoom deep into interiority or pull back to a wide establishing shot. The ending should feel different from the middle.
- Do NOT use these words: tapestry, symphony, visceral, profound, resonance, liminal, ethereal.
- Do NOT use the pattern "It was not X. It was Y."
- Do NOT name characters Elara, Kael, Mara, or any AI-fantasy names."""

BARE_SUFFIX = """

Do NOT use these words: tapestry, symphony, visceral, profound, resonance, liminal, ethereal.
Do NOT use the pattern "It was not X. It was Y."
Do NOT name characters Elara, Kael, Mara, or any AI-fantasy names."""


def generate_story(prompt: str, model: str, endpoint: str, api_key: str) -> str:
    """Generate a story via OpenAI-compatible API."""
    from openai import OpenAI

    client = OpenAI(base_url=endpoint, api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a fiction writer. Write only the story — no headers, no meta-commentary, no author notes."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=8000,
        temperature=0.8,
    )
    return resp.choices[0].message.content.strip()


def score_story(filepath: Path) -> dict | None:
    """Run prose-doctor scan --deep --json on a story file."""
    result = subprocess.run(
        ["uv", "run", "--with", "sentence-transformers", "--with", "torch",
         "--with", "spacy", "--with", "numpy", "--with", "transformers",
         "python", "-m", "prose_doctor", "scan", "--deep", "--json", str(filepath)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  Scoring failed for {filepath.name}: {result.stderr[:200]}", file=sys.stderr)
        return None

    try:
        return json.loads(result.stdout)["chapters"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        print(f"  Parse failed for {filepath.name}", file=sys.stderr)
        return None


def extract_metrics(report: dict) -> dict:
    """Extract comparable metrics from a scan report."""
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
    "pd_mean": (0.336, "higher"),
    "pd_std": (0.093, "higher"),
    "fg_inversion": (44.2, "higher"),
    "fg_sl_cv": (0.706, "higher"),
    "fg_fragment": (6.7, "lower"),
    "ic_rhythmicity": (0.129, "lower"),
    "ic_spikes": (7.7, "higher"),
    "fg_index": (7.18, "higher"),
}


def distance_to_baseline(metrics: dict) -> dict:
    """Compute absolute distance to human baseline for each metric."""
    distances = {}
    for key, (baseline, _) in BASELINES.items():
        distances[key] = abs(metrics.get(key, 0) - baseline)
    return distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-oss-120b",
                        help="Model name for generation")
    parser.add_argument("--endpoint", default="http://localhost:8081/v1",
                        help="API endpoint")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip generation, only score existing files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(PROMPTS_FILE.read_text())

    if not args.score_only:
        print(f"Generating {len(prompts)} stories × 2 conditions...")
        print(f"Model: {args.model}")
        print(f"Endpoint: {args.endpoint}")
        print()

        for i, premise in enumerate(prompts):
            for condition, suffix in [("bare", BARE_SUFFIX), ("informed", TECHNIQUE_SUFFIX)]:
                outfile = OUTPUT_DIR / f"story_{i:02d}_{condition}.md"
                if outfile.exists():
                    print(f"  {outfile.name} exists, skipping")
                    continue

                prompt = premise + suffix
                print(f"  Generating {outfile.name}...")
                try:
                    text = generate_story(prompt, args.model, args.endpoint, args.api_key)
                    outfile.write_text(text)
                    print(f"    {len(text.split())} words")
                except Exception as e:
                    print(f"    FAILED: {e}", file=sys.stderr)

                time.sleep(1)  # rate limit courtesy

    # Score all stories
    print("\nScoring all stories...")
    results = {"bare": [], "informed": []}

    for i in range(len(prompts)):
        for condition in ["bare", "informed"]:
            filepath = OUTPUT_DIR / f"story_{i:02d}_{condition}.md"
            if not filepath.exists():
                print(f"  Missing {filepath.name}")
                results[condition].append(None)
                continue

            print(f"  Scoring {filepath.name}...")
            report = score_story(filepath)
            if report:
                metrics = extract_metrics(report)
                results[condition].append(metrics)
            else:
                results[condition].append(None)

    # Analysis
    print("\n" + "=" * 70)
    print("EXPERIMENT 5 RESULTS")
    print("=" * 70)

    import numpy as np

    metric_names = list(BASELINES.keys())
    bare_dists = {m: [] for m in metric_names}
    informed_dists = {m: [] for m in metric_names}
    bare_raw = {m: [] for m in metric_names}
    informed_raw = {m: [] for m in metric_names}

    valid_pairs = 0
    for i in range(len(prompts)):
        b = results["bare"][i]
        inf = results["informed"][i]
        if b is None or inf is None:
            continue
        valid_pairs += 1
        b_dist = distance_to_baseline(b)
        i_dist = distance_to_baseline(inf)
        for m in metric_names:
            bare_dists[m].append(b_dist[m])
            informed_dists[m].append(i_dist[m])
            bare_raw[m].append(b[m])
            informed_raw[m].append(inf[m])

    print(f"\nValid pairs: {valid_pairs}/{len(prompts)}")
    print(f"\n{'Metric':<20} {'Baseline':>8} {'Bare':>8} {'Informed':>8} {'Bare Δ':>8} {'Inf Δ':>8} {'Winner':>8}")
    print("-" * 72)

    bare_wins, informed_wins, ties = 0, 0, 0
    for m in metric_names:
        baseline, direction = BASELINES[m]
        b_mean = np.mean(bare_raw[m])
        i_mean = np.mean(informed_raw[m])
        b_dist_mean = np.mean(bare_dists[m])
        i_dist_mean = np.mean(informed_dists[m])

        if i_dist_mean < b_dist_mean - 0.001:
            winner = "INFORM"
            informed_wins += 1
        elif b_dist_mean < i_dist_mean - 0.001:
            winner = "BARE"
            bare_wins += 1
        else:
            winner = "tie"
            ties += 1

        print(f"  {m:<18} {baseline:>8.2f} {b_mean:>8.3f} {i_mean:>8.3f} "
              f"{b_dist_mean:>8.3f} {i_dist_mean:>8.3f} {winner:>8}")

    print(f"\nInformed wins: {informed_wins}  Bare wins: {bare_wins}  Ties: {ties}")

    # Wilcoxon signed-rank test on total distance
    bare_total = [sum(bare_dists[m][i] for m in metric_names) for i in range(valid_pairs)]
    informed_total = [sum(informed_dists[m][i] for m in metric_names) for i in range(valid_pairs)]

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(bare_total, informed_total)
        print(f"\nWilcoxon signed-rank (total distance to baseline):")
        print(f"  statistic={stat:.3f}, p={p:.4f}")
        print(f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'} at p<0.05")
        print(f"  Bare mean total distance: {np.mean(bare_total):.3f}")
        print(f"  Informed mean total distance: {np.mean(informed_total):.3f}")
    except ImportError:
        print("\n  (scipy not available for Wilcoxon test)")

    # Save raw results
    results_file = OUTPUT_DIR / "exp5_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "valid_pairs": valid_pairs,
            "bare": [r for r in results["bare"] if r],
            "informed": [r for r in results["informed"] if r],
            "baselines": BASELINES,
        }, f, indent=2)
    print(f"\nRaw results saved to {results_file}")


if __name__ == "__main__":
    main()
