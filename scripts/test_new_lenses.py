#!/usr/bin/env python3
"""Run the four new lenses on real corpus data and compare human vs LLM prose."""

from __future__ import annotations

import statistics
from pathlib import Path

from prose_doctor.providers import ProviderPool, require_ml
from prose_doctor.lenses.defaults import default_registry
from prose_doctor.lenses.runner import LensRunner
from prose_doctor.validation.tiers import load_tiers

ROOT = Path(__file__).resolve().parent.parent

HUMAN_DIR = ROOT / "corpus" / "baselines"
LLM_DIR = ROOT / "corpus" / "qwen3-30b-a3b"

TARGET_LENSES = ["concreteness", "referential_cohesion", "situation_shifts", "discourse_relations"]


def load_files(directory: Path, n: int = 5) -> list[tuple[str, str]]:
    """Return list of (filename, text) for the first n .md files sorted alphabetically."""
    files = sorted(directory.glob("*.md"))[:n]
    result = []
    for f in files:
        result.append((f.name, f.read_text()))
    return result


def run_lenses(runner: LensRunner, files: list[tuple[str, str]]) -> dict[str, list[dict]]:
    """Run target lenses on each file, return {lens_name: [per_chapter_dict, ...]}."""
    results: dict[str, list[dict]] = {name: [] for name in TARGET_LENSES}
    for filename, text in files:
        print(f"  Processing {filename}...")
        for lens_name in TARGET_LENSES:
            result = runner.run_one(lens_name, text, filename)
            results[lens_name].append(result.per_chapter or {})
    return results


def collect_raw(runner: LensRunner, files: list[tuple[str, str]], lens_name: str) -> list[dict]:
    """Run a single lens and return raw dicts for each file."""
    raws = []
    for filename, text in files:
        result = runner.run_one(lens_name, text, filename)
        raws.append(result.raw or {})
    return raws


def mean_of(dicts: list[dict], key: str) -> float | str:
    vals = [d[key] for d in dicts if key in d]
    if not vals:
        return "N/A"
    if isinstance(vals[0], (int, float)):
        return round(statistics.mean(vals), 4)
    return "N/A"


def print_comparison_table(
    lens_name: str,
    human_results: list[dict],
    llm_results: list[dict],
) -> None:
    # Collect all metric keys from both sets
    all_keys: set[str] = set()
    for d in human_results + llm_results:
        all_keys.update(d.keys())
    keys = sorted(all_keys)

    if not keys:
        print(f"  No per_chapter metrics found for {lens_name}")
        return

    # Determine column widths
    metric_width = max(len(k) for k in keys)
    metric_width = max(metric_width, len("Metric"))

    header = f"  {'Metric':<{metric_width}}  {'Human':>10}  {'LLM':>10}  {'Delta':>10}"
    print(header)
    print(f"  {'-' * metric_width}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for key in keys:
        h = mean_of(human_results, key)
        l = mean_of(llm_results, key)
        if isinstance(h, float) and isinstance(l, float):
            delta = round(l - h, 4)
            delta_str = f"{delta:>+10.4f}"
        else:
            delta_str = f"{'N/A':>10}"
        h_str = f"{h:>10.4f}" if isinstance(h, float) else f"{str(h):>10}"
        l_str = f"{l:>10.4f}" if isinstance(l, float) else f"{str(l):>10}"
        print(f"  {key:<{metric_width}}  {h_str}  {l_str}  {delta_str}")


def main() -> None:
    require_ml()

    print("Loading providers...")
    providers = ProviderPool()

    print("Building registry and runner...")
    registry = default_registry()
    tiers = load_tiers()
    runner = LensRunner(registry, providers, tier_filter="experimental", tiers=tiers)

    print(f"\nLoading human files from {HUMAN_DIR}")
    human_files = load_files(HUMAN_DIR, n=5)
    print(f"  Loaded: {[f[0] for f in human_files]}")

    print(f"\nLoading LLM files from {LLM_DIR}")
    llm_files = load_files(LLM_DIR, n=5)
    print(f"  Loaded: {[f[0] for f in llm_files]}")

    print("\n--- Running lenses on HUMAN files ---")
    human_results = run_lenses(runner, human_files)

    print("\n--- Running lenses on LLM files ---")
    llm_results = run_lenses(runner, llm_files)

    # Print comparison tables
    print("\n" + "=" * 70)
    print("COMPARISON: Human vs LLM (mean per-chapter metrics)")
    print("=" * 70)

    for lens_name in TARGET_LENSES:
        print(f"\n[{lens_name}]")
        print_comparison_table(lens_name, human_results[lens_name], llm_results[lens_name])

    # Qualitative sample: discourse_relations raw labels
    print("\n" + "=" * 70)
    print("QUALITATIVE SAMPLE: discourse_relations sentence labels")
    print("=" * 70)

    print(f"\nHuman sample ({human_files[0][0]}):")
    human_raw = collect_raw(runner, [human_files[0]], "discourse_relations")
    labels = human_raw[0].get("sentence_labels", [])
    evidence = human_raw[0].get("sentence_evidence", [])
    for i, (label, ev) in enumerate(zip(labels[:30], evidence[:30])):
        ev_str = f" ({ev})" if ev else ""
        print(f"  sent {i:3d}: {label}{ev_str}")
    if len(labels) > 30:
        print(f"  ... ({len(labels)} sentences total)")

    print(f"\nLLM sample ({llm_files[0][0]}):")
    llm_raw = collect_raw(runner, [llm_files[0]], "discourse_relations")
    labels = llm_raw[0].get("sentence_labels", [])
    evidence = llm_raw[0].get("sentence_evidence", [])
    for i, (label, ev) in enumerate(zip(labels[:30], evidence[:30])):
        ev_str = f" ({ev})" if ev else ""
        print(f"  sent {i:3d}: {label}{ev_str}")
    if len(labels) > 30:
        print(f"  ... ({len(labels)} sentences total)")


if __name__ == "__main__":
    main()
