"""Arena orchestrator — ties sampling, scanning, revision, judging, and ELO together."""
from __future__ import annotations

import asyncio
import itertools
import json
import sys
from pathlib import Path

from prose_doctor.critique_config import CritiqueConfig
from prose_doctor.arena.sampler import sample_stories
from prose_doctor.arena.scanner import scan_stories
from prose_doctor.arena.revision import run_match
from prose_doctor.arena.elo import EloTracker


def run_arena(
    config_paths: list[Path],
    dataset_path: Path,
    n_stories: int = 20,
    endpoint: str = "http://localhost:8081/v1",
    revision_model: str = "gpt-oss-120b",
    judge_model: str = "minimax",
    scan_workers: int = 2,
    revision_slots: int = 3,
    arena_dir: Path = Path("arena"),
) -> None:
    """Run a full arena tournament."""
    configs = [CritiqueConfig.from_yaml(p) for p in config_paths]
    print(f"Loaded {len(configs)} configs: {[c.name for c in configs]}", file=sys.stderr)

    matches_path = arena_dir / "results" / "matches.jsonl"
    ratings_path = arena_dir / "results" / "ratings.json"
    tracker = EloTracker.load(matches_path, ratings_path)
    for c in configs:
        tracker.add_config(c.name)

    # Create held-out set if it doesn't exist
    holdout_path = arena_dir / "holdout" / "story_ids.txt"
    if not holdout_path.exists():
        from prose_doctor.arena.sampler import create_holdout
        print("Creating held-out validation set (50 stories)...", file=sys.stderr)
        create_holdout(dataset_path, holdout_path, n=50)
    exclude = set(holdout_path.read_text().strip().split("\n"))

    print(f"Sampling {n_stories} stories...", file=sys.stderr)
    stories = sample_stories(dataset_path, n=n_stories, exclude_ids=exclude)
    print(f"  Got {len(stories)} stories", file=sys.stderr)

    print("Scanning stories...", file=sys.stderr)
    scanned = scan_stories(stories, arena_dir / "scanned", max_workers=scan_workers)
    scanned = [s for s in scanned if s.get("metrics", {}).get("total_distance", 0) > 2.0]
    print(f"  {len(scanned)} stories with total_distance > 2.0", file=sys.stderr)

    config_pairs = list(itertools.combinations(configs, 2))
    total = len(config_pairs) * len(scanned)
    print(f"Round-robin: {len(config_pairs)} config pairs x {len(scanned)} stories = {total} matches", file=sys.stderr)

    sem = asyncio.Semaphore(revision_slots)
    errors_path = arena_dir / "results" / "errors.jsonl"

    async def _run_all():
        for story in scanned:
            for cfg_a, cfg_b in config_pairs:
                try:
                    result = await run_match(
                        story, cfg_a, cfg_b,
                        endpoint, revision_model, judge_model, sem,
                    )
                    tracker.record_match(
                        result["config_a"], result["config_b"],
                        winner=result["winner"],
                        story_id=result["story_id"],
                        reason=result["reason"],
                        metadata={
                            "metrics_a_after": result["metrics_a_after"],
                            "metrics_b_after": result["metrics_b_after"],
                            "position_map": result["position_map"],
                        },
                    )
                    tracker.save(matches_path, ratings_path)
                    _print_leaderboard(tracker)
                except Exception as e:
                    print(f"  Match failed: {e}", file=sys.stderr)
                    errors_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(errors_path, "a") as ef:
                        ef.write(json.dumps({
                            "story_id": story.get("story_id", "unknown"),
                            "config_a": cfg_a.name,
                            "config_b": cfg_b.name,
                            "error": str(e),
                        }) + "\n")

    asyncio.run(_run_all())

    print("\n=== Final Leaderboard ===", file=sys.stderr)
    _print_leaderboard(tracker)


def _print_leaderboard(tracker: EloTracker) -> None:
    for name, elo in tracker.leaderboard():
        print(f"  {name:30s} {elo:7.1f}", file=sys.stderr)


def print_ratings(arena_dir: Path = Path("arena")) -> None:
    ratings_path = arena_dir / "results" / "ratings.json"
    if not ratings_path.exists():
        print("No ratings found. Run `prose-doctor arena run` first.")
        return
    tracker = EloTracker.load(Path("/dev/null"), ratings_path)
    for name, elo in tracker.leaderboard():
        print(f"  {name:30s} {elo:7.1f}")
