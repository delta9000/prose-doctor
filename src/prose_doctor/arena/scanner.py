"""Parallel scan worker pool using ProcessPoolExecutor with spawn context."""
from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _scan_one(story: dict) -> dict | None:
    """Scan a single story. Runs in a spawned child process."""
    try:
        from prose_doctor.providers import require_ml
        require_ml()
        from prose_doctor.agent_scan import scan_deep

        metrics, report = scan_deep(story["text"], filename=f"{story['story_id']}.md")
        return {
            "story_id": story["story_id"],
            "genre": story["genre"],
            "text": story["text"],
            "word_count": story["word_count"],
            "chapter_num": story["chapter_num"],
            "metrics": metrics.model_dump(),
            "report": report,
        }
    except Exception as e:
        import sys
        print(f"Scan failed for {story['story_id']}: {e}", file=sys.stderr)
        return None


def scan_stories(
    stories: list[dict],
    cache_dir: Path,
    max_workers: int = 2,
) -> list[dict]:
    """Scan stories in parallel, caching results.

    Uses spawn context to avoid CUDA/spaCy fork corruption.
    """
    results = []
    to_scan = []

    cache_dir.mkdir(parents=True, exist_ok=True)
    for story in stories:
        cache_path = cache_dir / f"{story['story_id']}.json"
        if cache_path.exists():
            results.append(json.loads(cache_path.read_text()))
        else:
            to_scan.append(story)

    if not to_scan:
        return results

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        for result in pool.map(_scan_one, to_scan):
            if result is not None:
                cache_path = cache_dir / f"{result['story_id']}.json"
                cache_path.write_text(json.dumps(result))
                results.append(result)

    return results
