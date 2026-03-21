"""Async concurrent revision runner for arena matches."""
from __future__ import annotations

import asyncio
import sys

from prose_doctor.critique_config import CritiqueConfig


def revise_story_sync(
    text: str,
    config: CritiqueConfig,
    endpoint: str,
    model: str,
) -> tuple[str, dict]:
    """Run orchestrated revision synchronously.

    Not threaded — ProviderPool's lazy-loaded models (GPT-2, spaCy)
    are not thread-safe, so concurrent threads cause meta tensor errors.
    Revisions run sequentially; the LLM network calls are the real bottleneck.
    """
    from prose_doctor.orchestrated_revise import run_orchestrated
    result = run_orchestrated(
        text,
        max_turns=config.max_turns,
        endpoint=endpoint,
        model_name=model,
        critique_config=config,
    )
    return result.final_text, result.metrics_final.model_dump()


async def run_match(
    story: dict,
    config_a: CritiqueConfig,
    config_b: CritiqueConfig,
    endpoint: str,
    revision_model: str,
    judge_model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run one arena match: revise under both configs, judge the results."""
    from prose_doctor.arena.judge import judge_pair

    original = story["text"]
    story_id = story["story_id"]

    print(f"  Match: {config_a.name} vs {config_b.name} on {story_id}", file=sys.stderr)

    # Revise sequentially — ProviderPool models are not thread-safe
    revised_a_text, metrics_a = revise_story_sync(
        original, config_a, endpoint, revision_model,
    )
    revised_b_text, metrics_b = revise_story_sync(
        original, config_b, endpoint, revision_model,
    )

    # Judge
    judge_result = await judge_pair(
        original, revised_a_text, revised_b_text,
        config_a.name, config_b.name,
        endpoint, judge_model,
    )

    return {
        "story_id": story_id,
        "config_a": config_a.name,
        "config_b": config_b.name,
        "winner": judge_result["winner"],
        "reason": judge_result["reason"],
        "position_map": judge_result.get("position_map", {}),
        "metrics_a_before": story.get("metrics", {}),
        "metrics_a_after": metrics_a,
        "metrics_b_before": story.get("metrics", {}),
        "metrics_b_after": metrics_b,
    }
