"""Async concurrent revision runner for arena matches."""
from __future__ import annotations

import asyncio
import sys

from prose_doctor.critique_config import CritiqueConfig


async def revise_story(
    text: str,
    config: CritiqueConfig,
    endpoint: str,
    model: str,
) -> tuple[str, dict]:
    """Run orchestrated revision in a thread (it's sync internally)."""
    loop = asyncio.get_event_loop()

    def _run():
        from prose_doctor.orchestrated_revise import run_orchestrated
        result = run_orchestrated(
            text,
            max_turns=config.max_turns,
            endpoint=endpoint,
            model_name=model,
            critique_config=config,
        )
        return result.final_text, result.metrics_final.model_dump()

    return await loop.run_in_executor(None, _run)


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

    # Revise under both configs concurrently (bounded by semaphore)
    async def _bounded(cfg):
        async with semaphore:
            return await revise_story(original, cfg, endpoint, revision_model)

    (revised_a_text, metrics_a), (revised_b_text, metrics_b) = await asyncio.gather(
        _bounded(config_a),
        _bounded(config_b),
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
