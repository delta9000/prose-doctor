"""Tests for the blind pairwise judge module."""
from __future__ import annotations

import random

import pytest

from prose_doctor.arena.judge import build_judge_prompt, parse_judge_response


def test_build_judge_prompt_randomizes_position():
    rng = random.Random(42)
    prompt, mapping = build_judge_prompt("orig", "rev_a", "rev_b", rng)
    assert "VERSION X:" in prompt
    assert "VERSION Y:" in prompt
    assert mapping["X"] in ("a", "b")
    assert mapping["Y"] in ("a", "b")
    assert mapping["X"] != mapping["Y"]


def test_build_judge_prompt_contains_original():
    prompt, _ = build_judge_prompt("the original text", "rev a", "rev b")
    assert "the original text" in prompt


def test_build_judge_prompt_identical_texts_no_diffs():
    """Identical revisions produce a no-differences prompt."""
    prompt, _ = build_judge_prompt("same text", "same text", "same text")
    assert "No differences found" in prompt


def test_build_judge_prompt_shows_diffs():
    """Changed paragraphs appear in the prompt with context."""
    orig = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    rev_a = "First paragraph.\n\nRevised A second.\n\nThird paragraph."
    rev_b = "First paragraph.\n\nRevised B second.\n\nThird paragraph."
    prompt, _ = build_judge_prompt(orig, rev_a, rev_b)
    assert "Change 1" in prompt
    assert "Second paragraph" in prompt  # original shown
    assert "VERSION X:" in prompt
    assert "VERSION Y:" in prompt


def test_build_judge_prompt_both_assignments_reachable():
    """With enough seeds, both X->a and X->b assignments should occur."""
    assignments = set()
    for seed in range(20):
        _, mapping = build_judge_prompt("o", "a", "b", random.Random(seed))
        assignments.add(mapping["X"])
    assert assignments == {"a", "b"}


def test_parse_judge_response_valid_json():
    resp = '{"winner": "X", "reason": "better rhythm"}'
    result = parse_judge_response(resp, {"X": "config_a", "Y": "config_b"})
    assert result["winner"] == "config_a"
    assert result["reason"] == "better rhythm"


def test_parse_judge_response_tie():
    resp = '{"winner": "tie", "reason": "too close"}'
    result = parse_judge_response(resp, {"X": "a", "Y": "b"})
    assert result["winner"] == "tie"


def test_parse_judge_response_with_think_tags():
    resp = '<think>Let me analyze...</think>{"winner": "Y", "reason": "more varied"}'
    result = parse_judge_response(resp, {"X": "a", "Y": "b"})
    assert result["winner"] == "b"


def test_parse_judge_response_malformed():
    resp = "I think version X is better because of reasons"
    result = parse_judge_response(resp, {"X": "a", "Y": "b"})
    assert result["winner"] == "tie"  # fallback


def test_parse_judge_response_multiline_think_tags():
    resp = "<think>\nStep 1: consider rhythm.\nStep 2: consider flow.\n</think>\n{\"winner\": \"X\", \"reason\": \"stronger voice\"}"
    result = parse_judge_response(resp, {"X": "cfg1", "Y": "cfg2"})
    assert result["winner"] == "cfg1"
    assert result["reason"] == "stronger voice"


def test_parse_judge_response_markdown_fences():
    resp = "```json\n{\"winner\": \"Y\", \"reason\": \"cleaner\"}\n```"
    result = parse_judge_response(resp, {"X": "alpha", "Y": "beta"})
    assert result["winner"] == "beta"


def test_parse_judge_response_y_winner():
    resp = '{"winner": "Y", "reason": "flows better"}'
    result = parse_judge_response(resp, {"X": "config_a", "Y": "config_b"})
    assert result["winner"] == "config_b"
    assert result["reason"] == "flows better"
