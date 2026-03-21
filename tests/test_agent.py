"""Tests for the revision agent.

These tests mock scan_deep to test tool logic
without requiring GPU or a running model server.
"""
from unittest.mock import patch
from prose_doctor.agent import RevisionContext, _do_replace
from prose_doctor.agent_models import ProseMetrics


def _make_metrics(**overrides) -> ProseMetrics:
    defaults = dict(
        pd_mean=0.336, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=8, ic_flatlines=3,
    )
    defaults.update(overrides)
    return ProseMetrics(**defaults)


def test_replace_passage_accepts_improvement():
    """replace_passage accepts when total_distance decreases."""
    ctx = RevisionContext(
        current_text="The old boring sentence was here.",
        filename="test.md",
    )
    ctx.last_metrics = _make_metrics(pd_mean=0.2)  # bad pd_mean
    improved = _make_metrics(pd_mean=0.3)  # better pd_mean

    with patch("prose_doctor.agent._do_scan", return_value=(improved, {})):
        result = _do_replace(
            ctx,
            old_text="The old boring sentence was here.",
            new_text="Rain hit the tin roof. She flinched.",
        )
    assert result.accepted
    assert ctx.current_text == "Rain hit the tin roof. She flinched."
    assert ctx.edits_accepted == 1


def test_replace_passage_rejects_regression():
    """replace_passage rejects when total_distance increases."""
    ctx = RevisionContext(
        current_text="The old sentence.",
        filename="test.md",
    )
    ctx.last_metrics = _make_metrics(pd_mean=0.3)
    worse = _make_metrics(pd_mean=0.1)  # regression

    with patch("prose_doctor.agent._do_scan", return_value=(worse, {})):
        result = _do_replace(
            ctx,
            old_text="The old sentence.",
            new_text="Bad rewrite.",
        )
    assert not result.accepted
    assert ctx.current_text == "The old sentence."  # reverted
    assert ctx.edits_rejected == 1


def test_replace_passage_old_text_not_found():
    """replace_passage fails gracefully when old_text doesn't match."""
    ctx = RevisionContext(
        current_text="Actual text in the chapter.",
        filename="test.md",
    )
    ctx.last_metrics = _make_metrics()

    result = _do_replace(ctx, old_text="nonexistent passage", new_text="whatever")
    assert not result.accepted
    assert "not found" in result.reason.lower()


def test_replace_passage_increments_turn():
    """Each replace_passage call increments turn count."""
    ctx = RevisionContext(
        current_text="Some text here.",
        filename="test.md",
    )
    ctx.last_metrics = _make_metrics(pd_mean=0.2)
    improved = _make_metrics(pd_mean=0.3)

    with patch("prose_doctor.agent._do_scan", return_value=(improved, {})):
        _do_replace(ctx, old_text="Some text here.", new_text="Better text.")
    assert ctx.turn == 1


def test_replace_passage_no_change_same_distance():
    """Reject if total_distance stays exactly the same (no improvement)."""
    ctx = RevisionContext(
        current_text="Text to replace.",
        filename="test.md",
    )
    same = _make_metrics(pd_mean=0.2)
    ctx.last_metrics = same

    with patch("prose_doctor.agent._do_scan", return_value=(same, {})):
        result = _do_replace(ctx, old_text="Text to replace.", new_text="New text.")
    assert not result.accepted
    assert ctx.current_text == "Text to replace."
