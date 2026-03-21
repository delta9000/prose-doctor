"""Tests for per-passage issue finder."""
from prose_doctor.agent_issues import find_fragment_issues, find_issues, format_issues


SAMPLE_TEXT = """\
Rook checked the pack straps again. Third time in an hour. Not because \
you expected to use it, but because the weight kept you anchored.

Not one voice. Not ten. A chorus. Layered and overlapping, speaking in \
fragments of sentences that never quite completed. A woman laughing. \
A child calling a name. An old man coughing.

The corridor stretched ahead. Rook walked forward. He moved carefully. \
The floor was wet. The walls were dark. He kept going.
"""


def test_find_fragment_issues_classifies_crutch_vs_intentional():
    """Fragment finder should distinguish crutch fragments from intentional ones."""
    issues = find_fragment_issues(SAMPLE_TEXT, {})

    # Should find fragments
    assert len(issues) > 0

    # The chorus paragraph fragments (3+ in sequence) should be marked preserve
    chorus_issues = [i for i in issues if i.paragraph_idx == 1]
    if chorus_issues:
        preserves = [i for i in chorus_issues if i.preserve]
        assert len(preserves) > 0, "Chorus fragments should be marked as preserve"

    # Check that at least some are marked as fixable
    fixable = [i for i in issues if not i.preserve]
    # The third paragraph has monotonous short sentences
    assert any(not i.preserve for i in issues), "Should have some fixable fragments"


def test_format_issues_separates_fix_and_preserve():
    """Formatter should separate fix vs preserve sections."""
    issues = find_fragment_issues(SAMPLE_TEXT, {})
    output = format_issues(issues)

    # Should have both sections if we have both types
    fix_count = sum(1 for i in issues if not i.preserve)
    preserve_count = sum(1 for i in issues if i.preserve)

    if fix_count > 0:
        assert "Fix These" in output
    if preserve_count > 0:
        assert "Preserve These" in output


def test_find_issues_unknown_metric():
    """Unknown metrics should return empty list."""
    result = find_issues("nonexistent_metric", "some text", {})
    assert result == []


def test_format_issues_empty():
    """Empty issue list should return a helpful message."""
    output = format_issues([])
    assert "No specific issues" in output
