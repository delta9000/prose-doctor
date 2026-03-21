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


def test_find_generic_issues():
    """Generic finder should return paragraphs with actionable prescriptions."""
    # Need enough paragraphs for meaningful analysis
    text = "\n\n".join([
        "The sun rose over the mountains. Birds sang in the trees. A light breeze rustled the leaves.",
        "She walked to the door and opened it. The hallway was dark. She stepped inside carefully.",
        "The old man sat by the fire. His hands were weathered and cracked. He stared into the flames.",
        '"I told you," she said. "We can\'t go back there." Her voice was sharp, cutting.',
        "He ran down the corridor, heart pounding. The footsteps behind him grew louder. He turned the corner.",
        "The room was empty except for a table. A single candle burned on it. Shadows danced on the walls.",
        "She thought about what he had said. Maybe he was right. Perhaps they should have waited longer.",
        "The garden was overgrown. Weeds pushed through the flagstones. A rusted gate hung open.",
        "He picked up the letter and read it again. The words hadn't changed. They never would.",
        '"Wait," he called. But she was already gone. The door closed behind her with a soft click.',
        "The city stretched below them. Lights flickered in windows. Somewhere a dog barked once, then fell silent.",
        "She remembered the summer they spent by the lake. The water had been warm. The days had been long.",
    ])
    report = {"psychic_distance": {"paragraph_means": [0.3] * 12}}
    issues = find_issues("generic", text, report)
    assert len(issues) > 0
    # Should have actionable prescriptions
    for issue in issues:
        assert "most average in" in issue.reason
        assert not issue.preserve


def test_find_issues_unknown_metric():
    """Unknown metrics should return empty list."""
    result = find_issues("nonexistent_metric", "some text", {})
    assert result == []


def test_format_issues_empty():
    """Empty issue list should return a helpful message."""
    output = format_issues([])
    assert "No specific issues" in output
