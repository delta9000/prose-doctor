"""Tests for per-passage issue finder."""
from prose_doctor.agent_issues import find_fragment_issues, find_discourse_issues, find_concreteness_issues, find_shift_issues, find_issues, format_issues


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


def test_find_shift_issues_flags_static_scene():
    """A long scene with no shifts should get flagged."""
    # 6 paragraphs, same time/place/character
    text = "\n\n".join([
        "Marcus sat at the table. He stared at the map.",
        "He traced the route with his finger. The line ran north.",
        "He leaned back in the chair. The map was wrong.",
        "He checked the coordinates again. Still wrong.",
        "He pulled out a second map. This one was older.",
        "He compared the two. The discrepancy was clear.",
    ])
    report = {"situation_shifts": {"total_shifts": 0, "time_shifts": 0, "space_shifts": 0, "actor_shifts": 0}}
    issues = find_shift_issues(text, report)
    assert len(issues) > 0
    assert any("static" in i.reason.lower() or "shift" in i.reason.lower() for i in issues)


def test_find_concreteness_issues_flags_no_abstraction():
    """Relentlessly concrete prose should get flagged."""
    text = (
        "He pressed his back against the brick wall. His fingers found "
        "the crack in the mortar, cold and damp. The flashlight in his left "
        "hand was dead weight now, batteries drained hours ago.\n\n"
        "She set the coffee mug on the counter. The ceramic clinked against "
        "the granite. Outside, rain hammered the tin roof.\n\n"
        "He grabbed the door handle. The metal was cold. He twisted it. "
        "The hinges squeaked. The hallway stretched out before him.\n\n"
        "The knife lay on the table. Steel glinted in the lamplight. A bead "
        "of water ran down the blade.\n\n"
        "She pulled the rope taut. The knot held. The boat rocked against "
        "the dock, wood scraping wood."
    )
    report = {"concreteness": {
        "concreteness_mean": 3.4,
        "abstractness_ratio": 0.05,
        "vague_noun_density": 0.0,
    }}
    issues = find_concreteness_issues(text, report)
    assert len(issues) > 0
    assert any("abstract" in i.reason.lower() or "reflect" in i.reason.lower() for i in issues)


def test_find_discourse_issues():
    """Additive-only prose should produce discourse issues."""
    text = (
        "Marcus walked down the corridor. And the walls were grey. And the "
        "floor was concrete. And the lights hummed overhead.\n\n"
        "He reached the end of the hall. And there was a door. And the door "
        "was locked. And he tried the handle again.\n\n"
        "The room beyond was small. And it was empty. And the window was "
        "boarded up. And dust covered every surface."
    )
    report = {"discourse_relations": {"relation_entropy": 0.15, "implicit_ratio": 0.95}}
    issues = find_discourse_issues(text, report)
    assert len(issues) > 0
    assert any("connective" in i.reason.lower() or "implicit" in i.reason.lower() for i in issues)


def test_find_discourse_issues_healthy():
    """Prose with good entropy and low implicit ratio should have no issues."""
    text = (
        "Marcus waited because the guard hadn't passed yet. Although the "
        "corridor looked clear, he knew better.\n\n"
        "Then he moved, fast and low. But the floor creaked under his weight. "
        "Consequently, the guard turned.\n\n"
        "Meanwhile, Elena watched from the rooftop. She could see the whole "
        "courtyard, so she tracked his progress."
    )
    report = {"discourse_relations": {"relation_entropy": 0.70, "implicit_ratio": 0.85}}
    issues = find_discourse_issues(text, report)
    assert len(issues) == 0


def test_format_issues_empty():
    """Empty issue list should return a helpful message."""
    output = format_issues([])
    assert "No specific issues" in output
