"""Snapshot tests for regex patterns against sample prose."""

from prose_doctor.patterns.rules import build_rule_patterns, check_rules


def test_not_x_period_y():
    text = "Not a sound. A structure."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "not_x_period_y" in names


def test_not_x_not_y_z():
    text = "Not a voice. Not a memory. A scar."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "not_x_not_y_z" in names


def test_wasnt_it_was():
    text = "It wasn\u2019t fear. It was something older."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "wasnt_it_was" in names


def test_began_to():
    text = "He began to walk toward the door."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "began_started_to" in names


def test_seemed_to():
    text = "The ground seemed to shift beneath them."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "seemed_to" in names


def test_weight_of_abstraction():
    text = "She felt the weight of the silence pressing down."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "weight_of_abstraction" in names


def test_something_in_pronoun():
    text = "Something in her shifted."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "something_in_pronoun" in names


def test_something_in_character_name():
    patterns = build_rule_patterns(["Rook", "Cassian"])
    text = "Something in Rook changed."
    hits = check_rules(text, patterns)
    names = [h["pattern_name"] for h in hits]
    assert "something_in_pronoun" in names


def test_phantom_sensation():
    text = "The truth settled in his chest."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "phantom_sensation" in names


def test_forbidden_word():
    text = "The tapestry of stars hung above them."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "forbidden_word" in names


def test_emotion_naming():
    text = "She felt a surge of dread."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "emotion_naming" in names


def test_adverb_dialogue_tag():
    text = '"Stop," he said quietly.'
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "adverb_dialogue_tag" in names


def test_clean_text_no_hits():
    text = "The rain fell on the roof. She opened the window."
    hits = check_rules(text)
    # Should have no hits (or at most benign ones)
    high_hits = [h for h in hits if h["severity"] == "high"]
    assert len(high_hits) == 0


def test_didnt_verb_verbed():
    text = "He didn\u2019t reach. He walked away."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "didnt_verb_verbed" in names


def test_that_was_what():
    text = "That was what it meant."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "that_was_what" in names


def test_role_quality_label():
    text = "She moved with a cartographer's precision."
    hits = check_rules(text)
    names = [h["pattern_name"] for h in hits]
    assert "role_quality_label" in names
