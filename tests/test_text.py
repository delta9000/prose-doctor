"""Tests for text splitting and word counting."""

from prose_doctor.text import (
    split_paragraphs,
    split_paragraphs_with_breaks,
    count_words,
    is_dialogue_line,
)


def test_split_paragraphs_frontmatter(sample_chapter):
    paras = split_paragraphs(sample_chapter)
    # Should skip the header and scene break marker
    assert len(paras) >= 5
    assert paras[0].startswith("The bunker smelled")
    # Should not contain "---" or header lines
    for p in paras:
        assert p != "---"
        assert not p.startswith("title:")


def test_split_paragraphs_plain(sample_plain):
    paras = split_paragraphs(sample_plain)
    assert len(paras) >= 3
    assert paras[0].startswith("The forest was quiet")


def test_split_paragraphs_with_breaks(sample_chapter):
    items = split_paragraphs_with_breaks(sample_chapter)
    # Should contain at least one None (scene break)
    assert None in items
    # Non-None items should be strings
    for item in items:
        if item is not None:
            assert isinstance(item, str)
            assert len(item) > 0


def test_count_words(sample_chapter):
    wc = count_words(sample_chapter)
    assert wc > 50


def test_count_words_plain(sample_plain):
    wc = count_words(sample_plain)
    assert wc > 20


def test_is_dialogue_line():
    assert is_dialogue_line('"Hello," she said.')
    assert is_dialogue_line('\u201cHello,\u201d she said.')
    assert not is_dialogue_line("The sky was dark.")
