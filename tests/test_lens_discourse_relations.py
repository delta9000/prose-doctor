import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.discourse_relations import DiscourseRelationsLens
from prose_doctor.providers import ProviderPool


DIVERSE_SAMPLE = '''
Marcus waited because the guard hadn't passed yet. Although the
corridor looked clear, he knew better than to trust appearances.

Then he moved, fast and low. But the floor creaked under his weight.
Consequently, the guard turned. However, Marcus was already through
the door.

Meanwhile, Elena watched from the rooftop. She could see the whole
courtyard, so she tracked his progress through the shadows.
'''

ADDITIVE_SAMPLE = '''
Marcus walked down the corridor. And the walls were grey. And the
floor was concrete. And the lights hummed overhead.

He reached the end of the hall. And there was a door. And the door
was locked. And he tried the handle again.

The room beyond was small. And it was empty. And the window was
boarded up. And dust covered every surface.
'''


@pytest.mark.slow
def test_discourse_returns_result():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    result = lens.analyze(DIVERSE_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "discourse_relations"
    assert result.per_chapter is not None
    assert "relation_entropy" in result.per_chapter


@pytest.mark.slow
def test_diverse_has_higher_entropy():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    diverse = lens.analyze(DIVERSE_SAMPLE, "diverse.md", pool)
    additive = lens.analyze(ADDITIVE_SAMPLE, "additive.md", pool)
    assert diverse.per_chapter["relation_entropy"] > additive.per_chapter["relation_entropy"]


@pytest.mark.slow
def test_additive_sample_flags_additive_zones():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    result = lens.analyze(ADDITIVE_SAMPLE, "additive.md", pool)
    assert result.per_chapter.get("additive_only_zones", 0) > 0


@pytest.mark.slow
def test_discourse_has_per_paragraph():
    lens = DiscourseRelationsLens()
    pool = ProviderPool()
    result = lens.analyze(DIVERSE_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "relation_diversity" in result.per_paragraph


def test_discourse_metadata():
    lens = DiscourseRelationsLens()
    assert lens.name == "discourse_relations"
    assert lens.requires_providers == ["spacy"]
