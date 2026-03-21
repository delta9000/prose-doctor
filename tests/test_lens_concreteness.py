import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.concreteness import ConcretenessLens
from prose_doctor.providers import ProviderPool


CONCRETE_SAMPLE = '''
Marcus pressed his back against the brick wall. His fingers found
the crack in the mortar, cold and damp. The flashlight in his left
hand was dead weight now, batteries drained hours ago.

She set the coffee mug on the counter. The ceramic clinked against
the granite. Outside, rain hammered the tin roof.
'''

ABSTRACT_SAMPLE = '''
The significance of the moment was not lost on anyone. There was a
sense of possibility in the air, a feeling that things were about
to change in ways that mattered.

Something shifted in the dynamic between them. The weight of
unspoken truths hung in the silence, heavy with implication and
the kind of meaning that resisted easy articulation.
'''


def test_concreteness_returns_result():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(CONCRETE_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "concreteness"
    assert result.per_chapter is not None
    assert "concreteness_mean" in result.per_chapter
    assert "abstractness_ratio" in result.per_chapter


def test_concrete_scores_higher_than_abstract():
    lens = ConcretenessLens()
    pool = ProviderPool()
    concrete = lens.analyze(CONCRETE_SAMPLE, "concrete.md", pool)
    abstract = lens.analyze(ABSTRACT_SAMPLE, "abstract.md", pool)
    assert concrete.per_chapter["concreteness_mean"] > abstract.per_chapter["concreteness_mean"]


def test_concreteness_has_per_sentence():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(CONCRETE_SAMPLE, "test.md", pool)
    assert result.per_sentence is not None
    assert "concreteness" in result.per_sentence
    assert len(result.per_sentence["concreteness"]) > 0


def test_concreteness_has_per_paragraph():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(CONCRETE_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "concreteness" in result.per_paragraph


def test_abstract_sample_flags_vague_nouns():
    lens = ConcretenessLens()
    pool = ProviderPool()
    result = lens.analyze(ABSTRACT_SAMPLE, "abstract.md", pool)
    assert result.per_chapter["vague_noun_density"] > 0


def test_concreteness_metadata():
    lens = ConcretenessLens()
    assert lens.name == "concreteness"
    assert lens.requires_providers == ["spacy"]
    assert lens.consumes_lenses == []
