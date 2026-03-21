import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.fragment_classifier import FragmentClassifierLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
Marcus pressed his back against the wall, feeling the cold seep through
his jacket. His breath came in short, ragged bursts. The corridor
stretched ahead, dark and narrow.

Metal on concrete. A scrape. Then silence.

"Took you long enough," she said. Her fingers worked the bolt.

Something shifted in the darkness. The air felt different somehow.
A presence, maybe. Or nothing at all.

He pulled the map from his vest pocket. The paper was damp, ink
bleeding at the folds. Three exits marked in red.
'''


@pytest.mark.slow
def test_fragment_classifier_returns_result():
    lens = FragmentClassifierLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "fragment_classifier"
    assert result.per_chapter is not None
    assert "craft_count" in result.per_chapter
    assert "crutch_count" in result.per_chapter


@pytest.mark.slow
def test_fragment_classifier_per_paragraph():
    lens = FragmentClassifierLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "fragment_ratio" in result.per_paragraph


def test_fragment_classifier_metadata():
    lens = FragmentClassifierLens()
    assert lens.name == "fragment_classifier"
    assert "spacy" in lens.requires_providers
