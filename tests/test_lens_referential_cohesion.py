import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.referential_cohesion import ReferentialCohesionLens
from prose_doctor.providers import ProviderPool


COHERENT_SAMPLE = '''
Marcus pressed his back against the wall. His fingers found the crack
in the mortar. He counted to three, then moved.

Marcus reached the junction. The corridor split into two passages.
He chose the left one, where the air felt cooler.

The passage narrowed until Marcus had to turn sideways. His pack
scraped the concrete. Ahead, a faint light marked the exit.
'''

INCOHERENT_SAMPLE = '''
Marcus pressed his back against the wall. The government had been
overthrown. Several species of birds migrate annually.

A philosophy professor once noted the significance of the occasion.
The submarine dove to three hundred meters. Rain fell on the
abandoned parking lot.

She picked up the telephone. The economic indicators suggested
otherwise. Mountains covered the northern border.
'''


@pytest.mark.slow
def test_cohesion_returns_result():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    result = lens.analyze(COHERENT_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "referential_cohesion"
    assert result.per_chapter is not None
    assert "coherence_score" in result.per_chapter


@pytest.mark.slow
def test_coherent_scores_higher():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    coherent = lens.analyze(COHERENT_SAMPLE, "good.md", pool)
    incoherent = lens.analyze(INCOHERENT_SAMPLE, "bad.md", pool)
    assert coherent.per_chapter["coherence_score"] > incoherent.per_chapter["coherence_score"]


@pytest.mark.slow
def test_cohesion_has_per_paragraph():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    result = lens.analyze(COHERENT_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "entity_continuity" in result.per_paragraph


@pytest.mark.slow
def test_cohesion_detects_subject_churn():
    lens = ReferentialCohesionLens()
    pool = ProviderPool()
    result = lens.analyze(INCOHERENT_SAMPLE, "bad.md", pool)
    assert result.per_chapter.get("subject_churn", 0) > 0


def test_cohesion_metadata():
    lens = ReferentialCohesionLens()
    assert lens.name == "referential_cohesion"
    assert "spacy" in lens.requires_providers
