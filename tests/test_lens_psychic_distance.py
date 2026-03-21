import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.psychic_distance import PsychicDistanceLens, PERCEPTION_VERBS, COGNITION_VERBS, PROXIMAL_DEICTICS
from prose_doctor.providers import ProviderPool


SAMPLE = '''
Rain hammered the corrugated roof. Marcus pressed his back against
the wall, feeling the cold seep through his jacket. His breath came
in short, ragged bursts.

Down the corridor, something scraped. Metal on concrete. He counted
to three, then moved — low, fast, keeping to the shadows.

She was waiting at the junction, rifle across her knees. "Took you
long enough," she said, not looking up. Her fingers worked the bolt
with practiced ease.
'''


@pytest.mark.slow
def test_psychic_distance_returns_result():
    lens = PsychicDistanceLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "psychic_distance"
    assert result.per_chapter is not None
    assert "pd_mean" in result.per_chapter
    assert "pd_std" in result.per_chapter


@pytest.mark.slow
def test_psychic_distance_has_per_paragraph():
    lens = PsychicDistanceLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "pd_mean" in result.per_paragraph


@pytest.mark.slow
def test_psychic_distance_has_per_sentence():
    lens = PsychicDistanceLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_sentence is not None
    assert "distance" in result.per_sentence


def test_psychic_distance_constants_exported():
    assert len(PERCEPTION_VERBS) > 0
    assert len(COGNITION_VERBS) > 0
    assert len(PROXIMAL_DEICTICS) > 0


def test_psychic_distance_metadata():
    lens = PsychicDistanceLens()
    assert lens.name == "psychic_distance"
    assert "spacy" in lens.requires_providers
