import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.sensory import SensoryLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
Rain hammered the corrugated roof. The air tasted of rust and wet concrete.
Marcus pressed his back against the wall, feeling the cold seep through
his jacket. His breath came in short, ragged bursts.

Down the corridor, something scraped. Metal on concrete. He counted
to three, then moved — low, fast, keeping to the shadows.

The coffee was bitter, almost burnt. She wrapped her hands around the mug,
letting the heat soak into her fingers. Outside, a dog barked twice
and fell silent.
'''


@pytest.mark.slow
def test_sensory_lens_returns_result():
    lens = SensoryLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "sensory"
    assert result.per_chapter is not None
    assert "dominant_modality" in result.per_chapter or "balance_ratio" in result.per_chapter


@pytest.mark.slow
def test_sensory_lens_has_per_paragraph():
    lens = SensoryLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None


@pytest.mark.slow
def test_sensory_lens_metadata():
    lens = SensoryLens()
    assert lens.name == "sensory"
    assert "sentence_transformer" not in lens.requires_providers
