import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.foregrounding import ForegroundingLens
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
def test_foregrounding_lens_returns_result():
    lens = ForegroundingLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "foregrounding"
    assert result.per_chapter is not None
    assert "inversion_pct" in result.per_chapter
    assert "sl_cv" in result.per_chapter
    assert "fragment_pct" in result.per_chapter


@pytest.mark.slow
def test_foregrounding_lens_metadata():
    lens = ForegroundingLens()
    assert lens.name == "foregrounding"
    assert "spacy" in lens.requires_providers
