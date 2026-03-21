import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.perplexity import PerplexityLens
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
def test_perplexity_lens_returns_result():
    lens = PerplexityLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "perplexity"
    assert result.per_chapter is not None
    assert "mean_ppl" in result.per_chapter


@pytest.mark.slow
def test_perplexity_has_per_paragraph():
    lens = PerplexityLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "perplexity" in result.per_paragraph


def test_perplexity_metadata():
    lens = PerplexityLens()
    assert lens.name == "perplexity"
    assert "gpt2" in lens.requires_providers
