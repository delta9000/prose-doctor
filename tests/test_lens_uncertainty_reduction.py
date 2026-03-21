import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.uncertainty_reduction import UncertaintyReductionLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
The station had been empty for three days. Dust covered every surface —
the ticket counter, the benches, the departure board frozen at 14:32.

Marcus pushed through the turnstile. His footsteps echoed in the vaulted
ceiling, each step announcing his presence to whatever might be listening.

"Anyone here?" His voice came back flat, swallowed by the space.
Nothing moved. He pulled the map from his pocket and studied the exits.

The south corridor led to the maintenance tunnels. That was the plan.
Underground, out of sight, moving east until they cleared the perimeter.
'''


@pytest.mark.slow
def test_uncertainty_reduction_returns_result():
    lens = UncertaintyReductionLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "uncertainty_reduction"
    assert result.per_paragraph is not None
    assert "uncertainty_reduction" in result.per_paragraph
    assert result.per_chapter is not None
    assert "mean_reduction" in result.per_chapter


@pytest.mark.slow
def test_uncertainty_reduction_first_paragraph_zero():
    lens = UncertaintyReductionLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph["uncertainty_reduction"][0] == 0.0


def test_uncertainty_reduction_metadata():
    lens = UncertaintyReductionLens()
    assert lens.name == "uncertainty_reduction"
    assert "gpt2" in lens.requires_providers
