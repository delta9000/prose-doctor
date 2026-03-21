import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.boyd_narrative_role import BoydNarrativeRoleLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
The station stood at the edge of town, between the rail yards and the
river. Morning light filtered through the high windows, casting long
shadows across the marble floor.

Then Marcus moved. He started running, turned the corner, went through
the door. Suddenly the alarm began blaring. He continued down the hall,
finally reaching the exit.

But it wasn't enough. Not even close. Although he'd tried, nothing
had worked. Despite every effort, against all odds, he still couldn't
break through.

She watched from the balcony. The coffee was cold in her hands.
'''


def test_boyd_returns_result():
    lens = BoydNarrativeRoleLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "boyd_narrative_role"
    assert result.per_chapter is not None
    assert "staging" in result.per_chapter
    assert "progression" in result.per_chapter
    assert "tension" in result.per_chapter


def test_boyd_per_paragraph():
    lens = BoydNarrativeRoleLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "staging" in result.per_paragraph
    assert len(result.per_paragraph["staging"]) > 0


def test_boyd_dominant_mode_in_raw():
    lens = BoydNarrativeRoleLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert "dominant_modes" in result.raw
    assert len(result.raw["dominant_modes"]) > 0


def test_boyd_metadata():
    lens = BoydNarrativeRoleLens()
    assert lens.name == "boyd_narrative_role"
    assert lens.requires_providers == []
    assert lens.consumes_lenses == []
