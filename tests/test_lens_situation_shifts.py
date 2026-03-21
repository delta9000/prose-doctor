import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.situation_shifts import SituationShiftsLens
from prose_doctor.providers import ProviderPool


STABLE_SAMPLE = '''
Marcus crouched behind the counter. His breath came in short bursts.
The flashlight beam swept across the floor in front of him.

He waited until the light passed, then moved. His knees ached from
the cold tile. Three meters to the back door.

Marcus reached the door and tested the handle. Locked. He pulled
the pick set from his vest pocket and went to work.
'''

SHIFTING_SAMPLE = '''
Marcus crouched behind the counter in the pharmacy. His breath came
in short bursts.

Three days earlier, Elena had stood in the same spot. The shelves
had still been full then, the lights still working.

In the basement of City Hall, Alderman Voss studied the evacuation
map. His aide hovered behind him, nervous.

The next morning brought rain. Marcus woke in the culvert where
he'd made camp, every joint stiff.
'''


@pytest.mark.slow
def test_shifts_returns_result():
    lens = SituationShiftsLens()
    pool = ProviderPool()
    result = lens.analyze(STABLE_SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "situation_shifts"
    assert result.per_chapter is not None


@pytest.mark.slow
def test_shifting_text_has_more_shifts():
    lens = SituationShiftsLens()
    pool = ProviderPool()
    stable = lens.analyze(STABLE_SAMPLE, "stable.md", pool)
    shifting = lens.analyze(SHIFTING_SAMPLE, "shifting.md", pool)
    assert shifting.per_chapter["total_shifts"] > stable.per_chapter["total_shifts"]


@pytest.mark.slow
def test_shifts_has_per_paragraph():
    lens = SituationShiftsLens()
    pool = ProviderPool()
    result = lens.analyze(SHIFTING_SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert any(k in result.per_paragraph for k in ["time_shift", "space_shift", "actor_shift"])


def test_shifts_metadata():
    lens = SituationShiftsLens()
    assert lens.name == "situation_shifts"
    assert "spacy" in lens.requires_providers
