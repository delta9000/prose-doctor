from prose_doctor.lenses import LensResult
from prose_doctor.lenses.pacing import PacingLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
"You're late," she said, crossing her arms.

"Traffic," he muttered. "And the bridge was up."

"The bridge has been up for three days."

"I know."

He dropped his bag on the floor and walked to the window. The street below
was empty except for a delivery truck idling at the corner. Rain streaked
the glass.

She watched him for a long moment. Something had changed — the way he
held his shoulders, maybe, or the careful blankness in his face. She'd
seen that look before, years ago, when his mother was dying and he
wouldn't say so.

The kettle clicked off. Neither of them moved.
'''


def test_pacing_lens_returns_lens_result():
    lens = PacingLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "pacing"
    assert result.per_chapter is not None
    assert "dialogue_ratio" in result.per_chapter


def test_pacing_lens_metadata():
    lens = PacingLens()
    assert lens.name == "pacing"
    assert lens.requires_providers == []
    assert lens.consumes_lenses == []


def test_pacing_lens_per_paragraph():
    lens = PacingLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_paragraph is not None
    assert "mode" in result.per_paragraph
    assert len(result.per_paragraph["mode"]) > 0
