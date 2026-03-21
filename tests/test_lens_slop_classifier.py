import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.slop_classifier import SlopClassifierLens


SAMPLE = '''
"You're late," she said, crossing her arms.

"Traffic," he muttered. "And the bridge was up."

He dropped his bag on the floor and walked to the window. The street below
was empty except for a delivery truck idling at the corner. Rain streaked
the glass. Something about the light made him think of that summer in
Portland, the way everything had gone soft and grey.

She watched him for a long moment. Something had changed — the way he
held his shoulders, maybe, or the careful blankness in his face.

The kettle clicked off. Neither of them moved.
'''


@pytest.mark.slow
def test_slop_classifier_lens_returns_result():
    lens = SlopClassifierLens()
    pool = None  # no providers needed
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "slop_classifier"
    assert result.per_chapter is not None
    assert "flagged_pct" in result.per_chapter
    assert "mean_slop" in result.per_chapter
    assert result.per_paragraph is not None
    assert "slop_prob" in result.per_paragraph


def test_slop_classifier_lens_metadata():
    lens = SlopClassifierLens()
    assert lens.name == "slop_classifier"
    assert lens.requires_providers == []
