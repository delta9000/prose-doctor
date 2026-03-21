import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.emotion_arc import EmotionArcLens


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
def test_emotion_arc_lens_returns_result():
    lens = EmotionArcLens()
    pool = None  # no providers needed
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "emotion_arc"
    assert result.per_chapter is not None
    assert "flat" in result.per_chapter
    assert "dynamic_range" in result.per_chapter
    assert "mean_sentiment" in result.per_chapter
    assert "peaks" in result.per_chapter
    assert "valleys" in result.per_chapter


def test_emotion_arc_lens_metadata():
    lens = EmotionArcLens()
    assert lens.name == "emotion_arc"
    assert lens.requires_providers == []
