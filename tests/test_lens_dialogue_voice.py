import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.dialogue_voice import DialogueVoiceLens
from prose_doctor.providers import ProviderPool


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
def test_dialogue_voice_lens_returns_result():
    lens = DialogueVoiceLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "dialogue_voice"
    assert result.per_chapter is not None
    assert "speaker_separation" in result.per_chapter
    assert "dialogue_ratio" in result.per_chapter
    assert "talking_heads_count" in result.per_chapter


def test_dialogue_voice_lens_metadata():
    lens = DialogueVoiceLens()
    assert lens.name == "dialogue_voice"
    assert "spacy" in lens.requires_providers
    assert "sentence_transformer" in lens.requires_providers
