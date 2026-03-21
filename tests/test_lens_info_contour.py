import pytest
from prose_doctor.lenses import LensResult
from prose_doctor.lenses.info_contour import InfoContourLens
from prose_doctor.providers import ProviderPool


SAMPLE = '''
Rain hammered the corrugated roof. Marcus pressed his back against
the wall, feeling the cold seep through his jacket. His breath came
in short, ragged bursts.

Down the corridor, something scraped. Metal on concrete. He counted
to three, then moved — low, fast, keeping to the shadows. The flashlight
in his left hand was dead weight now, batteries drained hours ago.

She was waiting at the junction, rifle across her knees. "Took you
long enough," she said, not looking up. Her fingers worked the bolt
mechanism with practiced ease, the sound crisp in the wet air.

"They're coming from the east side," he said. "Four, maybe five." He knelt
beside her, pulling the map from his vest pocket. The paper was damp, ink
bleeding at the folds. "We go north through the service tunnel."

"And if it's flooded?"

"Then we swim."
'''


@pytest.mark.slow
def test_info_contour_returns_result():
    lens = InfoContourLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert isinstance(result, LensResult)
    assert result.lens_name == "info_contour"
    assert result.per_chapter is not None
    assert "rhythmicity" in result.per_chapter
    assert "mean_surprisal" in result.per_chapter


@pytest.mark.slow
def test_info_contour_has_per_sentence():
    lens = InfoContourLens()
    pool = ProviderPool()
    result = lens.analyze(SAMPLE, "test.md", pool)
    assert result.per_sentence is not None
    assert "surprisal" in result.per_sentence
    assert len(result.per_sentence["surprisal"]) > 0


@pytest.mark.slow
def test_info_contour_metadata():
    lens = InfoContourLens()
    assert lens.name == "info_contour"
    assert "gpt2" in lens.requires_providers
    assert "spacy" in lens.requires_providers
