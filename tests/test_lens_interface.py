from prose_doctor.lenses import Lens, LensResult, LensRegistry


def test_lens_result_has_standard_fields():
    r = LensResult(lens_name="test")
    assert r.lens_name == "test"
    assert r.per_sentence is None
    assert r.per_paragraph is None
    assert r.per_scene is None
    assert r.per_chapter is None
    assert r.issues == []
    assert r.raw == {}


def test_lens_result_with_data():
    r = LensResult(
        lens_name="psychic_distance",
        per_paragraph={"pd_mean": [0.3, 0.4, 0.2]},
        per_chapter={"pd_mean": 0.3, "pd_std": 0.08},
    )
    assert len(r.per_paragraph["pd_mean"]) == 3
    assert r.per_chapter["pd_mean"] == 0.3


def test_lens_registry_register_and_get():
    registry = LensRegistry()

    class FakeLens(Lens):
        name = "fake"
        requires_providers = []
        consumes_lenses = []

        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name=self.name)

    registry.register(FakeLens())
    assert registry.get("fake") is not None
    assert registry.get("nonexistent") is None
    assert "fake" in registry.all_names()


def test_lens_registry_rejects_duplicate():
    registry = LensRegistry()

    class FakeLens(Lens):
        name = "fake"
        requires_providers = []
        consumes_lenses = []

        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name=self.name)

    registry.register(FakeLens())
    try:
        registry.register(FakeLens())
        assert False, "Should have raised"
    except ValueError:
        pass


def test_default_registry_has_all_lenses():
    from prose_doctor.lenses.defaults import default_registry
    registry = default_registry()
    expected = [
        "pacing", "emotion_arc", "foregrounding", "info_contour",
        "psychic_distance", "sensory", "dialogue_voice", "slop_classifier",
        "perplexity", "uncertainty_reduction", "boyd_narrative_role",
        "fragment_classifier", "narrative_attention",
    ]
    for name in expected:
        assert registry.get(name) is not None, f"Missing lens: {name}"
