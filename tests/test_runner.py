from prose_doctor.lenses import Lens, LensResult, LensRegistry
from prose_doctor.lenses.runner import LensRunner
from prose_doctor.providers import ProviderPool


class StubLensA(Lens):
    name = "a"
    requires_providers = []
    consumes_lenses = []

    def analyze(self, text, filename, providers, consumed=None):
        return LensResult(
            lens_name="a",
            per_chapter={"a_score": 1.0},
        )


class StubLensB(Lens):
    name = "b"
    requires_providers = []
    consumes_lenses = ["a"]

    def analyze(self, text, filename, providers, consumed=None):
        a_score = consumed["a"].per_chapter["a_score"]
        return LensResult(
            lens_name="b",
            per_chapter={"b_score": a_score * 2},
        )


def test_runner_topological_order():
    registry = LensRegistry()
    registry.register(StubLensB())
    registry.register(StubLensA())
    runner = LensRunner(registry, ProviderPool())
    results = runner.run_all("test text", "test.md")
    assert "a" in results
    assert "b" in results
    assert results["b"].per_chapter["b_score"] == 2.0


def test_runner_run_one_with_deps():
    registry = LensRegistry()
    registry.register(StubLensA())
    registry.register(StubLensB())
    runner = LensRunner(registry, ProviderPool())
    result = runner.run_one("b", "test text", "test.md")
    assert result.per_chapter["b_score"] == 2.0


def test_runner_cycle_detection():
    class CycleA(Lens):
        name = "cycle_a"
        requires_providers = []
        consumes_lenses = ["cycle_b"]
        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name="cycle_a")

    class CycleB(Lens):
        name = "cycle_b"
        requires_providers = []
        consumes_lenses = ["cycle_a"]
        def analyze(self, text, filename, providers, consumed=None):
            return LensResult(lens_name="cycle_b")

    registry = LensRegistry()
    registry.register(CycleA())
    registry.register(CycleB())
    runner = LensRunner(registry, ProviderPool())
    try:
        runner.run_all("text", "f.md")
        assert False, "Should have raised"
    except ValueError as e:
        assert "cycle" in str(e).lower()
