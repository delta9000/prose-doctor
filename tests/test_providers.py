import pytest
from prose_doctor.providers import ProviderPool


def test_provider_pool_creates():
    pool = ProviderPool()
    assert pool is not None


def test_provider_pool_get_unknown_raises():
    pool = ProviderPool()
    with pytest.raises(KeyError):
        pool.get("nonexistent")


def test_provider_pool_knows_available():
    pool = ProviderPool()
    names = pool.available()
    assert "spacy" in names
    assert "gpt2" in names
    assert "sentence_transformer" in names
    assert "llm" in names


def test_require_ml_passes_when_installed():
    from prose_doctor.providers import require_ml
    require_ml()  # should not raise — ML deps are installed in this env
