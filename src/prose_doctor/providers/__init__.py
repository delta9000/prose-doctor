"""Shared resource providers -- lazy-loaded, instantiated once per session.

Replaces ml/__init__.py (require_ml guard) and ml/models.py (ModelManager).
"""
from __future__ import annotations

from typing import Any

ML_AVAILABLE = False
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    ML_AVAILABLE = True
except ImportError:
    pass


def require_ml() -> None:
    """Raise ImportError with install instructions if ML deps are missing."""
    if not ML_AVAILABLE:
        raise ImportError(
            "ML features require extra dependencies. Install with:\n"
            "  uv pip install -e '.[ml]'"
        )


class ProviderPool:
    """Lazy-loading shared resource pool. Replaces ModelManager."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._factories: dict[str, type] = {
            "spacy": _lazy_import_spacy,
            "gpt2": _lazy_import_gpt2,
            "sentence_transformer": _lazy_import_st,
            "llm": _lazy_import_llm,
        }

    def get(self, name: str) -> Any:
        if name not in self._factories:
            raise KeyError(f"Unknown provider: {name}")
        if name not in self._cache:
            self._cache[name] = self._factories[name]()
        return self._cache[name]

    def available(self) -> list[str]:
        return list(self._factories.keys())

    @property
    def spacy(self):
        return self.get("spacy")

    @property
    def gpt2(self):
        return self.get("gpt2")

    @property
    def sentence_transformer(self):
        return self.get("sentence_transformer")

    @property
    def llm(self):
        return self.get("llm")

    @property
    def device(self):
        if "device" not in self._cache:
            import torch
            self._cache["device"] = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._cache["device"]


def _lazy_import_spacy():
    from prose_doctor.providers.spacy import load_spacy
    return load_spacy()


def _lazy_import_gpt2():
    from prose_doctor.providers.gpt2 import load_gpt2
    return load_gpt2()


def _lazy_import_st():
    from prose_doctor.providers.sentence_transformer import load_sentence_transformer
    return load_sentence_transformer()


def _lazy_import_llm():
    from prose_doctor.providers.llm import load_llm_client
    return load_llm_client()
