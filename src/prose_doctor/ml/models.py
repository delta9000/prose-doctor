"""Shared ModelManager singleton for lazy-loading ML models."""

from __future__ import annotations

import sys
from pathlib import Path


class ModelManager:
    """Singleton that loads spacy/sentence-transformers/GPT-2 once.

    All ML analyzers receive this as a parameter to avoid redundant loading.
    """

    _instance: ModelManager | None = None

    def __new__(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._spacy_nlp = None
            cls._instance._st_model = None
            cls._instance._gpt2_model = None
            cls._instance._gpt2_tokenizer = None
            cls._instance._device = None
        return cls._instance

    @property
    def device(self) -> str:
        """Detect and cache CUDA availability."""
        if self._device is None:
            import torch

            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
        return self._device

    @property
    def spacy(self):
        """Lazy-load spaCy with auto-download."""
        if self._spacy_nlp is None:
            import spacy

            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Try importing the model package directly (works with pip/uv installs)
                try:
                    import en_core_web_sm

                    self._spacy_nlp = en_core_web_sm.load()
                except ImportError:
                    raise OSError(
                        "spaCy model 'en_core_web_sm' not found.\n"
                        "Install with: pip install prose-doctor[ml]\n"
                        "Or manually: python -m spacy download en_core_web_sm"
                    )
        return self._spacy_nlp

    @property
    def sentence_transformer(self):
        """Lazy-load sentence-transformers model."""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._st_model

    @property
    def gpt2(self):
        """Lazy-load GPT-2 model and tokenizer. Returns (model, tokenizer)."""
        if self._gpt2_model is None:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            self._gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self._gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self._gpt2_model.eval()

            if self.device == "cuda":
                try:
                    self._gpt2_model = self._gpt2_model.cuda()
                except Exception:
                    print("CUDA OOM for GPT-2 -- falling back to CPU", file=sys.stderr)
                    self._device = "cpu"

        return self._gpt2_model, self._gpt2_tokenizer
