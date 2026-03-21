"""OpenAI-compatible LLM client provider."""
from __future__ import annotations


def load_llm_client():
    from openai import OpenAI
    return OpenAI(base_url="http://localhost:8081/v1", api_key="none")
