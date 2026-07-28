"""
Pluggable LLM wrapper.

Supports OpenAI (cloud) and Ollama (local, free) as configured via
LLM_PROVIDER in .env. If neither is reachable (e.g. no API key configured
and no local Ollama server running), the app falls back to a simple
extractive responder so the RAG pipeline remains demonstrable end-to-end
without any external dependency - it just won't produce fluent prose.
"""
from typing import List

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class LLMProvider:
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OpenAILLM(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

class GroqLLM(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = settings.groq_model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

class OllamaLLM(LLMProvider):
    def __init__(self):
        import requests  # local import to keep it optional
        self._requests = requests
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        response = self._requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature}},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()


class ExtractiveFallbackLLM(LLMProvider):
    """No-key / offline fallback: returns the most relevant retrieved
    sentences verbatim instead of a generated answer. Keeps the pipeline
    runnable for local testing/demo without any LLM credentials."""

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        logger.warning(
            "No LLM provider configured/reachable - returning extractive "
            "fallback response. Set OPENAI_API_KEY or run Ollama for real answers."
        )
        context_start = prompt.find("Context:")
        context_end = prompt.find("Question:")
        context = prompt[context_start:context_end].strip() if context_start != -1 else ""
        snippet = context[:600] if context else "No context was retrieved for this query."
        return (
            "[Fallback mode - no LLM configured] Based on the retrieved context:\n\n"
            f"{snippet}\n\n"
            "Configure OPENAI_API_KEY or a local Ollama model in .env for a "
            "generated natural-language answer."
        )


def get_llm() -> LLMProvider:
    provider = settings.llm_provider.lower()
    try:
        if provider == "openai" and settings.openai_api_key:
            return OpenAILLM()
        if provider == "ollama":
            return OllamaLLM()
        if provider == "groq" and settings.groq_api_key:
            return GroqLLM()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to initialize LLM provider '%s': %s", provider, exc)
    return ExtractiveFallbackLLM()
