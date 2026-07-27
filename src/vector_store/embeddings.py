"""Pluggable embedding provider.

Defaults to a local HuggingFace sentence-transformers model
(all-MiniLM-L6-v2) so the project runs fully offline without any paid API
key. Set EMBEDDING_PROVIDER=openai in .env to use OpenAI embeddings instead.
"""
from functools import lru_cache
from typing import List

from config.settings import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingProvider:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()

class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "nomic-embed-text"):
        import requests
        self._requests = requests
        self.base_url = settings.ollama_base_url
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        response = self._requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider()
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider()
    return HuggingFaceEmbeddingProvider(settings.embedding_model)
