import math
import os
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI

from app.retrieval.models import RetrievedEvidence

load_dotenv()


class EmbeddingProvider(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str
    model: str = "text-embedding-3-small"
    timeout_seconds: float = 30
    max_retries: int = 2

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        )


class OpenAIEmbeddingProvider:
    def __init__(self, config: EmbeddingConfig):
        self.model_name = config.model
        self.request_count = 0
        self.input_tokens = 0
        self._client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model_name, input=texts)
        self.request_count += 1
        self.input_tokens += response.usage.prompt_tokens
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Embedding vectors must be non-empty and have equal dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


class EmbeddingRetriever:
    method_name = "embedding"

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider
        self._document_cache: dict[str, list[float]] = {}

    @property
    def cached_documents(self) -> int:
        return len(self._document_cache)

    def retrieve(
        self,
        query: str,
        evidence_units: list[tuple[str, str]],
        top_k: int,
    ) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query.strip() or not evidence_units:
            return []

        uncached_texts = list(
            dict.fromkeys(text for _, text in evidence_units if text not in self._document_cache)
        )
        vectors = self.provider.embed([query, *uncached_texts])
        if len(vectors) != len(uncached_texts) + 1:
            raise ValueError("Embedding provider returned an unexpected number of vectors")

        query_vector = vectors[0]
        self._document_cache.update(zip(uncached_texts, vectors[1:], strict=True))
        scored = [
            (
                _cosine_similarity(query_vector, self._document_cache[text]),
                index,
                evidence_id,
                text,
            )
            for index, (evidence_id, text) in enumerate(evidence_units)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        return [
            RetrievedEvidence(
                evidence_id=evidence_id,
                text=text,
                score=score,
                rank=rank,
                retrieval_method=self.method_name,
            )
            for rank, (score, _, evidence_id, text) in enumerate(scored[:top_k], start=1)
        ]
