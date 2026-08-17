import os

from app.retrieval.base import EvidenceRetriever
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embedding import EmbeddingConfig, EmbeddingRetriever, OpenAIEmbeddingProvider

SUPPORTED_RETRIEVERS = frozenset({"bm25", "embedding"})


def configured_retriever_name() -> str:
    name = os.getenv("LLM_RETRIEVER", "bm25").strip().lower()
    if name not in SUPPORTED_RETRIEVERS:
        choices = ", ".join(sorted(SUPPORTED_RETRIEVERS))
        raise ValueError(f"LLM_RETRIEVER must be one of: {choices}")
    return name


def build_evidence_retriever(name: str | None = None) -> EvidenceRetriever:
    selected = name.strip().lower() if name is not None else configured_retriever_name()
    if selected == "bm25":
        return BM25Retriever()
    if selected == "embedding":
        provider = OpenAIEmbeddingProvider(EmbeddingConfig.from_environment())
        return EmbeddingRetriever(provider)
    choices = ", ".join(sorted(SUPPORTED_RETRIEVERS))
    raise ValueError(f"Retriever must be one of: {choices}")
