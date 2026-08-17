"""Evidence retrieval contracts and implementations."""

from app.retrieval.base import EvidenceRetriever
from app.retrieval.bm25 import BM25NormalizedRetriever, BM25Retriever, BM25StopwordRetriever
from app.retrieval.embedding import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingRetriever,
    OpenAIEmbeddingProvider,
)
from app.retrieval.hybrid import ReciprocalRankFusionRetriever
from app.retrieval.models import RetrievedEvidence

__all__ = [
    "BM25Retriever",
    "BM25NormalizedRetriever",
    "BM25StopwordRetriever",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingRetriever",
    "EvidenceRetriever",
    "OpenAIEmbeddingProvider",
    "ReciprocalRankFusionRetriever",
    "RetrievedEvidence",
]
