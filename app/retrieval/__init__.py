"""Evidence retrieval contracts and implementations."""

from app.retrieval.base import EvidenceRetriever
from app.retrieval.bm25 import BM25NormalizedRetriever, BM25Retriever, BM25StopwordRetriever
from app.retrieval.embedding import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingRetriever,
    OpenAIEmbeddingProvider,
)
from app.retrieval.factory import build_evidence_retriever, configured_retriever_name
from app.retrieval.hybrid import ReciprocalRankFusionRetriever
from app.retrieval.models import RetrievedEvidence

__all__ = [
    "BM25Retriever",
    "BM25NormalizedRetriever",
    "BM25StopwordRetriever",
    "build_evidence_retriever",
    "configured_retriever_name",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingRetriever",
    "EvidenceRetriever",
    "OpenAIEmbeddingProvider",
    "ReciprocalRankFusionRetriever",
    "RetrievedEvidence",
]
