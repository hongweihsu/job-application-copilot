import math
import re
from collections import Counter

from app.retrieval.models import RetrievedEvidence

TOKEN_PATTERN = re.compile(r"c\+\+|c#|[a-z0-9]+(?:[./+#-][a-z0-9]+)*")
CONSERVATIVE_QUERY_STOPWORDS = frozenset(
    {
        "a",
        "ability",
        "an",
        "and",
        "are",
        "at",
        "commercial",
        "demonstrated",
        "essential",
        "experience",
        "for",
        "hands-on",
        "have",
        "in",
        "is",
        "must",
        "of",
        "or",
        "professional",
        "recent",
        "required",
        "skills",
        "strong",
        "the",
        "to",
        "using",
        "with",
    }
)
WORD_NORMALIZATION_ALIASES = {
    "administered": "administer",
    "administering": "administer",
    "administration": "administer",
    "applications": "application",
    "built": "build",
    "building": "build",
    "builds": "build",
    "communicated": "communicate",
    "communicating": "communicate",
    "communication": "communicate",
    "deployed": "deploy",
    "deploying": "deploy",
    "deployment": "deploy",
    "deployments": "deploy",
    "engineers": "engineer",
    "engineering": "engineer",
    "incidents": "incident",
    "managed": "manage",
    "management": "manage",
    "managing": "manage",
    "models": "model",
    "monitored": "monitor",
    "monitoring": "monitor",
    "monitors": "monitor",
    "operations": "operate",
    "operated": "operate",
    "operating": "operate",
    "optimized": "optimize",
    "optimizing": "optimize",
    "optimization": "optimize",
    "pipelines": "pipeline",
    "requirements": "requirement",
    "services": "service",
    "trained": "train",
    "training": "train",
    "trains": "train",
    "workloads": "workload",
}


def _tokenize(text: str) -> list[str]:
    """Normalize text into terms used by BM25."""

    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Rank resume evidence with the Okapi BM25 scoring function."""

    method_name = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        query_stopwords: frozenset[str] | None = None,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self.query_stopwords = query_stopwords or frozenset()

    def _tokens(self, text: str) -> list[str]:
        return _tokenize(text)

    def retrieve(
        self,
        query: str,
        evidence_units: list[tuple[str, str]],
        top_k: int,
    ) -> list[RetrievedEvidence]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not evidence_units:
            return []

        query_terms = set(self._tokens(query)) - self.query_stopwords
        if not query_terms:
            return []

        tokenized_documents = [self._tokens(text) for _, text in evidence_units]
        average_document_length = sum(map(len, tokenized_documents)) / len(tokenized_documents)
        if average_document_length == 0:
            return []

        document_frequency = Counter(
            term
            for document in tokenized_documents
            for term in set(document)
            if term in query_terms
        )
        document_count = len(tokenized_documents)
        scored_documents = []

        for original_index, ((evidence_id, text), terms) in enumerate(
            zip(evidence_units, tokenized_documents, strict=True)
        ):
            term_frequency = Counter(terms)
            document_length = len(terms)
            score = 0.0

            for term in query_terms:
                frequency = term_frequency[term]
                if frequency == 0:
                    continue
                frequency_in_documents = document_frequency[term]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - frequency_in_documents + 0.5)
                    / (frequency_in_documents + 0.5)
                )
                length_normalization = self.k1 * (
                    1 - self.b + self.b * document_length / average_document_length
                )
                score += inverse_document_frequency * (
                    frequency * (self.k1 + 1) / (frequency + length_normalization)
                )

            if score > 0:
                scored_documents.append((score, original_index, evidence_id, text))

        scored_documents.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedEvidence(
                evidence_id=evidence_id,
                text=text,
                score=score,
                rank=rank,
                retrieval_method=self.method_name,
            )
            for rank, (score, _, evidence_id, text) in enumerate(scored_documents[:top_k], start=1)
        ]


class BM25StopwordRetriever(BM25Retriever):
    """BM25 variant that removes conservative boilerplate terms from queries."""

    method_name = "bm25_stopwords"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(k1=k1, b=b, query_stopwords=CONSERVATIVE_QUERY_STOPWORDS)


class BM25NormalizedRetriever(BM25Retriever):
    """BM25 variant with transparent, conservative word-form normalization."""

    method_name = "bm25_normalized"

    def _tokens(self, text: str) -> list[str]:
        return [WORD_NORMALIZATION_ALIASES.get(token, token) for token in _tokenize(text)]
