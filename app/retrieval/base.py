from typing import Protocol

from app.retrieval.models import RetrievedEvidence


class EvidenceRetriever(Protocol):
    """Contract implemented by every evidence retrieval strategy."""

    method_name: str

    def retrieve(
        self,
        query: str,
        evidence_units: list[tuple[str, str]],
        top_k: int,
    ) -> list[RetrievedEvidence]: ...
