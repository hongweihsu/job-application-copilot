from collections import defaultdict

from app.retrieval.base import EvidenceRetriever
from app.retrieval.models import RetrievedEvidence


class ReciprocalRankFusionRetriever:
    """Fuse retriever rankings without mixing their incomparable raw scores."""

    method_name = "hybrid_rrf"

    def __init__(
        self,
        retrievers: list[EvidenceRetriever],
        rrf_k: int = 60,
        retrieval_depth: int = 20,
    ):
        if len(retrievers) < 2:
            raise ValueError("RRF requires at least two retrievers")
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if retrieval_depth < 1:
            raise ValueError("retrieval_depth must be at least 1")
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.retrieval_depth = retrieval_depth

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

        text_by_id = dict(evidence_units)
        original_order = {
            evidence_id: index for index, (evidence_id, _) in enumerate(evidence_units)
        }
        scores: dict[str, float] = defaultdict(float)

        depth = max(top_k, self.retrieval_depth)
        for retriever in self.retrievers:
            for result in retriever.retrieve(query, evidence_units, top_k=depth):
                expected_text = text_by_id.get(result.evidence_id)
                if expected_text is None or expected_text != result.text:
                    raise ValueError("Child retriever returned unknown or inconsistent evidence")
                scores[result.evidence_id] += 1 / (self.rrf_k + result.rank)

        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], original_order[item[0]]),
        )
        return [
            RetrievedEvidence(
                evidence_id=evidence_id,
                text=text_by_id[evidence_id],
                score=score,
                rank=rank,
                retrieval_method=self.method_name,
            )
            for rank, (evidence_id, score) in enumerate(ranked[:top_k], start=1)
        ]
