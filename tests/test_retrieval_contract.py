from app.retrieval import EvidenceRetriever, RetrievedEvidence


class FakeRetriever:
    method_name = "fake"

    def retrieve(
        self,
        query: str,
        evidence_units: list[tuple[str, str]],
        top_k: int,
    ) -> list[RetrievedEvidence]:
        del query
        return [
            RetrievedEvidence(
                evidence_id=evidence_id,
                text=text,
                score=1.0,
                rank=rank,
                retrieval_method=self.method_name,
            )
            for rank, (evidence_id, text) in enumerate(evidence_units[:top_k], start=1)
        ]


def use_retriever(retriever: EvidenceRetriever) -> list[RetrievedEvidence]:
    return retriever.retrieve(
        query="Python API experience",
        evidence_units=[
            ("resume-s1", "Built Python APIs with FastAPI."),
            ("resume-s2", "Created React user interfaces."),
        ],
        top_k=1,
    )


def test_retriever_contract_returns_ranked_evidence():
    results = use_retriever(FakeRetriever())

    assert results == [
        RetrievedEvidence(
            evidence_id="resume-s1",
            text="Built Python APIs with FastAPI.",
            score=1.0,
            rank=1,
            retrieval_method="fake",
        )
    ]
