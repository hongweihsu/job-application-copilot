import pytest

from app.retrieval import ReciprocalRankFusionRetriever, RetrievedEvidence


class RankedFakeRetriever:
    def __init__(self, method_name: str, evidence_ids: list[str]):
        self.method_name = method_name
        self.evidence_ids = evidence_ids
        self.requested_top_k = None

    def retrieve(self, query, evidence_units, top_k):
        del query
        self.requested_top_k = top_k
        text_by_id = dict(evidence_units)
        return [
            RetrievedEvidence(
                evidence_id=evidence_id,
                text=text_by_id[evidence_id],
                score=float(len(self.evidence_ids) - rank),
                rank=rank,
                retrieval_method=self.method_name,
            )
            for rank, evidence_id in enumerate(self.evidence_ids[:top_k], start=1)
        ]


def test_rrf_promotes_evidence_found_by_both_retrievers():
    lexical = RankedFakeRetriever("lexical", ["resume-s1", "resume-s2"])
    semantic = RankedFakeRetriever("semantic", ["resume-s3", "resume-s2"])
    retriever = ReciprocalRankFusionRetriever([lexical, semantic], rrf_k=60, retrieval_depth=10)

    results = retriever.retrieve(
        "Python experience",
        [
            ("resume-s1", "Python keyword match"),
            ("resume-s2", "Evidence found by both"),
            ("resume-s3", "Semantic match"),
        ],
        top_k=3,
    )

    assert [result.evidence_id for result in results] == [
        "resume-s2",
        "resume-s1",
        "resume-s3",
    ]
    assert [result.rank for result in results] == [1, 2, 3]
    assert all(result.retrieval_method == "hybrid_rrf" for result in results)
    assert lexical.requested_top_k == semantic.requested_top_k == 10


def test_rrf_uses_original_evidence_order_to_break_equal_scores():
    first = RankedFakeRetriever("first", ["resume-s2"])
    second = RankedFakeRetriever("second", ["resume-s1"])

    results = ReciprocalRankFusionRetriever([first, second]).retrieve(
        "query",
        [("resume-s1", "first"), ("resume-s2", "second")],
        top_k=2,
    )

    assert [result.evidence_id for result in results] == ["resume-s1", "resume-s2"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"retrievers": [RankedFakeRetriever("one", [])]}, "at least two"),
        (
            {
                "retrievers": [RankedFakeRetriever("one", []), RankedFakeRetriever("two", [])],
                "rrf_k": 0,
            },
            "rrf_k",
        ),
        (
            {
                "retrievers": [RankedFakeRetriever("one", []), RankedFakeRetriever("two", [])],
                "retrieval_depth": 0,
            },
            "retrieval_depth",
        ),
    ],
)
def test_rrf_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReciprocalRankFusionRetriever(**kwargs)


def test_rrf_handles_empty_evidence_and_rejects_invalid_top_k():
    retriever = ReciprocalRankFusionRetriever(
        [RankedFakeRetriever("one", []), RankedFakeRetriever("two", [])]
    )

    assert retriever.retrieve("query", [], top_k=5) == []
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("query", [("resume-s1", "text")], top_k=0)
