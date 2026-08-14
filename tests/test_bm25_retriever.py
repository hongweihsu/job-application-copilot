import pytest

from app.retrieval import BM25Retriever


def test_ranks_document_matching_more_query_terms_first():
    results = BM25Retriever().retrieve(
        query="Python AWS experience",
        evidence_units=[
            ("resume-s1", "Built Python services and deployed them to AWS."),
            ("resume-s2", "Built Python automation scripts."),
            ("resume-s3", "Created React user interfaces."),
        ],
        top_k=5,
    )

    assert [result.evidence_id for result in results] == ["resume-s1", "resume-s2"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score > results[1].score > 0
    assert all(result.retrieval_method == "bm25" for result in results)


def test_respects_top_k():
    results = BM25Retriever().retrieve(
        query="Python",
        evidence_units=[
            ("resume-s1", "Used Python for APIs."),
            ("resume-s2", "Used Python for data analysis."),
        ],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1


def test_excludes_documents_without_query_terms():
    results = BM25Retriever().retrieve(
        query="Kubernetes",
        evidence_units=[
            ("resume-s1", "Built React user interfaces."),
            ("resume-s2", "Created SQL reports."),
        ],
        top_k=5,
    )

    assert results == []


def test_equal_scores_preserve_original_evidence_order():
    results = BM25Retriever().retrieve(
        query="Python",
        evidence_units=[
            ("resume-s8", "Used Python for APIs."),
            ("resume-s2", "Used Python for jobs."),
        ],
        top_k=5,
    )

    assert [result.evidence_id for result in results] == ["resume-s8", "resume-s2"]


def test_tokenizer_preserves_common_technical_terms():
    results = BM25Retriever().retrieve(
        query="C++ C# CI/CD Node.js",
        evidence_units=[
            ("resume-s1", "Built C++ and C# services with CI/CD and Node.js tooling."),
            ("resume-s2", "Built C services."),
        ],
        top_k=5,
    )

    assert [result.evidence_id for result in results] == ["resume-s1"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k1": 0}, "k1 must be greater than 0"),
        ({"b": -0.1}, "b must be between 0 and 1"),
        ({"b": 1.1}, "b must be between 0 and 1"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BM25Retriever(**kwargs)


def test_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        BM25Retriever().retrieve("Python", [("resume-s1", "Used Python.")], top_k=0)


@pytest.mark.parametrize(
    ("query", "evidence_units"),
    [
        ("Python", []),
        ("...", [("resume-s1", "Used Python for APIs.")]),
        ("Python", [("resume-s1", "...")]),
    ],
)
def test_empty_search_inputs_return_no_results(query, evidence_units):
    assert BM25Retriever().retrieve(query, evidence_units, top_k=5) == []
