import pytest

from app.retrieval import EmbeddingRetriever


class FakeEmbeddingProvider:
    model_name = "fake-embedding-v1"

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


def test_embedding_retriever_ranks_by_cosine_similarity():
    provider = FakeEmbeddingProvider(
        {
            "Python API experience": [1.0, 0.0],
            "Built Python services.": [0.9, 0.1],
            "Created React interfaces.": [0.0, 1.0],
        }
    )
    results = EmbeddingRetriever(provider).retrieve(
        "Python API experience",
        [
            ("resume-s1", "Built Python services."),
            ("resume-s2", "Created React interfaces."),
        ],
        top_k=2,
    )

    assert [result.evidence_id for result in results] == ["resume-s1", "resume-s2"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score > results[1].score
    assert all(result.retrieval_method == "embedding" for result in results)


def test_embedding_retriever_caches_repeated_document_vectors():
    provider = FakeEmbeddingProvider(
        {
            "Python": [1.0, 0.0],
            "AWS": [0.0, 1.0],
            "Built Python services.": [1.0, 0.0],
            "Deployed to AWS.": [0.0, 1.0],
        }
    )
    retriever = EmbeddingRetriever(provider)
    evidence = [
        ("resume-s1", "Built Python services."),
        ("resume-s2", "Deployed to AWS."),
    ]

    retriever.retrieve("Python", evidence, top_k=1)
    retriever.retrieve("AWS", evidence, top_k=1)

    assert provider.calls == [
        ["Python", "Built Python services.", "Deployed to AWS."],
        ["AWS"],
    ]
    assert retriever.cached_documents == 2


def test_embedding_retriever_rejects_invalid_provider_output():
    provider = FakeEmbeddingProvider({"query": [1.0], "document": [1.0]})
    provider.embed = lambda texts: [[1.0]]

    with pytest.raises(ValueError, match="unexpected number"):
        EmbeddingRetriever(provider).retrieve("query", [("resume-s1", "document")], top_k=1)


def test_embedding_retriever_rejects_mismatched_dimensions():
    provider = FakeEmbeddingProvider({"query": [1.0, 0.0], "document": [1.0]})

    with pytest.raises(ValueError, match="equal dimensions"):
        EmbeddingRetriever(provider).retrieve("query", [("resume-s1", "document")], top_k=1)


def test_embedding_retriever_handles_empty_inputs_and_invalid_top_k():
    provider = FakeEmbeddingProvider({})
    retriever = EmbeddingRetriever(provider)

    assert retriever.retrieve("query", [], top_k=5) == []
    assert retriever.retrieve("", [("resume-s1", "document")], top_k=5) == []
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        retriever.retrieve("query", [("resume-s1", "document")], top_k=0)
