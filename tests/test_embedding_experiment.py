import hashlib

from evaluation.run_embedding_experiment import evaluate_embedding_experiment, markdown_report


class HashEmbeddingProvider:
    model_name = "fake-hash-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(value) for value in hashlib.sha256(text.encode()).digest()[:8]] for text in texts
        ]


def test_embedding_experiment_uses_dev_only_and_fake_provider():
    report = evaluate_embedding_experiment(HashEmbeddingProvider())

    assert report["evaluated_split"] == "dev"
    assert report["profiles"] == 7
    assert report["cases"] == 28
    assert set(report["methods"]) == {"bm25_raw", "embedding"}
    assert report["embedding_usage"]["cached_documents"] == 80


def test_embedding_experiment_markdown_preserves_split_boundary():
    output = markdown_report(evaluate_embedding_experiment(HashEmbeddingProvider()))

    assert "Embedding Retrieval Experiment" in output
    assert "fake-hash-embedding" in output
    assert "Validation and test were not evaluated" in output
