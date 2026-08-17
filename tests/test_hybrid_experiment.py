import hashlib

import pytest

from evaluation.run_hybrid_experiment import evaluate_hybrid_experiment, markdown_report


class HashEmbeddingProvider:
    model_name = "fake-hash-embedding"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(value) for value in hashlib.sha256(text.encode()).digest()[:8]] for text in texts
        ]


def test_hybrid_experiment_compares_three_methods_on_dev_only():
    report = evaluate_hybrid_experiment(HashEmbeddingProvider())

    assert report["evaluated_split"] == "dev"
    assert report["profiles"] == 7
    assert report["cases"] == 28
    assert set(report["methods"]) == {"bm25_raw", "embedding", "hybrid_rrf"}
    assert report["methods"]["hybrid_rrf"]["configuration"] == {
        "children": ["bm25", "embedding"],
        "rrf_k": 60,
        "retrieval_depth": 20,
    }


def test_hybrid_experiment_keeps_validation_and_test_unseen():
    with pytest.raises(ValueError, match="selected on dev"):
        evaluate_hybrid_experiment(HashEmbeddingProvider(), split="validation")

    output = markdown_report(evaluate_hybrid_experiment(HashEmbeddingProvider()))
    assert "Validation and test were not evaluated" in output
