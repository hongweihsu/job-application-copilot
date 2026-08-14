from evaluation.run_retrieval_evaluation import evaluate_retrieval, markdown_report


def test_retrieval_evaluation_compares_methods_on_every_case():
    report = evaluate_retrieval()

    assert report["dataset_cases"] == 32
    assert set(report["methods"]) == {"deterministic", "bm25"}
    for method in report["methods"].values():
        assert method["splits"]["all"]["cases"] == 32
        assert method["splits"]["all"]["metrics"]["answerable_cases"] > 0


def test_retrieval_markdown_contains_comparison_metrics():
    output = markdown_report(evaluate_retrieval())

    assert "# Retrieval Evaluation" in output
    assert "Recall@5" in output
    assert "deterministic" in output
    assert "bm25" in output
