import pytest

from evaluation.run_retrieval_evaluation_v2 import evaluate_retrieval_v2, markdown_report


def test_v2_dev_evaluation_uses_only_dev_profiles():
    report = evaluate_retrieval_v2()

    assert report["evaluated_split"] == "dev"
    assert report["profiles"] == 7
    assert report["cases"] == 28
    assert set(report["methods"]) == {"deterministic", "bm25"}
    for method in report["methods"].values():
        assert len(method["case_results"]) == 28
        assert all(result["case_id"].startswith("v2-dev-") for result in method["case_results"])


def test_v2_evaluation_reports_no_gold_candidate_behavior():
    report = evaluate_retrieval_v2()

    for method in report["methods"].values():
        metrics = method["metrics"]
        assert metrics["no_gold_cases"] >= 5
        assert 0 <= metrics["no_gold_candidate_rate"] <= 1


def test_v2_markdown_identifies_dev_only_baseline():
    output = markdown_report(evaluate_retrieval_v2())

    assert "Retrieval Evaluation V2" in output
    assert "Evaluated split: `dev`" in output
    assert "No-gold candidate rate" in output


def test_v2_evaluation_rejects_invalid_cutoff():
    with pytest.raises(ValueError, match="at least 5"):
        evaluate_retrieval_v2(top_k=3)
