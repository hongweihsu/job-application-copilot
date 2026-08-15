from evaluation.run_bm25_normalization_experiment import (
    evaluate_normalization_experiment,
    markdown_report,
)


def test_normalization_experiment_compares_dev_cases_only():
    report = evaluate_normalization_experiment()

    assert report["evaluated_split"] == "dev"
    assert report["profiles"] == 7
    assert report["cases"] == 28
    assert set(report["methods"]) == {"bm25_raw", "bm25_normalized"}


def test_normalization_experiment_has_pre_registered_decision_rule():
    report = evaluate_normalization_experiment()

    assert report["decision"]["status"] in {
        "candidate_for_validation",
        "rejected_as_standalone",
    }
    assert (
        report["decision"]["minimum_recall_at_5"]
        == report["methods"]["bm25_raw"]["metrics"]["recall_at_5"]
    )


def test_normalization_experiment_markdown_preserves_split_boundary():
    output = markdown_report(evaluate_normalization_experiment())

    assert "BM25 Word Normalization Experiment" in output
    assert "Validation and test were not evaluated" in output
