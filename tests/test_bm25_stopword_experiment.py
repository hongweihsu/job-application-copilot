from evaluation.run_bm25_stopword_experiment import (
    evaluate_stopword_experiment,
    markdown_report,
)


def test_stopword_experiment_compares_dev_cases_only():
    report = evaluate_stopword_experiment()

    assert report["evaluated_split"] == "dev"
    assert report["profiles"] == 7
    assert report["cases"] == 28
    assert set(report["methods"]) == {"bm25_raw", "bm25_stopwords"}
    assert report["decision"]["status"] == "rejected_as_standalone"
    assert report["decision"]["precision_improved"] is True
    assert report["decision"]["recall_gate_passed"] is False


def test_stopword_experiment_markdown_reports_attribution():
    output = markdown_report(evaluate_stopword_experiment())

    assert "BM25 Stopword Experiment" in output
    assert "query only" in output
    assert "Validation and test were not evaluated" in output
