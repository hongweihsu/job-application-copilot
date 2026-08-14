from evaluation.run_retrieval_impact import evaluate_retrieval_impact, markdown_report


def test_retrieval_impact_compares_recall_and_prompt_tokens():
    report = evaluate_retrieval_impact()

    assert report["dataset_cases"] == 32
    assert 0 <= report["retrieval"]["bm25_recall_at_5"] <= 1
    assert (
        report["tokens"]["bm25_top_k"]["evidence_units"]
        <= report["tokens"]["full_context"]["evidence_units"]
    )
    assert (
        report["tokens"]["bm25_top_k"]["prompt_tokens"]
        <= report["tokens"]["full_context"]["prompt_tokens"]
    )


def test_retrieval_impact_markdown_states_estimate_scope():
    output = markdown_report(evaluate_retrieval_impact())

    assert "Retrieval Impact Report" in output
    assert "Recall@5" in output
    assert "reproducible estimates rather than billing data" in output
