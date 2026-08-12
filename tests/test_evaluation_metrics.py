import pytest

from evaluation.metrics import (
    CaseResult,
    citation_metrics,
    classification_metrics,
    retrieval_metrics,
    safety_metrics,
)


def result(
    case_id: str,
    expected_status: str,
    predicted_status: str,
    expected_evidence_ids: list[str],
    retrieved_evidence_ids: list[str],
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        split="test",
        expected_status=expected_status,
        predicted_status=predicted_status,
        expected_evidence_ids=expected_evidence_ids,
        retrieved_evidence_ids=retrieved_evidence_ids,
        forbidden_claims=["AWS experience"],
        generated_text="Review the available evidence.",
    )


def test_classification_metrics_include_false_supported_rate():
    results = [
        result("one", "supported", "supported", ["resume-s1"], ["resume-s1"]),
        result("two", "missing", "supported", [], ["resume-s2"]),
        result("three", "partial", "partial", ["resume-s1"], ["resume-s1"]),
    ]
    metrics = classification_metrics(results)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["false_supported_rate"] == pytest.approx(1 / 2)
    assert metrics["confusion_matrix"]["missing"]["supported"] == 1


def test_retrieval_metrics_measure_ranked_gold_evidence():
    results = [
        result(
            "one",
            "supported",
            "supported",
            ["resume-s2", "resume-s4"],
            ["resume-s1", "resume-s2", "resume-s3", "resume-s4"],
        )
    ]
    metrics = retrieval_metrics(results)
    assert metrics["recall_at_1"] == 0
    assert metrics["recall_at_3"] == pytest.approx(0.5)
    assert metrics["recall_at_5"] == 1
    assert metrics["mrr"] == pytest.approx(0.5)


def test_citation_metrics_separate_precision_and_coverage():
    results = [
        result("one", "supported", "supported", ["resume-s2"], ["resume-s1", "resume-s2"]),
        result("two", "partial", "partial", ["resume-s1"], []),
    ]
    metrics = citation_metrics(results)
    assert metrics["citation_precision"] == pytest.approx(0.5)
    assert metrics["citation_coverage"] == pytest.approx(0.5)


def test_safety_metrics_find_forbidden_claims():
    case = result("one", "missing", "missing", [], [])
    unsafe = CaseResult(**{**case.__dict__, "generated_text": "Add AWS experience."})
    metrics = safety_metrics([unsafe])
    assert metrics["unsupported_claim_rate"] == 1
    assert metrics["violations"][0]["case_id"] == "one"


def test_safety_metrics_do_not_flag_negated_or_conditional_claims():
    case = result("one", "missing", "missing", [], [])
    output = "No AWS experience was found. If you have AWS experience, add real evidence."
    safe = CaseResult(**{**case.__dict__, "generated_text": output})
    assert safety_metrics([safe])["unsupported_claim_rate"] == 0
