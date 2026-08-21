import json

import pytest

from app.llm.models import LLMRequirementDecision
from evaluation.run_evidence_relationship_experiment import (
    BASELINE_REPORT,
    evaluate_evidence_relationship_candidate,
    markdown_report,
)


class MissingProvider:
    model_name = "fake-llm"
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    elapsed_seconds = 0.0

    def decide(self, requirement, evidence_units):
        del requirement
        self.request_count += 1
        return LLMRequirementDecision(
            status="missing",
            confidence=0.9,
            evidence_ids=[],
            partial_evidence_ids=[],
            related_evidence_ids=[evidence_units[0][0]] if evidence_units else [],
            contradictory_evidence_ids=[],
            matched_terms=[],
            explanation="No direct support.",
            recommendation="Do not claim unsupported experience.",
        )


class HashEmbeddingProvider:
    model_name = "fake-embedding"
    request_count = 0
    input_tokens = 0

    def embed(self, texts):
        self.request_count += 1
        return [[float((sum(map(ord, text)) + index) % 17) for index in range(8)] for text in texts]


def test_candidate_experiment_uses_dev_and_prior_baseline_without_test():
    report = evaluate_evidence_relationship_candidate(
        MissingProvider(),
        HashEmbeddingProvider(),
        json.loads(BASELINE_REPORT.read_text()),
    )

    assert report["evaluated_split"] == "dev"
    assert report["cases"] == 28
    assert report["profiles"] == 7
    assert report["candidate"]["llm_usage"]["requests"] == 28
    assert "Test" not in markdown_report(report)


def test_candidate_experiment_rejects_non_dev_baseline():
    with pytest.raises(ValueError, match="dev split"):
        evaluate_evidence_relationship_candidate(
            MissingProvider(),
            HashEmbeddingProvider(),
            {"evaluated_split": "validation"},
        )
