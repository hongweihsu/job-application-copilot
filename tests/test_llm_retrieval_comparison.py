from app.llm.models import LLMRequirementDecision
from app.retrieval import RetrievedEvidence
from evaluation.run_llm_retrieval_comparison import evaluate_llm_pipeline


class FirstEvidenceRetriever:
    method_name = "first"

    def retrieve(self, query, evidence_units, top_k):
        del query
        return [
            RetrievedEvidence(
                evidence_id=evidence_id,
                text=text,
                score=1.0,
                rank=rank,
                retrieval_method=self.method_name,
            )
            for rank, (evidence_id, text) in enumerate(evidence_units[:top_k], start=1)
        ]


class MissingProvider:
    model_name = "fake-model"
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    elapsed_seconds = 0.0

    def decide(self, requirement, evidence_units):
        del requirement, evidence_units
        self.request_count += 1
        return LLMRequirementDecision(
            status="missing",
            confidence=0.9,
            evidence_ids=[],
            related_evidence_ids=[],
            contradictory_evidence_ids=[],
            matched_terms=[],
            explanation="No direct evidence.",
            recommendation="Do not add unsupported experience.",
        )


def test_llm_pipeline_evaluation_records_quality_usage_and_cases():
    cases = [
        {
            "id": "dev-case",
            "requirement": "Python experience required.",
            "expected_status": "missing",
            "expected_evidence_ids": [],
            "forbidden_claims": ["Python experience"],
            "resume_evidence": [{"id": "resume-s1", "text": "Built React applications."}],
        }
    ]
    provider = MissingProvider()

    result = evaluate_llm_pipeline(cases, provider, FirstEvidenceRetriever())

    assert result["metrics"]["classification"]["accuracy"] == 1.0
    assert result["metrics"]["safety"]["unsupported_claim_rate"] == 0.0
    assert result["llm_usage"]["requests"] == 1
    assert result["case_results"][0]["case_id"] == "dev-case"
    assert result["case_results"][0]["related_evidence_ids"] == []
