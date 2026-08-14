import pytest

from app.llm.analyzer import analyze_with_llm
from app.llm.models import LLMRequirementDecision
from app.retrieval import RetrievedEvidence


class FakeProvider:
    model_name = "fake-model-v1"

    def __init__(self, decision: LLMRequirementDecision):
        self.decision = decision
        self.received_evidence_units = []

    def decide(self, requirement, evidence_units):
        self.received_evidence_units.append(evidence_units)
        return self.decision


class FakeRetriever:
    method_name = "fake"

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


def decision(**overrides) -> LLMRequirementDecision:
    values = {
        "status": "supported",
        "confidence": 0.9,
        "evidence_ids": ["resume-s1"],
        "matched_terms": ["leadership"],
        "explanation": "The evidence directly describes leading engineers.",
        "recommendation": "Keep the evidence and quantify the delivery outcome.",
    }
    return LLMRequirementDecision(**{**values, **overrides})


def test_llm_analysis_returns_versioned_structured_result():
    result = analyze_with_llm(
        "Led four engineers through a billing migration.",
        "Demonstrated engineering leadership experience is required.",
        FakeProvider(decision()),
        FakeRetriever(),
    )
    assert result.analyzer == "llm"
    assert result.model == "fake-model-v1"
    assert result.prompt_version == "requirement-match-v1"
    assert result.matches[0].evidence[0].text.startswith("Led four engineers")


def test_llm_analysis_rejects_unknown_citation_ids():
    provider = FakeProvider(decision(evidence_ids=["resume-s99"]))
    with pytest.raises(ValueError, match="unknown evidence IDs"):
        analyze_with_llm(
            "Led four engineers through a billing migration.",
            "Demonstrated engineering leadership experience is required.",
            provider,
            FakeRetriever(),
        )


def test_llm_analysis_sends_only_bm25_results_to_provider():
    provider = FakeProvider(decision())

    analyze_with_llm(
        "Built production Python APIs for internal users. Created React user interfaces.",
        "Python experience is required.",
        provider,
        top_k=1,
    )

    assert provider.received_evidence_units == [
        [("resume-s1", "Built production Python APIs for internal users.")]
    ]


def test_llm_analysis_rejects_citation_outside_retrieved_evidence():
    provider = FakeProvider(decision(evidence_ids=["resume-s2"]))

    with pytest.raises(ValueError, match="unknown evidence IDs"):
        analyze_with_llm(
            "Built production Python APIs for internal users. Created React user interfaces.",
            "Python experience is required.",
            provider,
            top_k=1,
        )


def test_llm_schema_rejects_missing_decision_with_evidence():
    with pytest.raises(ValueError, match="missing requirement cannot cite evidence"):
        decision(status="missing", evidence_ids=["resume-s1"])


def test_llm_schema_requires_evidence_for_positive_decisions():
    with pytest.raises(ValueError, match="require evidence"):
        decision(status="partial", evidence_ids=[])
