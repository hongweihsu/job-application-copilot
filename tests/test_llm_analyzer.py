import pytest

from app.llm.analyzer import analyze_with_llm
from app.llm.models import LLMRequirementDecision


class FakeProvider:
    model_name = "fake-model-v1"

    def __init__(self, decision: LLMRequirementDecision):
        self.decision = decision

    def decide(self, requirement, evidence_units):
        return self.decision


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
        )


def test_llm_schema_rejects_missing_decision_with_evidence():
    with pytest.raises(ValueError, match="missing requirement cannot cite evidence"):
        decision(status="missing", evidence_ids=["resume-s1"])


def test_llm_schema_requires_evidence_for_positive_decisions():
    with pytest.raises(ValueError, match="require evidence"):
        decision(status="partial", evidence_ids=[])
