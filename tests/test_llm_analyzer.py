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
        "related_evidence_ids": [],
        "contradictory_evidence_ids": [],
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
    assert result.prompt_version == "requirement-match-v2-evidence-relationships"
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
    with pytest.raises(ValueError, match="missing requirement cannot cite supporting evidence"):
        decision(status="missing", evidence_ids=["resume-s1"])


def test_llm_schema_requires_supporting_evidence_for_supported_decisions():
    with pytest.raises(ValueError, match="requires supporting evidence"):
        decision(status="supported", evidence_ids=[])


def test_llm_analysis_downgrades_ungrounded_partial_to_missing():
    result = analyze_with_llm(
        "Evaluated React Native in a hackathon without shipping it.",
        "Production React Native application experience is required.",
        FakeProvider(
            decision(
                status="partial",
                evidence_ids=[],
                related_evidence_ids=["resume-s1"],
            )
        ),
        FakeRetriever(),
    )

    assert result.matches[0].status == "missing"
    assert result.matches[0].evidence == []
    assert result.matches[0].related_evidence[0].evidence_id == "resume-s1"


def test_llm_schema_keeps_related_and_contradictory_evidence_out_of_supporting_citations():
    result = analyze_with_llm(
        "Reviewed Kubernetes dashboards but did not administer production clusters.",
        "Production Kubernetes administration experience is required.",
        FakeProvider(
            decision(
                status="missing",
                evidence_ids=[],
                matched_terms=["kubernetes"],
                related_evidence_ids=[],
                contradictory_evidence_ids=["resume-s1"],
            )
        ),
        FakeRetriever(),
    )

    assert result.matches[0].evidence == []
    assert result.matches[0].contradictory_evidence[0].evidence_id == "resume-s1"


def test_llm_schema_rejects_overlapping_relationship_groups():
    with pytest.raises(ValueError, match="more than one relationship"):
        decision(related_evidence_ids=["resume-s1"])


def test_prompt_treats_named_technology_as_a_material_constraint():
    from app.llm.prompt import SYSTEM_PROMPT

    normalized_prompt = " ".join(SYSTEM_PROMPT.split())
    assert "named language, framework, platform" in normalized_prompt
    assert "native Android instead of React Native" in normalized_prompt
