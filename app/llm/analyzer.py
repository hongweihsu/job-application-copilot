from app.analyzer import _sentences, extract_requirements
from app.llm.models import LLMRequirementDecision
from app.llm.prompt import PROMPT_VERSION
from app.llm.provider import RequirementDecisionProvider
from app.models import AnalysisResponse, AnalysisSummary, Evidence, RequirementMatch


def _score(matches: list[RequirementMatch]) -> AnalysisSummary:
    supported = sum(match.status == "supported" for match in matches)
    partial = sum(match.status == "partial" for match in matches)
    missing = sum(match.status == "missing" for match in matches)
    denominator = max(len(matches), 1)
    return AnalysisSummary(
        match_score=round((supported + partial * 0.5) / denominator * 100),
        supported=supported,
        partial=partial,
        missing=missing,
    )


def _validated_evidence(
    decision: LLMRequirementDecision, evidence_by_id: dict[str, str]
) -> list[Evidence]:
    unknown_ids = set(decision.evidence_ids) - evidence_by_id.keys()
    if unknown_ids:
        raise ValueError(f"Model cited unknown evidence IDs: {sorted(unknown_ids)}")
    return [Evidence(evidence_id=item, text=evidence_by_id[item]) for item in decision.evidence_ids]


def analyze_with_llm(
    resume_text: str,
    job_description: str,
    provider: RequirementDecisionProvider,
) -> AnalysisResponse:
    evidence_by_id = {
        f"resume-s{index}": sentence
        for index, sentence in enumerate(_sentences(resume_text), start=1)
    }
    evidence_units = list(evidence_by_id.items())
    matches = []
    for requirement in extract_requirements(job_description):
        decision = provider.decide(requirement, evidence_units)
        matches.append(
            RequirementMatch(
                requirement=requirement,
                status=decision.status,
                confidence=decision.confidence,
                matched_terms=decision.matched_terms,
                evidence=_validated_evidence(decision, evidence_by_id),
                recommendation=decision.recommendation,
            )
        )

    return AnalysisResponse(
        summary=_score(matches),
        matches=matches,
        disclaimer=(
            "AI output can be wrong. Verify every recommendation and never add experience that "
            "you cannot support with evidence."
        ),
        analyzer="llm",
        model=provider.model_name,
        prompt_version=PROMPT_VERSION,
    )
