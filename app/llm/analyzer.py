from app.analyzer import _sentences, extract_requirements
from app.llm.prompt import PROMPT_VERSION
from app.llm.provider import RequirementDecisionProvider
from app.models import AnalysisResponse, AnalysisSummary, Evidence, RequirementMatch
from app.retrieval import BM25Retriever, EvidenceRetriever


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


def _validated_evidence_ids(
    evidence_ids: list[str], evidence_by_id: dict[str, str]
) -> list[Evidence]:
    unknown_ids = set(evidence_ids) - evidence_by_id.keys()
    if unknown_ids:
        raise ValueError(f"Model cited unknown evidence IDs: {sorted(unknown_ids)}")
    return [Evidence(evidence_id=item, text=evidence_by_id[item]) for item in evidence_ids]


def analyze_with_llm(
    resume_text: str,
    job_description: str,
    provider: RequirementDecisionProvider,
    retriever: EvidenceRetriever | None = None,
    top_k: int = 5,
) -> AnalysisResponse:
    evidence_by_id = {
        f"resume-s{index}": sentence
        for index, sentence in enumerate(_sentences(resume_text), start=1)
    }
    evidence_units = list(evidence_by_id.items())
    active_retriever = retriever if retriever is not None else BM25Retriever()
    matches = []
    for requirement in extract_requirements(job_description):
        retrieved = active_retriever.retrieve(requirement, evidence_units, top_k=top_k)
        retrieved_by_id = {item.evidence_id: item.text for item in retrieved}
        decision = provider.decide(requirement, list(retrieved_by_id.items()))
        grounded_status = (
            "missing"
            if decision.status == "partial"
            and not (decision.evidence_ids or decision.partial_evidence_ids)
            else decision.status
        )
        matches.append(
            RequirementMatch(
                requirement=requirement,
                status=grounded_status,
                confidence=decision.confidence,
                matched_terms=decision.matched_terms,
                evidence=_validated_evidence_ids(decision.evidence_ids, retrieved_by_id),
                partial_evidence=_validated_evidence_ids(
                    decision.partial_evidence_ids, retrieved_by_id
                ),
                related_evidence=_validated_evidence_ids(
                    decision.related_evidence_ids, retrieved_by_id
                ),
                contradictory_evidence=_validated_evidence_ids(
                    decision.contradictory_evidence_ids, retrieved_by_id
                ),
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
