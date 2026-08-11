import re
from collections.abc import Iterable

from app.models import AnalysisResponse, AnalysisSummary, Evidence, RequirementMatch

SKILL_VOCABULARY = {
    "agile",
    "aws",
    "azure",
    "ci/cd",
    "communication",
    "docker",
    "fastapi",
    "gcp",
    "git",
    "github actions",
    "javascript",
    "kubernetes",
    "leadership",
    "machine learning",
    "next.js",
    "node.js",
    "postgresql",
    "python",
    "rag",
    "react",
    "rest api",
    "sql",
    "stakeholder",
    "teamwork",
    "typescript",
}

REQUIREMENT_MARKERS = (
    "experience",
    "proficient",
    "proficiency",
    "knowledge",
    "familiar",
    "required",
    "requirements",
    "skills",
    "ability",
    "understanding",
    "must",
    "should",
)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+|[•●▪]", text)
    return [part.strip(" -\t") for part in parts if len(part.strip()) >= 12]


def _terms(text: str) -> list[str]:
    lower = _normalise(text)
    return sorted(term for term in SKILL_VOCABULARY if term in lower)


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def extract_requirements(job_description: str, limit: int = 12) -> list[str]:
    sentences = _sentences(job_description)
    candidates = [
        sentence
        for sentence in sentences
        if _terms(sentence) or any(marker in sentence.lower() for marker in REQUIREMENT_MARKERS)
    ]
    if not candidates:
        candidates = sentences
    return _unique(candidates)[:limit]


def _evidence_for(
    requirement: str, resume_sentences: list[str]
) -> tuple[list[Evidence], list[str]]:
    required_terms = _terms(requirement)
    matches: list[tuple[int, int, str, list[str]]] = []
    for sentence_index, sentence in enumerate(resume_sentences, start=1):
        sentence_terms = _terms(sentence)
        overlapping = [term for term in required_terms if term in sentence_terms]
        if overlapping:
            matches.append((len(overlapping), sentence_index, sentence, overlapping))
    matches.sort(key=lambda item: (-item[0], len(item[2]), item[1]))
    evidence = [Evidence(evidence_id=f"resume-s{item[1]}", text=item[2]) for item in matches[:5]]
    matched_terms = _unique(term for item in matches for term in item[3])
    return evidence, matched_terms


def analyze(resume_text: str, job_description: str) -> AnalysisResponse:
    requirements = extract_requirements(job_description)
    resume_sentences = _sentences(resume_text)
    results: list[RequirementMatch] = []

    for requirement in requirements:
        required_terms = _terms(requirement)
        evidence, matched_terms = _evidence_for(requirement, resume_sentences)
        coverage = len(matched_terms) / max(len(required_terms), 1)

        if not evidence:
            status = "missing"
            confidence = 0.82
            recommendation = (
                "No supporting evidence was found. Add this only if you can describe a genuine "
                "example, your actions, and a measurable outcome."
            )
        elif coverage >= 0.75:
            status = "supported"
            confidence = min(0.95, 0.70 + coverage * 0.25)
            recommendation = (
                "Evidence was found. Strengthen it with scope, ownership, and a measurable result."
            )
        else:
            status = "partial"
            confidence = 0.68
            recommendation = (
                "Related evidence was found, but it does not cover the full requirement. Clarify "
                "the connection without claiming experience you do not have."
            )

        results.append(
            RequirementMatch(
                requirement=requirement,
                status=status,
                confidence=round(confidence, 2),
                matched_terms=matched_terms,
                evidence=evidence,
                recommendation=recommendation,
            )
        )

    supported = sum(result.status == "supported" for result in results)
    partial = sum(result.status == "partial" for result in results)
    missing = sum(result.status == "missing" for result in results)
    denominator = max(len(results), 1)
    score = round((supported + partial * 0.5) / denominator * 100)

    return AnalysisResponse(
        summary=AnalysisSummary(
            match_score=score,
            supported=supported,
            partial=partial,
            missing=missing,
        ),
        matches=results,
        disclaimer=(
            "This baseline uses transparent term matching. Verify every recommendation and never "
            "add skills or experience that you cannot support with evidence."
        ),
    )
