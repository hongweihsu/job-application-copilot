from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnalysisRequest(BaseModel):
    resume_text: str = Field(min_length=30, max_length=50_000)
    job_description: str = Field(min_length=30, max_length=30_000)
    analysis_mode: Literal["baseline", "llm"] = "baseline"


class Evidence(BaseModel):
    evidence_id: str
    text: str


class RequirementMatch(BaseModel):
    requirement: str
    status: Literal["supported", "partial", "missing"]
    confidence: float = Field(ge=0, le=1)
    matched_terms: list[str]
    evidence: list[Evidence]
    partial_evidence: list[Evidence] = Field(default_factory=list)
    related_evidence: list[Evidence] = Field(default_factory=list)
    contradictory_evidence: list[Evidence] = Field(default_factory=list)
    recommendation: str

    @model_validator(mode="after")
    def missing_requirements_cannot_have_supporting_evidence(self):
        if self.status == "missing" and (self.evidence or self.partial_evidence):
            raise ValueError(
                "Missing requirements cannot cite supporting or partial resume evidence"
            )
        return self


class AnalysisSummary(BaseModel):
    match_score: int = Field(ge=0, le=100)
    supported: int
    partial: int
    missing: int


class AnalysisResponse(BaseModel):
    summary: AnalysisSummary
    matches: list[RequirementMatch]
    disclaimer: str
    analyzer: Literal["deterministic", "llm"] = "deterministic"
    model: str | None = None
    prompt_version: str | None = None
