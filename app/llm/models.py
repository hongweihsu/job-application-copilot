from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LLMRequirementDecision(BaseModel):
    status: Literal["supported", "partial", "missing"]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    related_evidence_ids: list[str]
    contradictory_evidence_ids: list[str]
    matched_terms: list[str]
    explanation: str
    recommendation: str

    @model_validator(mode="after")
    def validate_grounding(self):
        groups = [
            self.evidence_ids,
            self.related_evidence_ids,
            self.contradictory_evidence_ids,
        ]
        if sum(len(group) for group in groups) != len(set().union(*map(set, groups))):
            raise ValueError("Evidence IDs cannot appear in more than one relationship group")
        if self.status == "missing" and self.evidence_ids:
            raise ValueError("A missing requirement cannot cite supporting evidence")
        if self.status == "supported" and not self.evidence_ids:
            raise ValueError("A supported decision requires supporting evidence")
        if self.status == "partial" and not any(groups):
            raise ValueError("A partial decision requires at least related evidence")
        return self
