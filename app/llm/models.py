from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LLMRequirementDecision(BaseModel):
    status: Literal["supported", "partial", "missing"]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    matched_terms: list[str]
    explanation: str
    recommendation: str

    @model_validator(mode="after")
    def validate_grounding(self):
        if self.status == "missing" and self.evidence_ids:
            raise ValueError("A missing requirement cannot cite evidence")
        if self.status != "missing" and not self.evidence_ids:
            raise ValueError("Supported and partial decisions require evidence")
        return self
