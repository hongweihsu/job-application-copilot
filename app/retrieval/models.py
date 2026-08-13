from pydantic import BaseModel, Field


class RetrievedEvidence(BaseModel):
    """A resume evidence unit ranked for one requirement query."""

    evidence_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float
    rank: int = Field(ge=1)
    retrieval_method: str = Field(min_length=1)
