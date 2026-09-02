"""Response-related schemas for SATQuery AI.

Defines the evidence status, comparison result, and the final response
returned to the user.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from schemas.evidence import EvidenceRecord
from schemas.trace import TraceStep


class EvidenceStatus(str, Enum):
    """Final evidence status for a query."""

    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT = "INSUFFICIENT"


class ComparisonResult(BaseModel):
    """Result of comparing a VLM claim against collected evidence."""

    status: EvidenceStatus
    reason: str
    supporting_evidence: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class FinalResponse(BaseModel):
    """Final response returned to the user."""

    trace_id: str
    task: str
    answer: str
    status: EvidenceStatus
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    execution_trace: list[TraceStep] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
