"""Execution-trace schemas for GeoVision.

Defines the per-step trace record and the aggregate execution trace
that forms the audit trail for every query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from schemas.tools import ToolStatus


class TraceStep(BaseModel):
    """A single step in the execution trace."""

    step: int
    tool: str
    status: ToolStatus
    duration_ms: Optional[int] = None
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    error: Optional[str] = None


class ExecutionTrace(BaseModel):
    """Complete execution trace for a workflow run."""

    trace_id: str
    workflow_id: str
    steps: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
