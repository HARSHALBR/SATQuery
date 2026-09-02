"""Workflow planning schemas for SATQuery AI.

Defines the WorkflowStep, WorkflowPlan, and WorkflowStatus used by
the constrained planner and the execution engine.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from schemas.query import Modality, TaskType


class WorkflowStatus(str, Enum):
    """Lifecycle status of a workflow plan."""

    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class WorkflowStep(BaseModel):
    """A single step in a planned workflow.

    Attributes:
        tool: Name of the tool to execute.
        parameters: Tool-specific parameters.
        depends_on: Indices (0-based) of prerequisite steps.
    """

    tool: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)


class WorkflowPlan(BaseModel):
    """A planned workflow for executing a query.

    Created by the constrained planner, consumed by the execution engine.
    """

    workflow_id: str
    task: TaskType
    steps: list[WorkflowStep] = Field(default_factory=list)
    required_modalities: list[Modality] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    fallback: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PLANNED
