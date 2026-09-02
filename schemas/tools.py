"""Tool-related schemas for SATQuery AI.

Defines the tool contract, machine-readable applicability conditions,
and the result model returned after executing a tool.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from schemas.query import Modality


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolStatus(str, Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


class ToolApplicability(BaseModel):
    """Machine-readable applicability conditions for a tool.

    Every field maps to a concrete, testable prerequisite that the registry
    can evaluate against a ToolExecutionContext.
    """

    required_modalities: list[Modality] = Field(default_factory=list)
    required_bands: list[str] = Field(default_factory=list)
    min_observations: int = 1
    requires_temporal: bool = False
    requires_registration: bool = False
    min_valid_pixel_fraction: Optional[float] = None
    prerequisite_tools: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------


class ToolDefinition(BaseModel):
    """Full contract for a registered tool.

    Attributes:
        name: Unique tool identifier.
        description: Human-readable purpose.
        input_schema: JSON-like description of expected inputs.
        output_schema: JSON-like description of produced outputs.
        applicability: Machine-readable conditions for when this tool can run.
        priority: Higher-priority tools are preferred during selection.
        version: Semantic version of the tool implementation.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    applicability: ToolApplicability = Field(default_factory=ToolApplicability)
    priority: int = 0
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Tool Result
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """Result of executing a single tool."""

    tool: str
    status: ToolStatus
    output: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
