"""Tool registry for GeoVision.

Provides the ToolRegistry class that manages tool definitions,
evaluates applicability against a structured execution context,
and supports loading tool contracts from YAML configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from schemas.query import Modality
from schemas.tools import ToolApplicability, ToolDefinition


# ---------------------------------------------------------------------------
# Execution Context
# ---------------------------------------------------------------------------


class ToolExecutionContext(BaseModel):
    """Structured context used to evaluate tool applicability.

    Rather than passing loose modalities and bands, every applicability
    check receives a full context object that captures observations,
    temporal state, registration status, quality metrics, and the set
    of tools that have already completed.
    """

    available_modalities: list[Modality] = Field(default_factory=list)
    available_bands: list[str] = Field(default_factory=list)
    num_observations: int = 0
    has_temporal: bool = False
    registration_status: Optional[bool] = None  # None=unknown, True=ok, False=failed
    valid_pixel_fraction: Optional[float] = None
    completed_tools: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Central registry for tool definitions and applicability checks.

    Usage::

        registry = ToolRegistry()
        registry.register_tool(tool_def)
        applicable = registry.get_applicable_tools(context)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # -- Registration -------------------------------------------------------

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool definition. Raises on duplicate names."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    # -- Lookup -------------------------------------------------------------

    def get_tool(self, name: str) -> ToolDefinition:
        """Retrieve a tool by name. Raises KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return self._tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    # -- Applicability ------------------------------------------------------

    def check_applicability(
        self,
        tool: ToolDefinition,
        context: ToolExecutionContext,
    ) -> tuple[bool, Optional[str]]:
        """Evaluate whether *tool* can execute in *context*.

        Returns:
            A ``(applicable, reason)`` tuple.  When ``applicable`` is
            ``False``, *reason* explains which prerequisite failed.
        """
        app: ToolApplicability = tool.applicability

        # 1. Required modalities
        for mod in app.required_modalities:
            if mod not in context.available_modalities:
                return False, f"Required modality '{mod.value}' is not available."

        # 2. Required spectral bands
        for band in app.required_bands:
            if band not in context.available_bands:
                return False, f"Required band '{band}' is not available."

        # 3. Minimum observations
        if context.num_observations < app.min_observations:
            return (
                False,
                f"Requires at least {app.min_observations} observation(s), "
                f"got {context.num_observations}.",
            )

        # 4. Temporal requirement
        if app.requires_temporal and not context.has_temporal:
            return False, "Temporal data is required but not available."

        # 5. Registration requirement
        if app.requires_registration:
            if context.registration_status is None:
                return False, "Registration status is unknown."
            if not context.registration_status:
                return False, "Image registration has failed."

        # 6. Minimum valid-pixel fraction
        if app.min_valid_pixel_fraction is not None:
            if context.valid_pixel_fraction is None:
                return False, "Valid pixel fraction is unknown."
            if context.valid_pixel_fraction < app.min_valid_pixel_fraction:
                return (
                    False,
                    f"Valid pixel fraction {context.valid_pixel_fraction:.2f} "
                    f"is below minimum {app.min_valid_pixel_fraction:.2f}.",
                )

        # 7. Prerequisite tools
        for prereq in app.prerequisite_tools:
            if prereq not in context.completed_tools:
                return False, f"Prerequisite tool '{prereq}' has not been completed."

        return True, None

    def get_applicable_tools(
        self,
        context: ToolExecutionContext,
    ) -> list[ToolDefinition]:
        """Return all tools applicable in *context*, sorted by priority (desc)."""
        applicable: list[ToolDefinition] = []
        for tool in self._tools.values():
            ok, _ = self.check_applicability(tool, context)
            if ok:
                applicable.append(tool)
        applicable.sort(key=lambda t: t.priority, reverse=True)
        return applicable

    # -- YAML loader --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ToolRegistry:
        """Load tool definitions from a YAML configuration file."""
        registry = cls()
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)

        for tool_data in data.get("tools", []):
            app_raw = tool_data.pop("applicability", {})

            # Convert modality strings → Modality enum members
            if "required_modalities" in app_raw:
                app_raw["required_modalities"] = [
                    Modality(m) for m in app_raw["required_modalities"]
                ]

            applicability = ToolApplicability(**app_raw)
            tool_def = ToolDefinition(applicability=applicability, **tool_data)
            registry.register_tool(tool_def)

        return registry
