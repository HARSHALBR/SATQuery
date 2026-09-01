"""Tests for the SATQuery AI tool registry.

Covers registration, lookup, duplicate guards, YAML loading,
and comprehensive applicability checks against structured context.
"""

from pathlib import Path

import pytest

from agents.tool_registry import ToolExecutionContext, ToolRegistry
from schemas.query import Modality
from schemas.tools import ToolApplicability, ToolDefinition


# ===================================================================
# Helpers
# ===================================================================

TOOLS_YAML = Path(__file__).resolve().parent.parent / "configs" / "tools.yaml"


def _make_tool(
    name: str = "test_tool",
    modalities: list[Modality] | None = None,
    bands: list[str] | None = None,
    min_obs: int = 1,
    temporal: bool = False,
    registration: bool = False,
    min_vpf: float | None = None,
    prereqs: list[str] | None = None,
    priority: int = 50,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Test tool: {name}",
        applicability=ToolApplicability(
            required_modalities=modalities or [],
            required_bands=bands or [],
            min_observations=min_obs,
            requires_temporal=temporal,
            requires_registration=registration,
            min_valid_pixel_fraction=min_vpf,
            prerequisite_tools=prereqs or [],
        ),
        priority=priority,
    )


def _make_context(
    modalities: list[Modality] | None = None,
    bands: list[str] | None = None,
    num_obs: int = 1,
    temporal: bool = False,
    reg_status: bool | None = None,
    vpf: float | None = None,
    completed: list[str] | None = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        available_modalities=modalities or [],
        available_bands=bands or [],
        num_observations=num_obs,
        has_temporal=temporal,
        registration_status=reg_status,
        valid_pixel_fraction=vpf,
        completed_tools=completed or [],
    )


# ===================================================================
# Registration
# ===================================================================


class TestRegistration:
    def test_register_and_retrieve(self):
        reg = ToolRegistry()
        tool = _make_tool("alpha")
        reg.register_tool(tool)
        assert reg.get_tool("alpha").name == "alpha"

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("a"))
        reg.register_tool(_make_tool("b"))
        assert len(reg.list_tools()) == 2

    def test_duplicate_registration_raises(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register_tool(_make_tool("dup"))

    def test_unavailable_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get_tool("nonexistent")


# ===================================================================
# Applicability — individual prerequisite failures
# ===================================================================


class TestApplicabilityFailures:
    """Each test targets exactly one failing prerequisite."""

    def test_missing_modality(self):
        reg = ToolRegistry()
        tool = _make_tool("ndvi", modalities=[Modality.OPTICAL])
        ctx = _make_context(modalities=[Modality.SAR])  # no optical
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "modality" in reason.lower()

    def test_missing_required_band(self):
        reg = ToolRegistry()
        tool = _make_tool("ndvi", modalities=[Modality.OPTICAL], bands=["red", "nir"])
        ctx = _make_context(
            modalities=[Modality.OPTICAL],
            bands=["red"],  # nir missing
        )
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "nir" in reason.lower()

    def test_insufficient_observations(self):
        reg = ToolRegistry()
        tool = _make_tool("change", min_obs=2)
        ctx = _make_context(num_obs=1)
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "observation" in reason.lower()

    def test_temporal_prerequisite_failure(self):
        reg = ToolRegistry()
        tool = _make_tool("delta", temporal=True)
        ctx = _make_context(temporal=False)
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "temporal" in reason.lower()

    def test_registration_unknown(self):
        reg = ToolRegistry()
        tool = _make_tool("delta", registration=True)
        ctx = _make_context(reg_status=None)  # unknown
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "registration" in reason.lower()

    def test_registration_failed(self):
        reg = ToolRegistry()
        tool = _make_tool("delta", registration=True)
        ctx = _make_context(reg_status=False)  # failed
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "registration" in reason.lower()

    def test_low_valid_pixel_fraction(self):
        reg = ToolRegistry()
        tool = _make_tool("ndvi", min_vpf=0.5)
        ctx = _make_context(vpf=0.3)
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "pixel fraction" in reason.lower()

    def test_unknown_valid_pixel_fraction(self):
        reg = ToolRegistry()
        tool = _make_tool("ndvi", min_vpf=0.5)
        ctx = _make_context(vpf=None)
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "pixel fraction" in reason.lower()

    def test_missing_prerequisite_tool(self):
        reg = ToolRegistry()
        tool = _make_tool("ndvi", prereqs=["validate_inputs"])
        ctx = _make_context(completed=[])  # prerequisite not completed
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is False
        assert "prerequisite" in reason.lower()


# ===================================================================
# Applicability — success
# ===================================================================


class TestApplicabilitySuccess:
    def test_all_conditions_met(self):
        """Tool with every condition enabled passes when context satisfies all."""
        reg = ToolRegistry()
        tool = _make_tool(
            "ndvi_full",
            modalities=[Modality.OPTICAL],
            bands=["red", "nir"],
            min_obs=2,
            temporal=True,
            registration=True,
            min_vpf=0.5,
            prereqs=["validate_inputs"],
        )
        ctx = _make_context(
            modalities=[Modality.OPTICAL],
            bands=["red", "nir", "blue"],
            num_obs=2,
            temporal=True,
            reg_status=True,
            vpf=0.85,
            completed=["validate_inputs"],
        )
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is True
        assert reason is None

    def test_minimal_tool_always_applicable(self):
        """A tool with no special requirements is applicable in any context."""
        reg = ToolRegistry()
        tool = _make_tool("simple", min_obs=0)
        ctx = _make_context()
        ok, reason = reg.check_applicability(tool, ctx)
        assert ok is True


# ===================================================================
# get_applicable_tools
# ===================================================================


class TestGetApplicableTools:
    def test_filters_by_context(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("always", priority=10, min_obs=0))
        reg.register_tool(
            _make_tool("needs_optical", modalities=[Modality.OPTICAL], priority=20)
        )
        reg.register_tool(
            _make_tool("needs_sar", modalities=[Modality.SAR], priority=30)
        )

        ctx = _make_context(modalities=[Modality.OPTICAL])
        applicable = reg.get_applicable_tools(ctx)

        names = [t.name for t in applicable]
        assert "always" in names
        assert "needs_optical" in names
        assert "needs_sar" not in names

    def test_sorted_by_priority_descending(self):
        reg = ToolRegistry()
        reg.register_tool(_make_tool("low", priority=10, min_obs=0))
        reg.register_tool(_make_tool("high", priority=90, min_obs=0))
        reg.register_tool(_make_tool("mid", priority=50, min_obs=0))

        ctx = _make_context()
        applicable = reg.get_applicable_tools(ctx)
        priorities = [t.priority for t in applicable]
        assert priorities == sorted(priorities, reverse=True)


# ===================================================================
# YAML loading
# ===================================================================


class TestYAMLLoading:
    def test_load_tools_from_yaml(self):
        reg = ToolRegistry.from_yaml(TOOLS_YAML)
        tools = reg.list_tools()
        assert len(tools) == 10

    def test_ndvi_delta_has_correct_applicability(self):
        reg = ToolRegistry.from_yaml(TOOLS_YAML)
        ndvi = reg.get_tool("ndvi_delta")
        app = ndvi.applicability
        assert Modality.OPTICAL in app.required_modalities
        assert "red" in app.required_bands
        assert "nir" in app.required_bands
        assert app.min_observations == 2
        assert app.requires_temporal is True
        assert app.requires_registration is True
        assert app.min_valid_pixel_fraction == 0.5

    def test_validate_inputs_has_no_hard_requirements(self):
        reg = ToolRegistry.from_yaml(TOOLS_YAML)
        vi = reg.get_tool("validate_inputs")
        app = vi.applicability
        assert app.required_modalities == []
        assert app.required_bands == []
        assert app.requires_temporal is False
        assert app.requires_registration is False
        assert app.prerequisite_tools == []

    def test_all_tool_names_present(self):
        reg = ToolRegistry.from_yaml(TOOLS_YAML)
        expected = {
            "validate_inputs",
            "run_rs_vlm",
            "grounding",
            "ndvi_delta",
            "ndbi_delta",
            "sar_change",
            "change_statistics",
            "area_measurement",
            "compare_evidence",
            "generate_response",
        }
        actual = {t.name for t in reg.list_tools()}
        assert actual == expected
