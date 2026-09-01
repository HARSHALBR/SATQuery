"""Tests for SATQuery AI Pydantic schemas.

Covers construction, validation, serialization, enum correctness,
and rejection of invalid data.
"""

from datetime import date, datetime, timezone

import pytest

from schemas.query import (
    ImageMetadata,
    Modality,
    ObservationInput,
    ObservationRole,
    ParsedQuery,
    QueryInput,
    TaskType,
)
from schemas.tools import (
    ToolApplicability,
    ToolDefinition,
    ToolResult,
    ToolStatus,
)
from schemas.evidence import EvidenceRecord, Provenance, QualityReport
from schemas.trace import ExecutionTrace, TraceStep
from schemas.workflow import WorkflowPlan, WorkflowStatus, WorkflowStep
from schemas.response import ComparisonResult, EvidenceStatus, FinalResponse


# ===================================================================
# TaskType enum
# ===================================================================


class TestTaskType:
    def test_all_task_types_exist(self):
        expected = {
            "single_image_vqa",
            "captioning",
            "grounding",
            "bi_temporal_change",
            "vegetation_change",
            "built_up_change",
            "optical_sar_cross_check",
            "spatial_measurement",
            "insufficient_capability",
        }
        actual = {t.value for t in TaskType}
        assert actual == expected

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError):
            TaskType("nonexistent_task")


# ===================================================================
# Modality enum
# ===================================================================


class TestModality:
    def test_all_modalities_exist(self):
        expected = {"optical", "sar", "multispectral"}
        actual = {m.value for m in Modality}
        assert actual == expected

    def test_invalid_modality_raises(self):
        with pytest.raises(ValueError):
            Modality("radar")


# ===================================================================
# ToolStatus enum
# ===================================================================


class TestToolStatus:
    def test_all_statuses_exist(self):
        expected = {"success", "error", "unavailable", "skipped"}
        actual = {s.value for s in ToolStatus}
        assert actual == expected


# ===================================================================
# EvidenceStatus enum
# ===================================================================


class TestEvidenceStatus:
    def test_all_evidence_statuses_exist(self):
        expected = {"SUPPORTED", "UNCERTAIN", "INSUFFICIENT"}
        actual = {s.value for s in EvidenceStatus}
        assert actual == expected


# ===================================================================
# WorkflowStatus enum
# ===================================================================


class TestWorkflowStatus:
    def test_all_workflow_statuses_exist(self):
        expected = {"planned", "executing", "completed", "failed", "aborted"}
        actual = {s.value for s in WorkflowStatus}
        assert actual == expected


# ===================================================================
# ImageMetadata
# ===================================================================


class TestImageMetadata:
    def test_valid_construction(self):
        meta = ImageMetadata(
            sensor="Sentinel-2",
            modality=Modality.OPTICAL,
            bands=["red", "nir", "blue", "green"],
            acquisition_date=date(2024, 6, 15),
            crs="EPSG:4326",
            resolution_m=10.0,
        )
        assert meta.sensor == "Sentinel-2"
        assert meta.modality == Modality.OPTICAL
        assert "red" in meta.bands
        assert meta.resolution_m == 10.0

    def test_modality_required(self):
        with pytest.raises(Exception):
            ImageMetadata()  # modality is required


# ===================================================================
# ObservationInput
# ===================================================================


class TestObservationInput:
    def test_valid_construction(self):
        obs = ObservationInput(
            observation_id="obs_001",
            image_path="/data/images/s2_2024.tif",
            role=ObservationRole.T1,
            metadata=ImageMetadata(
                modality=Modality.OPTICAL,
                bands=["red", "nir"],
            ),
        )
        assert obs.observation_id == "obs_001"
        assert obs.role == ObservationRole.T1


# ===================================================================
# ParsedQuery
# ===================================================================


class TestParsedQuery:
    def test_vegetation_change_query(self):
        pq = ParsedQuery(
            raw_query="Has vegetation decreased?",
            task=TaskType.VEGETATION_CHANGE,
            intent="vegetation_decrease",
            claim="vegetation_decrease",
            observations=2,
            temporal=True,
            spatial=True,
            quantification=True,
            modalities=[Modality.OPTICAL],
            required_evidence=["ndvi", "change_statistics"],
        )
        assert pq.task == TaskType.VEGETATION_CHANGE
        assert pq.temporal is True
        assert len(pq.required_evidence) == 2

    def test_serialization_round_trip(self):
        pq = ParsedQuery(
            raw_query="What is visible?",
            task=TaskType.SINGLE_IMAGE_VQA,
            intent="describe",
        )
        data = pq.model_dump()
        reconstructed = ParsedQuery(**data)
        assert reconstructed == pq

    def test_invalid_task_rejected(self):
        with pytest.raises(Exception):
            ParsedQuery(
                raw_query="test",
                task="invalid_task",
                intent="test",
            )


# ===================================================================
# ToolDefinition
# ===================================================================


class TestToolDefinition:
    def test_valid_tool_definition(self):
        td = ToolDefinition(
            name="ndvi_delta",
            description="Compute NDVI delta",
            applicability=ToolApplicability(
                required_modalities=[Modality.OPTICAL],
                required_bands=["red", "nir"],
                min_observations=2,
                requires_temporal=True,
                requires_registration=True,
                min_valid_pixel_fraction=0.5,
                prerequisite_tools=["validate_inputs"],
            ),
            version="0.1.0",
        )
        assert td.name == "ndvi_delta"
        assert len(td.applicability.required_bands) == 2


# ===================================================================
# ToolResult
# ===================================================================


class TestToolResult:
    def test_success_result(self):
        tr = ToolResult(
            tool="ndvi_delta",
            status=ToolStatus.SUCCESS,
            output={"delta_ndvi": -0.071},
            duration_ms=430,
        )
        assert tr.status == ToolStatus.SUCCESS
        assert tr.output["delta_ndvi"] == -0.071

    def test_unavailable_result(self):
        tr = ToolResult(
            tool="ndvi_delta",
            status=ToolStatus.UNAVAILABLE,
            error="Required Red/NIR bands unavailable",
        )
        assert tr.status == ToolStatus.UNAVAILABLE
        assert tr.output is None


# ===================================================================
# EvidenceRecord
# ===================================================================


class TestEvidenceRecord:
    def test_valid_evidence(self):
        er = EvidenceRecord(
            evidence_id="ev_001",
            type="ndvi_delta",
            source="sentinel_2",
            tool_version="0.1.0",
            value=-0.071,
            quality=QualityReport(
                valid_pixel_fraction=0.84,
                registration_ok=True,
            ),
            provenance=Provenance(
                input_ids=["obs_001", "obs_002"],
                tool="ndvi_delta",
                tool_version="0.1.0",
            ),
        )
        assert er.evidence_id == "ev_001"
        assert er.quality.valid_pixel_fraction == 0.84


# ===================================================================
# TraceStep and ExecutionTrace
# ===================================================================


class TestTrace:
    def test_trace_step(self):
        ts = TraceStep(
            step=1,
            tool="validate_inputs",
            status=ToolStatus.SUCCESS,
            duration_ms=132,
        )
        assert ts.step == 1
        assert ts.status == ToolStatus.SUCCESS

    def test_execution_trace(self):
        et = ExecutionTrace(
            trace_id="tr_001",
            workflow_id="wf_001",
            steps=[
                TraceStep(step=1, tool="validate_inputs", status=ToolStatus.SUCCESS, duration_ms=100),
                TraceStep(step=2, tool="run_rs_vlm", status=ToolStatus.SUCCESS, duration_ms=1800),
            ],
            total_duration_ms=1900,
        )
        assert len(et.steps) == 2
        assert et.total_duration_ms == 1900


# ===================================================================
# WorkflowPlan and WorkflowStep
# ===================================================================


class TestWorkflow:
    def test_workflow_step(self):
        ws = WorkflowStep(
            tool="ndvi_delta",
            parameters={"image_t1": "path1", "image_t2": "path2"},
            depends_on=[0],
        )
        assert ws.tool == "ndvi_delta"
        assert ws.depends_on == [0]

    def test_workflow_plan(self):
        wp = WorkflowPlan(
            workflow_id="wf_001",
            task=TaskType.VEGETATION_CHANGE,
            steps=[
                WorkflowStep(tool="validate_inputs"),
                WorkflowStep(tool="run_rs_vlm", depends_on=[0]),
                WorkflowStep(tool="ndvi_delta", depends_on=[0]),
                WorkflowStep(tool="compare_evidence", depends_on=[1, 2]),
                WorkflowStep(tool="generate_response", depends_on=[3]),
            ],
            required_modalities=[Modality.OPTICAL],
            required_evidence=["ndvi", "change_statistics"],
        )
        assert wp.status == WorkflowStatus.PLANNED
        assert len(wp.steps) == 5


# ===================================================================
# ComparisonResult and FinalResponse
# ===================================================================


class TestResponse:
    def test_comparison_supported(self):
        cr = ComparisonResult(
            status=EvidenceStatus.SUPPORTED,
            reason="NDVI delta and change statistics support vegetation decrease.",
            supporting_evidence=["ev_001", "ev_002"],
        )
        assert cr.status == EvidenceStatus.SUPPORTED

    def test_comparison_insufficient(self):
        cr = ComparisonResult(
            status=EvidenceStatus.INSUFFICIENT,
            reason="Required spectral bands are unavailable.",
            limitations=["Red/NIR bands missing"],
        )
        assert cr.status == EvidenceStatus.INSUFFICIENT

    def test_final_response(self):
        fr = FinalResponse(
            trace_id="tr_001",
            task="vegetation_change",
            answer="Vegetation has decreased in the observed region.",
            status=EvidenceStatus.SUPPORTED,
            limitations=[],
            model_versions={"vlm": "mock-0.1.0"},
        )
        assert fr.status == EvidenceStatus.SUPPORTED
        assert fr.trace_id == "tr_001"
