"""Mock Tool Runner for SATQuery AI Phase 6.

Implements the ToolRunner Protocol to allow testing the ExecutionEngine
without real VLM or remote-sensing models. Provides deterministic,
semantically meaningful outputs based on scenarios.
"""

from typing import Any, Dict
from enum import Enum
import uuid

from schemas.tools import ToolResult, ToolStatus
from schemas.evidence import EvidenceRecord, Provenance, QualityReport
from agents.execution_engine import ToolRunner


class MockScenario(str, Enum):
    """Supported scenarios for deterministic mock outputs."""
    NORMAL = "NORMAL"
    VEGETATION_DECREASE = "VEGETATION_DECREASE"
    VEGETATION_INCREASE = "VEGETATION_INCREASE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    LOW_QUALITY = "LOW_QUALITY"
    TOOL_FAILURE = "TOOL_FAILURE"


class MockToolRunner(ToolRunner):
    """Deterministic mock runner for all 10 registered tools."""
    
    def __init__(self, scenario: MockScenario = MockScenario.NORMAL):
        self.scenario = scenario

    def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        """Dispatch execution to the appropriate mock tool."""
        
        # Scenario TOOL_FAILURE: we make one specific tool fail
        if self.scenario == MockScenario.TOOL_FAILURE and tool_name == "ndvi_delta":
            return ToolResult(
                tool=tool_name,
                status=ToolStatus.ERROR,
                error="Mock scenario forced failure for ndvi_delta",
                duration_ms=50
            )

        dispatch_map = {
            "validate_inputs": self._mock_validate_inputs,
            "run_rs_vlm": self._mock_run_rs_vlm,
            "grounding": self._mock_grounding,
            "ndvi_delta": self._mock_ndvi_delta,
            "ndbi_delta": self._mock_ndbi_delta,
            "sar_change": self._mock_sar_change,
            "change_statistics": self._mock_change_statistics,
            "area_measurement": self._mock_area_measurement,
            "compare_evidence": self._mock_compare_evidence,
            "generate_response": self._mock_generate_response,
        }
        
        if tool_name not in dispatch_map:
            return ToolResult(
                tool=tool_name,
                status=ToolStatus.ERROR,
                error=f"Unknown mock tool: {tool_name}",
                duration_ms=10
            )
            
        try:
            return dispatch_map[tool_name](parameters)
        except Exception as e:
            return ToolResult(
                tool=tool_name,
                status=ToolStatus.ERROR,
                error=f"Mock tool internal error: {e}",
                duration_ms=10
            )

    def _create_evidence(self, tool_name: str, evidence_type: str, value: Any, quality_ok: bool = True) -> EvidenceRecord:
        """Helper to construct deterministic mock evidence."""
        quality = QualityReport(
            valid_pixel_fraction=0.95 if quality_ok else 0.40,
            registration_ok=True if quality_ok else False,
            cloud_cover=0.05 if quality_ok else 0.60,
            notes=[] if quality_ok else ["Low quality mock scenario"]
        )
        
        prov = Provenance(
            tool=tool_name,
            tool_version="mock-1.0"
        )
        
        return EvidenceRecord(
            evidence_id=f"mock_{tool_name}_{uuid.uuid4().hex[:8]}",
            type=evidence_type,
            tool_version="mock-1.0",
            value=value,
            quality=quality,
            provenance=prov
        )

    def _mock_validate_inputs(self, params: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool="validate_inputs",
            status=ToolStatus.SUCCESS,
            output={"is_valid": True, "notes": "Mock validation passed"},
            duration_ms=10
        )

    def _mock_run_rs_vlm(self, params: Dict[str, Any]) -> ToolResult:
        interpretation = "Vegetation appears to have decreased."
        if self.scenario == MockScenario.VEGETATION_INCREASE:
            interpretation = "Vegetation appears to have increased."
            
        value = {
            "interpretation": interpretation,
            "confidence": 0.82,
            "region": "roi_1"
        }
        ev = self._create_evidence("run_rs_vlm", "vlm_interpretation", value)
        return ToolResult(
            tool="run_rs_vlm",
            status=ToolStatus.SUCCESS,
            output={"interpretation": interpretation, "confidence": 0.82, "evidence": ev},
            duration_ms=500
        )

    def _mock_grounding(self, params: Dict[str, Any]) -> ToolResult:
        val = {"bounding_box": [10.0, 20.0, 10.1, 20.1], "confidence": 0.9}
        ev = self._create_evidence("grounding", "spatial_grounding", val)
        return ToolResult(
            tool="grounding",
            status=ToolStatus.SUCCESS,
            output={"bounding_box": val["bounding_box"], "evidence": ev},
            duration_ms=100
        )

    def _mock_ndvi_delta(self, params: Dict[str, Any]) -> ToolResult:
        delta = -0.18
        direction = "decrease"
        quality_ok = self.scenario != MockScenario.LOW_QUALITY
        
        if self.scenario == MockScenario.VEGETATION_INCREASE:
            delta = 0.22
            direction = "increase"
        elif self.scenario == MockScenario.CONFLICTING_EVIDENCE:
            # Conflicting with VLM (VLM says decrease, NDVI says increase)
            delta = 0.15
            direction = "increase"

        val = {
            "ndvi_delta": delta,
            "direction": direction,
            "affected_fraction": 0.21
        }
        ev = self._create_evidence("ndvi_delta", "vegetation_change", val, quality_ok)
        
        return ToolResult(
            tool="ndvi_delta",
            status=ToolStatus.SUCCESS,
            output={"ndvi_delta": delta, "direction": direction, "evidence": ev},
            duration_ms=250
        )

    def _mock_ndbi_delta(self, params: Dict[str, Any]) -> ToolResult:
        val = {"ndbi_delta": 0.12, "direction": "increase", "affected_fraction": 0.05}
        ev = self._create_evidence("ndbi_delta", "built_up_change", val)
        return ToolResult(
            tool="ndbi_delta",
            status=ToolStatus.SUCCESS,
            output={"ndbi_delta": 0.12, "direction": "increase", "evidence": ev},
            duration_ms=250
        )

    def _mock_sar_change(self, params: Dict[str, Any]) -> ToolResult:
        val = {"sar_change_detected": True, "change_score": 0.76}
        ev = self._create_evidence("sar_change", "sar_amplitude_change", val)
        return ToolResult(
            tool="sar_change",
            status=ToolStatus.SUCCESS,
            output={"sar_change_detected": True, "evidence": ev},
            duration_ms=300
        )

    def _mock_change_statistics(self, params: Dict[str, Any]) -> ToolResult:
        val = {"changed_pixel_fraction": 0.21, "change_detected": True}
        ev = self._create_evidence("change_statistics", "change_quantification", val)
        return ToolResult(
            tool="change_statistics",
            status=ToolStatus.SUCCESS,
            output={"changed_pixel_fraction": 0.21, "change_detected": True, "evidence": ev},
            duration_ms=150
        )

    def _mock_area_measurement(self, params: Dict[str, Any]) -> ToolResult:
        val = {"area_km2": 12.4}
        ev = self._create_evidence("area_measurement", "spatial_measurement", val)
        return ToolResult(
            tool="area_measurement",
            status=ToolStatus.SUCCESS,
            output={"area_km2": 12.4, "evidence": ev},
            duration_ms=50
        )

    def _mock_compare_evidence(self, params: Dict[str, Any]) -> ToolResult:
        status = "SUPPORTED"
        if self.scenario == MockScenario.CONFLICTING_EVIDENCE:
            status = "UNCERTAIN"
        elif self.scenario == MockScenario.LOW_QUALITY:
            status = "INSUFFICIENT"
            
        return ToolResult(
            tool="compare_evidence",
            status=ToolStatus.SUCCESS,
            output={
                "status": status,
                "reason": f"Mock reason for {status}",
                "supporting_evidence": ["ev1", "ev2"] if status == "SUPPORTED" else [],
                "conflicting_evidence": ["ev1", "ev2"] if status == "UNCERTAIN" else [],
                "limitations": ["Low quality data"] if status == "INSUFFICIENT" else []
            },
            duration_ms=100
        )

    def _mock_generate_response(self, params: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool="generate_response",
            status=ToolStatus.SUCCESS,
            output={
                "answer": f"Mock generated response for {self.scenario.value}",
                "status": "SUPPORTED" # Just echoing something
            },
            duration_ms=50
        )
