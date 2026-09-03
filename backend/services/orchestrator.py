"""Orchestrates SATQuery AI components for a single request.

Coordinates:
- TaskClassifier
- ConstrainedPlanner
- ToolRegistry
- ExecutionEngine
- MockToolRunner
- EvidenceStore
- EvidenceComparator
- TraceStore
"""

import logging
from typing import List, Optional
from schemas.query import QueryInput, Modality
from schemas.response import FinalResponse, EvidenceStatus, SpatialEvidence, BoundsWGS84, GeoCenter, SpatialChangeRegion
from schemas.workflow import WorkflowStatus
from agents.task_classifier import TaskClassifier
from agents.planner import ConstrainedPlanner
from agents.tool_registry import ToolRegistry, ToolExecutionContext
from agents.execution_engine import ExecutionEngine
from agents.mock_tools import MockToolRunner, MockScenario
from evidence.evidence_store import EvidenceStore
from evidence.comparator import EvidenceComparator
from trace.trace_store import TraceStore

logger = logging.getLogger(__name__)


class SATQueryOrchestrator:
    """Coordinates a single SATQuery analysis request."""

    def __init__(self, scenario: MockScenario = MockScenario.NORMAL, runner=None):
        """Initialize the orchestrator with clean state per request."""
        self.scenario = scenario
        self.classifier = TaskClassifier()
        self.registry = ToolRegistry.from_yaml("configs/tools.yaml")
        self.planner = ConstrainedPlanner(self.registry, "configs/workflows.yaml")
        self.runner = runner if runner is not None else MockToolRunner(self.scenario)
        self.engine = ExecutionEngine(self.runner)
        self.evidence_store = EvidenceStore()
        self.comparator = EvidenceComparator()
        self.trace_store = TraceStore()
        runner_version = "mock-1.0"
        if runner is not None and type(runner).__name__ == "RealToolRunner":
            runner_version = "real-1.0"
        self.model_versions = {
            "classifier": "1.0",
            "planner": "1.0",
            "engine": "1.0",
            "runner": runner_version,
        }

    def _prepare_context(self, query: QueryInput) -> ToolExecutionContext:
        """Derive ToolExecutionContext from the QueryInput observations."""
        modalities = set()
        bands = set()
        
        for obs in query.observations:
            modalities.add(obs.metadata.modality)
            for b in obs.metadata.bands:
                bands.add(b.lower())

        meta = query.metadata or {}

        return ToolExecutionContext(
            available_modalities=list(modalities),
            available_bands=list(bands),
            num_observations=len(query.observations),
            has_temporal=meta.get("has_temporal", len(query.observations) >= 2),
            registration_status=meta.get("registration_status", True),
            valid_pixel_fraction=meta.get("valid_pixel_fraction", 0.9),
        )

    def _build_spatial_evidence(self, evidence_list: list) -> Optional[SpatialEvidence]:
        """Extract spatial bounds from spatial_grounding evidence record.
        
        The grounding tool returns a bounding_box in WGS84 [west, south, east, north].
        We promote this into the top-level spatial_evidence response field so the
        frontend can zoom the Leaflet map to the actual observation extent.
        """
        for ev in evidence_list:
            if ev.type == "change_quantification" and ev.value:
                spatial = ev.value.get("spatial_evidence")
                if spatial:
                    try:
                        return SpatialEvidence(
                            available=bool(spatial.get("available")),
                            spatial_grounding=spatial.get("spatial_grounding"),
                            crs=spatial.get("crs"),
                            bounds_wgs84=BoundsWGS84(**spatial["bounds_wgs84"]) if spatial.get("bounds_wgs84") else None,
                            center=GeoCenter(**spatial["center"]) if spatial.get("center") else None,
                            observation_extent=spatial.get("observation_extent"),
                            change_regions=[SpatialChangeRegion(**region) for region in spatial.get("change_regions", [])],
                            reason=spatial.get("reason"),
                        )
                    except (TypeError, ValueError, KeyError):
                        logger.warning("Invalid spatial evidence returned by change statistics")
            if ev.type == "spatial_grounding" and ev.value:
                bb = ev.value.get("bounding_box")
                if bb and len(bb) == 4:
                    west, south, east, north = bb
                    try:
                        return SpatialEvidence(
                            available=True,
                            crs="EPSG:4326",
                            bounds_wgs84=BoundsWGS84(
                                west=float(west),
                                south=float(south),
                                east=float(east),
                                north=float(north),
                            ),
                            center=GeoCenter(
                                lat=float((south + north) / 2),
                                lon=float((west + east) / 2),
                            ),
                        )
                    except (TypeError, ValueError):
                        pass
        return SpatialEvidence(
            available=False,
            reason="Spatial bounds could not be derived from the input.",
        )

    def analyze(self, query: QueryInput) -> FinalResponse:
        """Execute the full end-to-end analysis pipeline."""
        # 1. Context Preparation
        context = self._prepare_context(query)

        # 2. Classification
        parsed = self.classifier.classify(query)

        # 3. Planning
        plan = self.planner.create_plan(parsed, context)

        # Handle planning failure
        if plan.status == WorkflowStatus.FAILED:
            return FinalResponse(
                trace_id=plan.workflow_id,  # Use workflow_id as fallback trace_id
                task=getattr(parsed.task, "value", str(parsed.task)),
                answer=plan.fallback or "Capability missing.",
                status=EvidenceStatus.INSUFFICIENT,
                evidence=[],
                limitations=[plan.fallback or "Missing required capabilities."],
                execution_trace=[],
                model_versions=self.model_versions,
                spatial_evidence=SpatialEvidence(available=False, reason="Analysis could not proceed."),
            )

        try:
            # 4. Execution
            trace, evidence_list = self.engine.execute_plan(plan)

            # 5. Store Evidence
            if evidence_list:
                self.evidence_store.add_many(evidence_list)

            # 6. Store Trace
            self.trace_store.add(trace)

            # Handle execution failure
            if plan.status == WorkflowStatus.FAILED:
                spatial_ev = self._build_spatial_evidence(self.evidence_store.list())
                return FinalResponse(
                    trace_id=trace.trace_id,
                    task=getattr(parsed.task, "value", str(parsed.task)),
                    answer="Execution failed.",
                    status=EvidenceStatus.INSUFFICIENT,
                    evidence=self.evidence_store.list(),
                    limitations=["Workflow execution encountered an error."],
                    execution_trace=trace.steps,
                    model_versions=self.model_versions,
                    spatial_evidence=spatial_ev,
                )

            # 7. Comparison
            comp_result = self.comparator.compare(parsed.claim, self.evidence_store.list())

            # 8. Final Answer (Mock)
            lines = []
            if comp_result.status == EvidenceStatus.SUPPORTED:
                lines.append(f"The claim '{parsed.claim}' is supported by the evidence.")
            elif comp_result.status == EvidenceStatus.UNCERTAIN:
                lines.append(f"The evidence for claim '{parsed.claim}' is conflicting or uncertain.")
            else:
                lines.append(f"There is insufficient evidence to evaluate the claim '{parsed.claim}'.")
                
            lines.append(f"Reason: {comp_result.reason}")
            
            if comp_result.supporting_evidence:
                lines.append(f"Supporting Evidence IDs: {', '.join(comp_result.supporting_evidence)}")
            if comp_result.conflicting_evidence:
                lines.append(f"Conflicting Evidence IDs: {', '.join(comp_result.conflicting_evidence)}")
                
            answer = "\n".join(lines)

            # 9. Build spatial evidence from grounding output
            spatial_ev = self._build_spatial_evidence(self.evidence_store.list())

            # 10. Return Response
            return FinalResponse(
                trace_id=trace.trace_id,
                task=getattr(parsed.task, "value", str(parsed.task)),
                answer=answer,
                status=comp_result.status,
                evidence=self.evidence_store.list(),
                limitations=comp_result.limitations,
                execution_trace=trace.steps,
                model_versions=self.model_versions,
                spatial_evidence=spatial_ev,
            )
        finally:
            if hasattr(self.runner, "cleanup"):
                self.runner.cleanup()
