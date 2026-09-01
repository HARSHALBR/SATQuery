"""Tests for the Phase 10 Trace Store."""

import pytest
from datetime import datetime

from schemas.trace import ExecutionTrace, TraceStep
from schemas.tools import ToolStatus
from trace.trace_store import TraceStore, DuplicateTraceError

from agents.task_classifier import TaskClassifier
from agents.planner import ConstrainedPlanner
from agents.tool_registry import ToolRegistry, ToolExecutionContext
from agents.execution_engine import ExecutionEngine
from agents.mock_tools import MockToolRunner, MockScenario
from schemas.query import QueryInput, Modality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(
    tid: str,
    wid: str = "wf_1",
    num_steps: int = 2
) -> ExecutionTrace:
    """Create a minimal ExecutionTrace for testing."""
    steps = []
    for i in range(num_steps):
        steps.append(
            TraceStep(
                step=i,
                tool=f"tool_{i}",
                status=ToolStatus.SUCCESS,
                duration_ms=100 + i,
                input_summary=f"{{'param': {i}}}",
                output_summary=f"{{'result': {i}}}",
            )
        )
    return ExecutionTrace(
        trace_id=tid,
        workflow_id=wid,
        steps=steps,
        total_duration_ms=200,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Basic CRUD & API
# ---------------------------------------------------------------------------

class TestBasicCRUD:
    def test_empty_store(self):
        store = TraceStore()
        assert store.list() == []
        assert store.get("nonexistent") is None

    def test_add_trace(self):
        store = TraceStore()
        t = _make_trace("t1")
        store.add(t)
        assert len(store.list()) == 1

    def test_get_trace_by_id(self):
        store = TraceStore()
        t = _make_trace("t_get")
        store.add(t)
        fetched = store.get("t_get")
        assert fetched is not None
        assert fetched.trace_id == "t_get"

    def test_list_traces(self):
        store = TraceStore()
        store.add(_make_trace("t1"))
        store.add(_make_trace("t2"))
        assert len(store.list()) == 2

    def test_duplicate_trace_id_rejected(self):
        store = TraceStore()
        store.add(_make_trace("dup"))
        with pytest.raises(DuplicateTraceError):
            store.add(_make_trace("dup"))

    def test_deterministic_ordering(self):
        store = TraceStore()
        for i in [3, 1, 4, 2]:
            store.add(_make_trace(f"t_{i}"))
        ids = [t.trace_id for t in store.list()]
        assert ids == ["t_3", "t_1", "t_4", "t_2"]

    def test_get_missing_trace(self):
        store = TraceStore()
        assert store.get("missing_id") is None

    def test_delete_trace(self):
        store = TraceStore()
        store.add(_make_trace("t1"))
        store.delete("t1")
        assert len(store.list()) == 0
        assert store.get("t1") is None
        # Idempotent
        store.delete("t1")

    def test_clear_store(self):
        store = TraceStore()
        store.add(_make_trace("t1"))
        store.add(_make_trace("t2"))
        store.clear()
        assert len(store.list()) == 0


# ---------------------------------------------------------------------------
# Advanced Retrieval
# ---------------------------------------------------------------------------

class TestWorkflowRetrieval:
    def test_get_by_workflow(self):
        store = TraceStore()
        store.add(_make_trace("t1", wid="wA"))
        store.add(_make_trace("t2", wid="wB"))
        store.add(_make_trace("t3", wid="wA"))
        
        wa_traces = store.get_by_workflow("wA")
        assert len(wa_traces) == 2
        assert {t.trace_id for t in wa_traces} == {"t1", "t3"}
        
        wb_traces = store.get_by_workflow("wB")
        assert len(wb_traces) == 1
        assert wb_traces[0].trace_id == "t2"

    def test_multiple_workflows(self):
        store = TraceStore()
        store.add(_make_trace("t1", wid="wf_A"))
        store.add(_make_trace("t2", wid="wf_B"))
        store.add(_make_trace("t3", wid="wf_C"))
        assert len(store.get_by_workflow("wf_B")) == 1

    def test_multiple_traces_for_same_workflow(self):
        # A workflow might be retried or executed in branches resulting in multiple traces
        store = TraceStore()
        store.add(_make_trace("t1_run1", wid="wf_X"))
        store.add(_make_trace("t1_run2", wid="wf_X"))
        assert len(store.get_by_workflow("wf_X")) == 2


# ---------------------------------------------------------------------------
# Integrity & Safety
# ---------------------------------------------------------------------------

class TestIntegrity:
    def test_mutation_safety(self):
        store = TraceStore()
        store.add(_make_trace("mut", num_steps=1))
        
        fetched = store.get("mut")
        fetched.steps[0].status = ToolStatus.ERROR
        
        original = store.get("mut")
        assert original.steps[0].status == ToolStatus.SUCCESS

    def test_trace_fields_remain_intact(self):
        store = TraceStore()
        t = _make_trace("intact", num_steps=1)
        t.steps[0].error = "A wild error appeared"
        store.add(t)
        
        fetched = store.get("intact")
        assert fetched.steps[0].error == "A wild error appeared"
        assert fetched.total_duration_ms == 200

    def test_empty_trace_handling(self):
        # A trace with no steps
        store = TraceStore()
        t = ExecutionTrace(trace_id="empty_t", workflow_id="w1")
        store.add(t)
        fetched = store.get("empty_t")
        assert len(fetched.steps) == 0


# ---------------------------------------------------------------------------
# Integration with Existing Pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_pipeline_trace_into_store(self):
        """TaskClassifier -> Planner -> Engine -> TraceStore."""
        # 1. Classify
        classifier = TaskClassifier()
        query = QueryInput(query="Has vegetation decreased?", image_ids=["img1", "img2"])
        parsed = classifier.classify(query)

        # 2. Plan
        registry = ToolRegistry.from_yaml("configs/tools.yaml")
        context = ToolExecutionContext(
            available_modalities=[Modality.OPTICAL],
            available_bands=["red", "nir"],
            num_observations=2,
            has_temporal=True,
            registration_status=True,
            valid_pixel_fraction=0.9,
        )
        planner = ConstrainedPlanner(registry, "configs/workflows.yaml")
        plan = planner.create_plan(parsed, context)

        # 3. Execute
        runner = MockToolRunner(MockScenario.VEGETATION_DECREASE)
        engine = ExecutionEngine(runner)
        trace, evidence_list = engine.execute_plan(plan)

        # 4. Store Trace
        store = TraceStore()
        store.add(trace)

        # 5. Verify Integration Integrity
        stored_trace = store.get(trace.trace_id)
        assert stored_trace is not None
        assert stored_trace.workflow_id == plan.workflow_id
        
        # Verify TraceSteps are preserved
        assert len(stored_trace.steps) == len(plan.steps)
        for i, step in enumerate(stored_trace.steps):
            assert step.tool == plan.steps[i].tool
            assert step.status == ToolStatus.SUCCESS
            assert step.duration_ms is not None
            assert step.duration_ms > 0
            
            # Note: We know input_summary and output_summary are strings (tech debt)
            # but they should be preserved as strings or None.
            assert step.input_summary is None or isinstance(step.input_summary, str)
            assert step.output_summary is None or isinstance(step.output_summary, str)
            
            # Error should be None in SUCCESS scenario
            assert step.error is None
            
        assert stored_trace.total_duration_ms >= 0
