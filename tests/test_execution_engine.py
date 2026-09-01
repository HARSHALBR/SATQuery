"""Tests for the Phase 6 Execution Engine."""

import pytest
from typing import Any, Dict
from datetime import datetime, timezone
import uuid

from schemas.query import TaskType
from schemas.workflow import WorkflowPlan, WorkflowStep, WorkflowStatus
from schemas.tools import ToolResult, ToolStatus
from schemas.trace import ExecutionTrace, TraceStep
from schemas.evidence import EvidenceRecord, Provenance
from agents.execution_engine import ExecutionEngine, ToolRunner


class MockToolRunner:
    """Mock runner that returns predefined results based on tool name."""
    
    def __init__(self):
        self.results = {}
        self.exceptions = set()
        
    def execute(self, tool_name: str, parameters: Dict[str, Any]) -> ToolResult:
        if tool_name in self.exceptions:
            raise RuntimeError(f"Mock unhandled exception in {tool_name}")
            
        if tool_name in self.results:
            return self.results[tool_name]
            
        return ToolResult(
            tool=tool_name,
            status=ToolStatus.SUCCESS,
            output={"result": f"mock_{tool_name}"},
            duration_ms=10
        )


@pytest.fixture
def base_plan():
    return WorkflowPlan(
        workflow_id="test_wf",
        task=TaskType.SINGLE_IMAGE_VQA,
        steps=[]
    )


def test_successful_execution(base_plan):
    runner = MockToolRunner()
    engine = ExecutionEngine(runner)
    
    # A -> B -> C
    base_plan.steps = [
        WorkflowStep(tool="tool_A", depends_on=[]),
        WorkflowStep(tool="tool_B", depends_on=[0]),
        WorkflowStep(tool="tool_C", depends_on=[1])
    ]
    
    trace, evidence = engine.execute_plan(base_plan)
    
    assert base_plan.status == WorkflowStatus.COMPLETED
    assert len(trace.steps) == 3
    for step in trace.steps:
        assert step.status == ToolStatus.SUCCESS
        assert step.duration_ms is not None
        assert step.error is None


def test_first_step_failure(base_plan):
    # Case 2: A fails, B and C skipped
    runner = MockToolRunner()
    runner.results["tool_A"] = ToolResult(tool="tool_A", status=ToolStatus.ERROR, error="Failed A")
    engine = ExecutionEngine(runner)
    
    base_plan.steps = [
        WorkflowStep(tool="tool_A", depends_on=[]),
        WorkflowStep(tool="tool_B", depends_on=[0]),
        WorkflowStep(tool="tool_C", depends_on=[1])
    ]
    
    trace, evidence = engine.execute_plan(base_plan)
    
    assert base_plan.status == WorkflowStatus.FAILED
    assert len(trace.steps) == 3
    assert trace.steps[0].status == ToolStatus.ERROR
    assert trace.steps[1].status == ToolStatus.SKIPPED
    assert trace.steps[2].status == ToolStatus.SKIPPED


def test_branch_failure(base_plan):
    # Case 3: A -> B, A -> C, B -> D, C -> D. If B fails, C still runs, D skips.
    runner = MockToolRunner()
    runner.results["tool_B"] = ToolResult(tool="tool_B", status=ToolStatus.ERROR, error="Failed B")
    engine = ExecutionEngine(runner)
    
    base_plan.steps = [
        WorkflowStep(tool="tool_A", depends_on=[]),       # 0
        WorkflowStep(tool="tool_B", depends_on=[0]),      # 1 (Fails)
        WorkflowStep(tool="tool_C", depends_on=[0]),      # 2 (Runs)
        WorkflowStep(tool="tool_D", depends_on=[1, 2])    # 3 (Skips)
    ]
    
    trace, evidence = engine.execute_plan(base_plan)
    
    assert base_plan.status == WorkflowStatus.FAILED
    assert trace.steps[0].status == ToolStatus.SUCCESS
    assert trace.steps[1].status == ToolStatus.ERROR
    assert trace.steps[2].status == ToolStatus.SUCCESS
    assert trace.steps[3].status == ToolStatus.SKIPPED


def test_independent_steps(base_plan):
    # Case 4: A and B independent. Both execute.
    runner = MockToolRunner()
    engine = ExecutionEngine(runner)
    
    base_plan.steps = [
        WorkflowStep(tool="tool_A", depends_on=[]),
        WorkflowStep(tool="tool_B", depends_on=[])
    ]
    
    trace, evidence = engine.execute_plan(base_plan)
    
    assert base_plan.status == WorkflowStatus.COMPLETED
    assert trace.steps[0].status == ToolStatus.SUCCESS
    assert trace.steps[1].status == ToolStatus.SUCCESS


def test_unhandled_exception_caught(base_plan):
    runner = MockToolRunner()
    runner.exceptions.add("tool_A")
    engine = ExecutionEngine(runner)
    
    base_plan.steps = [
        WorkflowStep(tool="tool_A", depends_on=[]),
        WorkflowStep(tool="tool_B", depends_on=[0])
    ]
    
    trace, evidence = engine.execute_plan(base_plan)
    
    assert base_plan.status == WorkflowStatus.FAILED
    assert trace.steps[0].status == ToolStatus.ERROR
    assert "Mock unhandled exception" in trace.steps[0].error
    assert trace.steps[1].status == ToolStatus.SKIPPED


def test_evidence_collection(base_plan):
    runner = MockToolRunner()
    prov = Provenance(tool="tool_A", tool_version="1.0")
    ev = EvidenceRecord(evidence_id="ev_1", type="test", tool_version="1.0", provenance=prov)
    
    runner.results["tool_A"] = ToolResult(
        tool="tool_A", 
        status=ToolStatus.SUCCESS,
        output={"evidence": [ev]}
    )
    engine = ExecutionEngine(runner)
    
    base_plan.steps = [WorkflowStep(tool="tool_A", depends_on=[])]
    trace, evidence = engine.execute_plan(base_plan)
    
    assert len(evidence) == 1
    assert evidence[0].evidence_id == "ev_1"
    
def test_empty_workflow(base_plan):
    runner = MockToolRunner()
    engine = ExecutionEngine(runner)
    base_plan.steps = []
    
    trace, evidence = engine.execute_plan(base_plan)
    
    assert base_plan.status == WorkflowStatus.COMPLETED
    assert len(trace.steps) == 0
