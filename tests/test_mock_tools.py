"""Tests for the Phase 6 Mock Tools and Integration."""

import pytest
from agents.mock_tools import MockToolRunner, MockScenario
from schemas.tools import ToolStatus
from agents.execution_engine import ExecutionEngine
from agents.task_classifier import TaskClassifier
from agents.planner import ConstrainedPlanner
from agents.tool_registry import ToolRegistry, ToolExecutionContext
from schemas.query import QueryInput, Modality

def test_unknown_tool():
    runner = MockToolRunner()
    res = runner.execute("invalid_tool", {})
    assert res.status == ToolStatus.ERROR
    assert "Unknown mock tool" in res.error

@pytest.mark.parametrize("tool_name", [
    "validate_inputs", "run_rs_vlm", "grounding", "ndvi_delta",
    "ndbi_delta", "sar_change", "change_statistics", "area_measurement",
    "compare_evidence", "generate_response"
])
def test_all_tools_dispatch(tool_name):
    runner = MockToolRunner()
    res = runner.execute(tool_name, {})
    assert res.status == ToolStatus.SUCCESS
    assert res.output is not None

def test_run_rs_vlm_structured():
    res = MockToolRunner().execute("run_rs_vlm", {})
    assert "interpretation" in res.output
    assert "confidence" in res.output
    assert "evidence" in res.output
    assert res.output["evidence"].type == "vlm_interpretation"

def test_deterministic_output():
    r1 = MockToolRunner(MockScenario.NORMAL).execute("ndvi_delta", {})
    r2 = MockToolRunner(MockScenario.NORMAL).execute("ndvi_delta", {})
    # Same value
    assert r1.output["ndvi_delta"] == r2.output["ndvi_delta"]

def test_scenario_vegetation_decrease():
    runner = MockToolRunner(MockScenario.VEGETATION_DECREASE)
    vlm = runner.execute("run_rs_vlm", {})
    ndvi = runner.execute("ndvi_delta", {})
    assert "decreased" in vlm.output["interpretation"]
    assert ndvi.output["ndvi_delta"] < 0
    assert ndvi.output["direction"] == "decrease"

def test_scenario_vegetation_increase():
    runner = MockToolRunner(MockScenario.VEGETATION_INCREASE)
    vlm = runner.execute("run_rs_vlm", {})
    ndvi = runner.execute("ndvi_delta", {})
    assert "increased" in vlm.output["interpretation"]
    assert ndvi.output["ndvi_delta"] > 0
    assert ndvi.output["direction"] == "increase"

def test_scenario_conflicting_evidence():
    runner = MockToolRunner(MockScenario.CONFLICTING_EVIDENCE)
    vlm = runner.execute("run_rs_vlm", {})
    ndvi = runner.execute("ndvi_delta", {})
    comp = runner.execute("compare_evidence", {})
    
    assert "decreased" in vlm.output["interpretation"]
    assert ndvi.output["direction"] == "increase"
    assert comp.output["status"] == "UNCERTAIN"

def test_scenario_low_quality():
    runner = MockToolRunner(MockScenario.LOW_QUALITY)
    ndvi = runner.execute("ndvi_delta", {})
    comp = runner.execute("compare_evidence", {})
    
    assert ndvi.output["evidence"].quality.valid_pixel_fraction < 0.5
    assert comp.output["status"] == "INSUFFICIENT"

def test_scenario_tool_failure():
    runner = MockToolRunner(MockScenario.TOOL_FAILURE)
    ndvi = runner.execute("ndvi_delta", {})
    assert ndvi.status == ToolStatus.ERROR


def test_end_to_end_integration():
    """Classifier -> Planner -> ExecutionEngine -> MockToolRunner"""
    # 1. Classifier
    classifier = TaskClassifier()
    query = QueryInput(
        query="Has vegetation decreased?",
        image_ids=["img1", "img2"]
    )
    parsed_query = classifier.classify(query)
    
    # 2. Planner context
    registry = ToolRegistry.from_yaml("configs/tools.yaml")
    context = ToolExecutionContext(
        available_modalities=[Modality.OPTICAL],
        available_bands=["red", "nir"],
        num_observations=2,
        has_temporal=True,
        registration_status=True,
        valid_pixel_fraction=0.9
    )
    
    # 3. Planner
    planner = ConstrainedPlanner(registry, "configs/workflows.yaml")
    plan = planner.create_plan(parsed_query, context)
    
    # Ensure plan is valid
    assert plan.status.value == "planned"
    tool_names = [s.tool for s in plan.steps]
    assert "ndvi_delta" in tool_names
    assert "change_statistics" in tool_names
    
    # 4. ExecutionEngine
    runner = MockToolRunner(MockScenario.VEGETATION_DECREASE)
    engine = ExecutionEngine(runner)
    
    trace, evidence = engine.execute_plan(plan)
    
    # Verify execution complete
    assert plan.status.value == "completed"
    
    # Verify trace generated
    assert len(trace.steps) == len(plan.steps)
    for t in trace.steps:
        assert t.status == ToolStatus.SUCCESS
        
    # Verify evidence generated
    assert len(evidence) > 0
    types = [e.type for e in evidence]
    assert "vegetation_change" in types
    assert "change_quantification" in types
