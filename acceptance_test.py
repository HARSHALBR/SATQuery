"""Acceptance tests for Phase 6 end-to-end mock execution."""

from agents.task_classifier import TaskClassifier
from agents.planner import ConstrainedPlanner
from agents.tool_registry import ToolRegistry, ToolExecutionContext
from schemas.query import QueryInput, Modality
from agents.mock_tools import MockToolRunner, MockScenario
from agents.execution_engine import ExecutionEngine
from schemas.workflow import WorkflowStatus

def run_acceptance_tests():
    # Setup
    classifier = TaskClassifier()
    registry = ToolRegistry.from_yaml("configs/tools.yaml")
    
    # Context 1: Perfect Optical
    optical_context = ToolExecutionContext(
        available_modalities=[Modality.OPTICAL],
        available_bands=["red", "nir"],
        num_observations=2,
        has_temporal=True,
        registration_status=True,
        valid_pixel_fraction=0.9
    )
    
    # Context 2: SAR Only
    sar_context = ToolExecutionContext(
        available_modalities=[Modality.SAR],
        available_bands=[],
        num_observations=2,
        has_temporal=True,
        registration_status=True,
        valid_pixel_fraction=0.9
    )
    
    query = QueryInput(
        query="Has vegetation decreased?",
        image_ids=["img1", "img2"]
    )
    parsed_query = classifier.classify(query)
    
    # ---------------------------------------------------------
    # SCENARIO 1 - SUPPORTED (VEGETATION_DECREASE)
    # ---------------------------------------------------------
    print("--- SCENARIO 1: SUPPORTED ---")
    planner = ConstrainedPlanner(registry, "configs/workflows.yaml")
    plan1 = planner.create_plan(parsed_query, optical_context)
    
    assert plan1.task.value == "vegetation_change"
    assert "ndvi_delta" in [s.tool for s in plan1.steps]
    
    runner1 = MockToolRunner(MockScenario.VEGETATION_DECREASE)
    engine1 = ExecutionEngine(runner1)
    trace1, evidence1 = engine1.execute_plan(plan1)
    
    assert plan1.status == WorkflowStatus.COMPLETED
    ev_types = [e.type for e in evidence1]
    assert "vlm_interpretation" in ev_types
    assert "vegetation_change" in ev_types
    assert "change_quantification" in ev_types
    
    # Check compare_evidence result from the trace (it's the last step usually)
    compare_step = next(s for s in trace1.steps if s.tool == "compare_evidence")
    assert "'status': 'SUPPORTED'" in compare_step.output_summary or '"status": "SUPPORTED"' in compare_step.output_summary
    print("Scenario 1 Passed.")
    
    # ---------------------------------------------------------
    # SCENARIO 2 - CONFLICT (CONFLICTING_EVIDENCE)
    # ---------------------------------------------------------
    print("--- SCENARIO 2: CONFLICT ---")
    plan2 = planner.create_plan(parsed_query, optical_context)
    runner2 = MockToolRunner(MockScenario.CONFLICTING_EVIDENCE)
    engine2 = ExecutionEngine(runner2)
    trace2, evidence2 = engine2.execute_plan(plan2)
    
    assert plan2.status == WorkflowStatus.COMPLETED
    # Check outputs for conflict
    vlm_step = next(s for s in trace2.steps if s.tool == "run_rs_vlm")
    ndvi_step = next(s for s in trace2.steps if s.tool == "ndvi_delta")
    compare_step2 = next(s for s in trace2.steps if s.tool == "compare_evidence")
    
    assert "decreased" in vlm_step.output_summary
    assert "increase" in ndvi_step.output_summary
    assert "'status': 'UNCERTAIN'" in compare_step2.output_summary or '"status": "UNCERTAIN"' in compare_step2.output_summary
    print("Scenario 2 Passed.")
    
    # ---------------------------------------------------------
    # SCENARIO 3 - LOW QUALITY (LOW_QUALITY)
    # ---------------------------------------------------------
    print("--- SCENARIO 3: LOW QUALITY ---")
    plan3 = planner.create_plan(parsed_query, optical_context)
    runner3 = MockToolRunner(MockScenario.LOW_QUALITY)
    engine3 = ExecutionEngine(runner3)
    trace3, evidence3 = engine3.execute_plan(plan3)
    
    assert plan3.status == WorkflowStatus.COMPLETED
    compare_step3 = next(s for s in trace3.steps if s.tool == "compare_evidence")
    
    # Check quality
    ndvi_ev = next(e for e in evidence3 if e.type == "vegetation_change")
    assert ndvi_ev.quality.valid_pixel_fraction == 0.40
    assert "'status': 'INSUFFICIENT'" in compare_step3.output_summary or '"status": "INSUFFICIENT"' in compare_step3.output_summary
    print("Scenario 3 Passed.")
    
    # ---------------------------------------------------------
    # SCENARIO 4 - CAPABILITY FAILURE
    # ---------------------------------------------------------
    print("--- SCENARIO 4: CAPABILITY FAILURE ---")
    plan4 = planner.create_plan(parsed_query, sar_context)
    
    assert parsed_query.task.value == "vegetation_change"
    assert plan4.task.value == "vegetation_change"
    assert plan4.status == WorkflowStatus.FAILED
    assert "Required modality" in plan4.fallback or "Required band" in plan4.fallback
    assert len(plan4.steps) == 0
    print("Scenario 4 Passed.")

if __name__ == "__main__":
    run_acceptance_tests()
    print("All Acceptance Tests Passed.")
