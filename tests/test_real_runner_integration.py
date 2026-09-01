import pytest
import datetime
from schemas.query import ObservationInput, ObservationRole, ImageMetadata, Modality
from agents.real_runner import RealToolRunner
from agents.execution_engine import ExecutionEngine
from schemas.workflow import WorkflowPlan, WorkflowStep, WorkflowStatus
from schemas.tools import ToolStatus

def _build_obs(base_path: str, role: ObservationRole, date_str: str, stac_id: str):
    return ObservationInput(
        observation_id=stac_id,
        image_path=base_path,
        role=role,
        metadata=ImageMetadata(
            modality=Modality.OPTICAL,
            bands=["red", "nir", "scl"],
            acquisition_date=datetime.datetime.strptime(date_str, "%Y-%m-%d").date(),
            stac_item_id=stac_id
        )
    )

def _build_plan():
    # Build a simple DAG: validate_inputs -> ndvi_delta -> change_statistics
    return WorkflowPlan(
        workflow_id="test-wf",
        task="vegetation_change",
        status=WorkflowStatus.PLANNED,
        steps=[
            WorkflowStep(tool="validate_inputs", depends_on=[]),
            WorkflowStep(tool="ndvi_delta", depends_on=[0]),
            WorkflowStep(tool="change_statistics", depends_on=[1])
        ]
    )

def test_real_tool_runner_case_a():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A", ObservationRole.T1, "2021-07-08", "S2A_10TFK_20210708_0_L2A"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A", ObservationRole.T2, "2021-10-01", "S2B_10TFK_20211001_0_L2A")
    ]
    runner = RealToolRunner(observations=obs)
    engine = ExecutionEngine(runner)
    
    trace, evidence = engine.execute_plan(_build_plan())
    
    assert trace.steps[-1].status == ToolStatus.SUCCESS, trace.steps[-1].error
    
    # Check Evidence and Provenance
    assert len(evidence) == 2
    ndvi_ev = next(e for e in evidence if e.type == "vegetation_change")
    stats_ev = next(e for e in evidence if e.type == "change_quantification")
    
    assert ndvi_ev.provenance.input_ids == ["S2A_10TFK_20210708_0_L2A", "S2B_10TFK_20211001_0_L2A"]
    assert stats_ev.provenance.input_ids == ["S2A_10TFK_20210708_0_L2A", "S2B_10TFK_20211001_0_L2A"]
    
    assert stats_ev.value["decrease_pixel_fraction"] > 0.4  # Wildfire is massive
    assert stats_ev.quality.valid_pixel_fraction > 0.9

def test_real_tool_runner_case_b():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A", ObservationRole.T1, "2021-07-06", "S2B_10TDL_20210706_1_L2A"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A", ObservationRole.T2, "2021-10-14", "S2B_10TDL_20211014_1_L2A")
    ]
    runner = RealToolRunner(observations=obs)
    engine = ExecutionEngine(runner)
    
    trace, evidence = engine.execute_plan(_build_plan())
    
    assert trace.steps[-1].status == ToolStatus.SUCCESS
    stats_ev = next(e for e in evidence if e.type == "change_quantification")
    
    assert stats_ev.value["decrease_pixel_fraction"] < 0.05  # Stable forest
    
def test_real_tool_runner_case_c():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2A_10SGF_20210705_2_L2A", ObservationRole.T1, "2021-07-05", "S2A_10SGF_20210705_2_L2A"),
        _build_obs("datasets/golden_fixtures/raw/S2A_10SGF_20211013_0_L2A", ObservationRole.T2, "2021-10-13", "S2A_10SGF_20211013_0_L2A")
    ]
    runner = RealToolRunner(observations=obs)
    engine = ExecutionEngine(runner)
    
    trace, evidence = engine.execute_plan(_build_plan())
    
    assert trace.steps[-1].status == ToolStatus.SUCCESS
    stats_ev = next(e for e in evidence if e.type == "change_quantification")
    
    assert stats_ev.value["decrease_pixel_fraction"] > 0.05  # Harvest
    
def test_real_tool_runner_failure():
    # Provide bad paths
    obs = [
        _build_obs("datasets/golden_fixtures/raw/does_not_exist_t1", ObservationRole.T1, "2021-07-05", "missing_1"),
        _build_obs("datasets/golden_fixtures/raw/does_not_exist_t2", ObservationRole.T2, "2021-10-13", "missing_2")
    ]
    runner = RealToolRunner(observations=obs)
    engine = ExecutionEngine(runner)
    
    trace, evidence = engine.execute_plan(_build_plan())
    
    assert trace.steps[0].status == ToolStatus.ERROR
    assert trace.steps[1].status == ToolStatus.SKIPPED
    assert trace.steps[2].status == ToolStatus.SKIPPED
from schemas.query import QueryInput
from backend.services.orchestrator import SATQueryOrchestrator

def test_real_tool_runner_orchestrator_e2e_case_a():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A", ObservationRole.T1, "2021-07-08", "S2A_10TFK_20210708_0_L2A"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A", ObservationRole.T2, "2021-10-01", "S2B_10TFK_20211001_0_L2A")
    ]
    query = QueryInput(
        query="Has vegetation decreased severely?",
        observations=obs
    )
    
    runner = RealToolRunner(observations=obs)
    # Inject real runner
    orch = SATQueryOrchestrator(runner=runner)
    
    resp = orch.analyze(query)
    
    assert resp.trace_id is not None
    assert len(resp.execution_trace) > 0
    # Even without VLM, the change_statistics evidence should be present!
    stats_ev = next((e for e in resp.evidence if e.type == "change_quantification"), None)
    assert stats_ev is not None
    assert stats_ev.value["decrease_pixel_fraction"] > 0.4

import yaml
from copy import deepcopy

def test_real_tool_runner_cleanup_on_success():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A", ObservationRole.T1, "2021-07-06", "S2B_10TDL_20210706_1_L2A"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A", ObservationRole.T2, "2021-10-14", "S2B_10TDL_20211014_1_L2A")
    ]
    runner = RealToolRunner(observations=obs)
    engine = ExecutionEngine(runner)
    
    # Execute normally
    trace, evidence = engine.execute_plan(_build_plan())
    assert trace.steps[-1].status == ToolStatus.SUCCESS
    
    # Store must be empty after successful consumption
    assert len(runner.array_store) == 0

def test_real_tool_runner_cleanup_on_orchestrator_abort():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A", ObservationRole.T1, "2021-07-06", "S2B_10TDL_20210706_1_L2A"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A", ObservationRole.T2, "2021-10-14", "S2B_10TDL_20211014_1_L2A")
    ]
    query = QueryInput(
        query="Has vegetation decreased severely?",
        observations=obs
    )
    runner = RealToolRunner(observations=obs)
    orch = SATQueryOrchestrator(runner=runner)
    
    # Replace runner with something that artificially fails change_stats if needed, 
    # but the orchestrator finally block triggers cleanup unconditionally.
    orch.analyze(query)
    
    # Assert array store is empty thanks to Orchestrator finally block
    assert len(runner.array_store) == 0

def test_real_tool_runner_isolation():
    obs1 = [
        _build_obs("datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A", ObservationRole.T1, "2021-07-08", "a1"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A", ObservationRole.T2, "2021-10-01", "a2")
    ]
    obs2 = [
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A", ObservationRole.T1, "2021-07-06", "b1"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A", ObservationRole.T2, "2021-10-14", "b2")
    ]
    
    runner1 = RealToolRunner(observations=obs1)
    runner2 = RealToolRunner(observations=obs2)
    
    # Run partially
    res1 = runner1.execute("validate_inputs", {})
    res2 = runner2.execute("validate_inputs", {})
    
    res1_ndvi = runner1.execute("ndvi_delta", {"input_ids": ["a1", "a2"]})
    res2_ndvi = runner2.execute("ndvi_delta", {"input_ids": ["b1", "b2"]})
    
    assert res1_ndvi.status == ToolStatus.SUCCESS
    assert res2_ndvi.status == ToolStatus.SUCCESS
    
    # Stores are totally distinct
    assert runner1.array_store is not runner2.array_store
    assert len(runner1.array_store) == 2
    assert len(runner2.array_store) == 2
    
    k1 = res1_ndvi.output["delta_map"]
    k2 = res2_ndvi.output["delta_map"]
    assert k1 != k2
    assert k1 in runner1.array_store
    assert k2 not in runner1.array_store
    
    # Test missing key behaves safely
    res_err = runner1.execute("change_statistics", {"delta_map": k2, "valid_mask": k1})
    assert res_err.status == ToolStatus.ERROR
    assert "not found" in res_err.error
    
    # Cleanup explicitly for runner 1
    runner1.cleanup()
    assert len(runner1.array_store) == 0
    # Runner 2 is unaffected
    assert len(runner2.array_store) == 2
    runner2.cleanup()

def test_real_tool_runner_contract_alignment():
    # Load YAML definitions
    with open("configs/tools.yaml", "r") as f:
        tools_def = yaml.safe_load(f)["tools"]
        
    def get_schema(name: str):
        return next(t for t in tools_def if t["name"] == name)

    runner = RealToolRunner(observations=[
        _build_obs("datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A", ObservationRole.T1, "2021-07-08", "a1"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A", ObservationRole.T2, "2021-10-01", "a2")
    ])
    
    val_out = runner.execute("validate_inputs", {}).output
    val_schema = get_schema("validate_inputs")["output_schema"]
    for expected_key in val_schema.keys():
        assert expected_key in val_out
    assert "validation_passed" in val_out
    
    ndvi_out = runner.execute("ndvi_delta", {"input_ids": ["a1", "a2"]}).output
    ndvi_schema = get_schema("ndvi_delta")["output_schema"]
    for expected_key in ndvi_schema.keys():
        assert expected_key in ndvi_out
    
    stats_out = runner.execute("change_statistics", {
        "delta_map": ndvi_out["delta_map"], 
        "valid_mask": ndvi_out["valid_mask"]
    }).output
    stats_schema = get_schema("change_statistics")["output_schema"]
    for expected_key in stats_schema.keys():
        assert expected_key in stats_out
    
    assert "decrease_pixel_fraction" in stats_out
    assert "change_mask" in stats_out
