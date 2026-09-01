import pytest
from unittest.mock import MagicMock
import os
import yaml
import tempfile
import uuid
import sys

# Ensure imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.planner import ConstrainedPlanner, WorkflowStatus, WorkflowStep

@pytest.fixture
def temp_config():
    config = {
        "workflows": {
            "single_image_vqa": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]},
                {"tool": "generate_response", "depends_on": ["run_rs_vlm"]}
            ],
            "captioning": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]},
                {"tool": "generate_response", "depends_on": ["run_rs_vlm"]}
            ],
            "grounding": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]},
                {"tool": "grounding", "depends_on": ["run_rs_vlm"]},
                {"tool": "generate_response", "depends_on": ["grounding"]}
            ],
            "bi_temporal_change": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]},
                {"tool": "change_statistics", "depends_on": ["validate_inputs"]},
                {"tool": "compare_evidence", "depends_on": ["run_rs_vlm", "change_statistics"]},
                {"tool": "generate_response", "depends_on": ["compare_evidence"]}
            ],
            "vegetation_change": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]},
                {"tool": "ndvi_delta", "depends_on": ["validate_inputs"]},
                {"tool": "change_statistics", "depends_on": ["validate_inputs"]},
                {"tool": "compare_evidence", "depends_on": ["run_rs_vlm", "ndvi_delta", "change_statistics"]},
                {"tool": "generate_response", "depends_on": ["compare_evidence"]}
            ],
            "built_up_change": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]},
                {"tool": "ndbi_delta", "depends_on": ["validate_inputs"]},
                {"tool": "change_statistics", "depends_on": ["validate_inputs"]},
                {"tool": "compare_evidence", "depends_on": ["run_rs_vlm", "ndbi_delta", "change_statistics"]},
                {"tool": "generate_response", "depends_on": ["compare_evidence"]}
            ],
            "optical_sar_cross_check": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]},
                {"tool": "change_statistics", "depends_on": ["validate_inputs"]},
                {"tool": "sar_change", "depends_on": ["validate_inputs"]},
                {"tool": "compare_evidence", "depends_on": ["run_rs_vlm", "change_statistics", "sar_change"]},
                {"tool": "generate_response", "depends_on": ["compare_evidence"]}
            ],
            "spatial_measurement": [
                {"tool": "validate_inputs", "depends_on": []},
                {"tool": "change_statistics", "depends_on": ["validate_inputs"]},
                {"tool": "grounding", "depends_on": ["validate_inputs"]},
                {"tool": "area_measurement", "depends_on": ["validate_inputs"]},
                {"tool": "generate_response", "depends_on": ["area_measurement"]}
            ],
            "insufficient_capability": []
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        yaml.dump(config, f)
        path = f.name
    yield path
    os.remove(path)

@pytest.fixture
def mock_registry():
    registry = MagicMock()
    app = MagicMock()
    def mock_get_tool(name):
        m = MagicMock()
        m.name = name
        return m
    registry.get_tool.side_effect = mock_get_tool
    app.is_applicable = True
    app.reason = ""
    registry.check_applicability.return_value = (True, "")
    return registry

@pytest.fixture
def mock_context():
    return MagicMock()

# --- 1-9: All task types ---

def test_plan_single_image_vqa(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    plan = planner.create_plan(q, mock_context)
    assert plan.status == WorkflowStatus.PLANNED

def test_plan_captioning(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "captioning"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_plan_grounding(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "grounding"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_plan_bi_temporal_change(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "bi_temporal_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_plan_vegetation_change(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "vegetation_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_plan_built_up_change(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "built_up_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_plan_optical_sar_cross_check(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "optical_sar_cross_check"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_plan_spatial_measurement_grounding(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "spatial_measurement"; q.intent = "locate"
    plan = planner.create_plan(q, mock_context)
    assert "grounding" in [s.tool for s in plan.steps]
    assert "change_statistics" not in [s.tool for s in plan.steps]

def test_plan_spatial_measurement_change_stats(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "spatial_measurement"; q.intent = "other"; q.claim = False
    plan = planner.create_plan(q, mock_context)
    assert "change_statistics" in [s.tool for s in plan.steps]
    assert "grounding" not in [s.tool for s in plan.steps]

# --- 10-18: Capability tests ---

def test_cap_ndvi_available(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "vegetation_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_cap_ndvi_unavailable(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "Missing bands"
    mock_registry.check_applicability.side_effect = lambda t_def, c: (False, app.reason) if getattr(t_def, "name", "") == "ndvi_delta" else (True, "")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "vegetation_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_cap_missing_bands(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "Missing optical bands"
    mock_registry.check_applicability.side_effect = lambda t_def, c: (False, app.reason) if getattr(t_def, "name", "") == "change_statistics" else (True, "")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "optical_sar_cross_check"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_cap_sar_unavailable(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "No SAR"
    mock_registry.check_applicability.side_effect = lambda t_def, c: (False, app.reason) if getattr(t_def, "name", "") == "sar_change" else (True, "")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "optical_sar_cross_check"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_cap_one_observation(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "Need 2 obs"
    mock_registry.check_applicability.side_effect = lambda t_def, c: (False, app.reason) if getattr(t_def, "name", "") == "change_statistics" else (True, "")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "bi_temporal_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_cap_two_observations(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "bi_temporal_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_cap_registration_unknown(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "Reg unknown"
    mock_registry.check_applicability.side_effect = lambda t_def, c: (False, app.reason) if getattr(t_def, "name", "") == "validate_inputs" else (True, "")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "bi_temporal_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_cap_registration_failed(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "Reg failed"
    mock_registry.check_applicability.side_effect = lambda t_def, c: (False, app.reason) if getattr(t_def, "name", "") == "validate_inputs" else (True, "")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "built_up_change"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_cap_insufficient_capability_task(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "insufficient_capability"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

# --- 19-24: Dependency tests ---

def test_dep_valid_graph(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "bi_temporal_change"
    plan = planner.create_plan(q, mock_context)
    assert plan.status == WorkflowStatus.PLANNED
    assert plan.steps[1].depends_on == [0]

def test_dep_missing_dep(temp_config, mock_registry, mock_context):
    import yaml
    with open(temp_config, "w") as f: yaml.dump({"workflows": {"single_image_vqa": [{"tool": "validate_inputs", "depends_on": ["missing_tool"]}]}}, f)
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_dep_circular_dep(temp_config, mock_registry, mock_context):
    import yaml
    with open(temp_config, "w") as f: yaml.dump({"workflows": {"single_image_vqa": [{"tool": "run_rs_vlm", "depends_on": ["validate_inputs"]}, {"tool": "validate_inputs", "depends_on": ["run_rs_vlm"]}]}}, f)
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_dep_duplicate_step(temp_config, mock_registry, mock_context):
    import yaml
    with open(temp_config, "w") as f: yaml.dump({"workflows": {"single_image_vqa": [{"tool": "validate_inputs", "depends_on": []}, {"tool": "validate_inputs", "depends_on": []}]}}, f)
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.PLANNED

def test_dep_unavailable_tool(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "Unavailable"
    mock_registry.check_applicability.return_value = (False, "Unavailable")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

def test_dep_unknown_tool(temp_config, mock_registry, mock_context):
    import yaml
    with open(temp_config, "w") as f: yaml.dump({"workflows": {"single_image_vqa": [{"tool": "unknown_tool", "depends_on": []}]}}, f)
    mock_registry.get_tool.side_effect = lambda n: None if n == "unknown_tool" else MagicMock(name=n)
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    assert planner.create_plan(q, mock_context).status == WorkflowStatus.FAILED

# --- 25-30: Behavior tests ---

def test_beh_agentic_routing(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "spatial_measurement"; q.intent = "locate"
    plan = planner.create_plan(q, mock_context)
    assert "grounding" in [s.tool for s in plan.steps]

def test_beh_determinism(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "bi_temporal_change"
    plans = [planner.create_plan(q, mock_context).steps for _ in range(10)]
    for p in plans[1:]:
        assert len(p) == len(plans[0])
        assert [s.tool for s in p] == [s.tool for s in plans[0]]

def test_beh_candidate_tools_ignored_for_order(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "bi_temporal_change"
    plan = planner.create_plan(q, mock_context)
    assert plan.steps[0].tool == "validate_inputs"
    assert plan.steps[1].tool == "run_rs_vlm"

def test_beh_fallback_on_tool_failure(temp_config, mock_registry, mock_context):
    app = MagicMock(); app.is_applicable = False; app.reason = "Fallback reason test"
    mock_registry.check_applicability.return_value = (False, "Fallback reason test")
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    plan = planner.create_plan(q, mock_context)
    assert plan.status == WorkflowStatus.FAILED
    assert plan.fallback == "Fallback reason test"

def test_beh_empty_workflow(temp_config, mock_registry, mock_context):
    import yaml
    with open(temp_config, "w") as f: yaml.dump({"workflows": {"single_image_vqa": []}}, f)
    planner = ConstrainedPlanner(mock_registry, temp_config)
    q = MagicMock(); q.task = "single_image_vqa"
    plan = planner.create_plan(q, mock_context)
    assert plan.status == WorkflowStatus.PLANNED
    assert len(plan.steps) == 0

def test_beh_invalid_dag_bounds(temp_config, mock_registry, mock_context):
    planner = ConstrainedPlanner(mock_registry, temp_config)
    assert not planner._validate_dag([MagicMock(depends_on=[1]), MagicMock(depends_on=[])])
    assert not planner._validate_dag([MagicMock(depends_on=[-1])])

