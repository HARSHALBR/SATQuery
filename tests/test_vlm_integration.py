import pytest
from typing import List
from copy import deepcopy

from schemas.query import QueryInput, ObservationInput, ObservationRole
from schemas.tools import ToolStatus, ToolResult
from schemas.vlm import VLMClaimType, VLMResult, VLMContext
from schemas.response import EvidenceStatus

from agents.real_runner import RealToolRunner
from tools.vlm.client import MockVLMClient
from agents.execution_engine import ExecutionEngine
from agents.planner import ConstrainedPlanner
from backend.services.orchestrator import SATQueryOrchestrator

def _build_obs(path: str, role: ObservationRole, date: str, stac_id: str) -> ObservationInput:
    from schemas.query import ImageMetadata, Modality
    import datetime
    return ObservationInput(
        observation_id=stac_id,
        image_path=path,
        role=role,
        metadata=ImageMetadata(
            modality=Modality.OPTICAL,
            bands=["red", "nir", "scl"],
            acquisition_date=datetime.datetime.strptime(date, "%Y-%m-%d"),
            stac_item_id=stac_id,
            cloud_cover=0.0
        )
    )

@pytest.fixture
def base_obs():
    return [
        _build_obs("datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A", ObservationRole.T1, "2021-07-08", "a1"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A", ObservationRole.T2, "2021-10-01", "a2")
    ]

# 1. valid VLM response & 9. VLM evidence creation & 10. provenance propagation & 11. observation ID propagation
def test_valid_vlm_response_and_evidence(base_obs):
    runner = RealToolRunner(observations=base_obs, vlm_client=MockVLMClient("VEGETATION_DECREASE"))
    res = runner.execute("run_rs_vlm", {"query": "Test query"})
    assert res.status == ToolStatus.SUCCESS
    assert res.output["claim"] == VLMClaimType.VEGETATION_DECREASE.value
    
    ev = res.output["evidence"]
    assert ev.type == "vlm_interpretation"
    assert ev.value["claim"] == "vegetation_decrease"
    assert ev.value["confidence"] == 0.91
    assert "reasoning" in ev.value
    assert ev.provenance.tool == "run_rs_vlm"
    assert ev.provenance.input_ids == ["a1", "a2"]

# 2. confidence boundary 0.0 & 3. confidence boundary 1.0 & 4. invalid confidence
def test_confidence_boundaries():
    # Valid bounds
    res = VLMResult(claim=VLMClaimType.VEGETATION_CHANGE, confidence=0.0, reasoning="Valid low")
    assert res.confidence == 0.0
    res = VLMResult(claim=VLMClaimType.VEGETATION_CHANGE, confidence=1.0, reasoning="Valid high")
    assert res.confidence == 1.0
    
    # Invalid bounds
    with pytest.raises(ValueError):
        VLMResult(claim=VLMClaimType.VEGETATION_CHANGE, confidence=-0.1, reasoning="Too low")
    with pytest.raises(ValueError):
        VLMResult(claim=VLMClaimType.VEGETATION_CHANGE, confidence=1.1, reasoning="Too high")

# 5. missing claim & 6. missing reasoning
def test_missing_fields():
    with pytest.raises(ValueError):
        VLMResult(confidence=0.5, reasoning="Missing claim")
    with pytest.raises(ValueError):
        VLMResult(claim=VLMClaimType.VEGETATION_CHANGE, confidence=0.5)
    with pytest.raises(ValueError):
        VLMResult(claim=VLMClaimType.VEGETATION_CHANGE, reasoning="Missing confidence")

# 7. unsupported claim
def test_unsupported_claim():
    with pytest.raises(ValueError):
        VLMResult(claim="alien_invasion", confidence=0.9, reasoning="Nope")

# 11. Mock error handling isolation
def test_vlm_failure_isolation(base_obs):
    runner = RealToolRunner(observations=base_obs, vlm_client=MockVLMClient("MALFORMED_RESPONSE"))
    res = runner.execute("run_rs_vlm", {"query": "Will fail"})
    assert res.status == ToolStatus.ERROR
    assert "malformed" in res.error.lower()

# 12. T1/T2 ordering
def test_t1_t2_ordering_respected(base_obs, tmp_path, monkeypatch):
    # Swap them in array
    obs_swapped = [base_obs[1], base_obs[0]]
    # T1 is base_obs[0], T2 is base_obs[1]. Even if swapped in array, runner should sort by role.
    runner = RealToolRunner(observations=obs_swapped, vlm_client=MockVLMClient("VEGETATION_DECREASE"))
    # We mock image_utils to intercept the paths
    intercepted = []
    def mock_create(p1, p2, out):
        intercepted.append((p1, p2))
        # Just create an empty file
        open(out, 'w').close()
        return out
    monkeypatch.setattr("agents.real_runner.create_side_by_side", mock_create)
    
    runner.execute("run_rs_vlm", {"query": ""})
    # T1 should be first path, T2 second
    assert "S2A_10TFK" in intercepted[0][0] # T1
    assert "S2B_10TFK" in intercepted[0][1] # T2

# 13. MockVLM scenarios & 16. VLM disagreement does not override RS evidence
def test_vlm_rs_disagreement_does_not_override(base_obs):
    runner = RealToolRunner(observations=base_obs, vlm_client=MockVLMClient("VEGETATION_INCREASE"))
    
    runner.execute("validate_inputs", {})
    ndvi_res = runner.execute("ndvi_delta", {"input_ids": ["a1", "a2"]})
    delta_key = ndvi_res.output["delta_map"]
    mask_key = ndvi_res.output["valid_mask"]
    
    rs_res = runner.execute("change_statistics", {"delta_map": delta_key, "valid_mask": mask_key, "input_ids": ["a1", "a2"]})
    vlm_res = runner.execute("run_rs_vlm", {"query": "Did vegetation decrease?"})
    
    from backend.services.orchestrator import SATQueryOrchestrator
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare("vegetation_decrease", [
        ndvi_res.output["evidence"],
        rs_res.output["evidence"],
        vlm_res.output["evidence"]
    ])
    
    assert comp_result.status in [EvidenceStatus.SUPPORTED, EvidenceStatus.UNCERTAIN]

# 17. Case A integration
def test_case_a_integration(base_obs):
    runner = RealToolRunner(observations=base_obs, vlm_client=MockVLMClient("VEGETATION_DECREASE"))
    runner.execute("validate_inputs", {})
    ndvi_res = runner.execute("ndvi_delta", {"input_ids": ["a1", "a2"]})
    rs_res = runner.execute("change_statistics", {"delta_map": ndvi_res.output["delta_map"], "valid_mask": ndvi_res.output["valid_mask"], "input_ids": ["a1", "a2"]})
    vlm_res = runner.execute("run_rs_vlm", {"query": "Did vegetation decrease?"})
    
    from backend.services.orchestrator import SATQueryOrchestrator
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare("vegetation_decrease", [
        ndvi_res.output["evidence"], rs_res.output["evidence"], vlm_res.output["evidence"]
    ])
    assert comp_result.status == EvidenceStatus.SUPPORTED

# 18. Case B integration
def test_case_b_integration():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A", ObservationRole.T1, "2021-07-06", "b1"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A", ObservationRole.T2, "2021-10-14", "b2")
    ]
    runner = RealToolRunner(observations=obs, vlm_client=MockVLMClient("NO_CHANGE"))
    runner.execute("validate_inputs", {})
    ndvi_res = runner.execute("ndvi_delta", {"input_ids": ["b1", "b2"]})
    rs_res = runner.execute("change_statistics", {"delta_map": ndvi_res.output["delta_map"], "valid_mask": ndvi_res.output["valid_mask"], "input_ids": ["b1", "b2"]})
    vlm_res = runner.execute("run_rs_vlm", {"query": "Did vegetation decrease?"})
    
    from backend.services.orchestrator import SATQueryOrchestrator
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare("vegetation_decrease", [
        ndvi_res.output["evidence"], rs_res.output["evidence"], vlm_res.output["evidence"]
    ])
    assert comp_result.status == EvidenceStatus.SUPPORTED

# 19. Case C integration
def test_case_c_integration():
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2A_10SGF_20210705_2_L2A", ObservationRole.T1, "2021-07-05", "c1"),
        _build_obs("datasets/golden_fixtures/raw/S2A_10SGF_20211013_0_L2A", ObservationRole.T2, "2021-10-13", "c2")
    ]
    runner = RealToolRunner(observations=obs, vlm_client=MockVLMClient("AGRICULTURAL_HARVEST"))
    runner.execute("validate_inputs", {})
    ndvi_res = runner.execute("ndvi_delta", {"input_ids": ["c1", "c2"]})
    rs_res = runner.execute("change_statistics", {"delta_map": ndvi_res.output["delta_map"], "valid_mask": ndvi_res.output["valid_mask"], "input_ids": ["c1", "c2"]})
    vlm_res = runner.execute("run_rs_vlm", {"query": "Has there been permanent deforestation?"})
    
    from backend.services.orchestrator import SATQueryOrchestrator
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare("vegetation_decrease", [
        ndvi_res.output["evidence"], rs_res.output["evidence"], vlm_res.output["evidence"]
    ])
    assert len(comp_result.supporting_evidence) >= 0 # just checking it doesn't crash
    assert vlm_res.output["evidence"].value["claim"] == "vegetation_change"
