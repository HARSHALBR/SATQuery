import os
import pytest
from tools.vlm.gemini_client import GeminiVLMClient
from schemas.vlm import VLMContext
from schemas.query import ObservationRole

@pytest.mark.skipif(not os.getenv("RUN_REAL_VLM_TESTS"), reason="RUN_REAL_VLM_TESTS not set")
def test_real_gemini_case_a():
    """Case A: Wildfire - Explicit vegetation decrease."""
    client = GeminiVLMClient()
    # We'll just run analyze on the pre-made composites if available, or just the red bands.
    # Actually, we can run it through the RealToolRunner to test the full E2E VLM adapter!
    from agents.real_runner import RealToolRunner
    from tests.test_vlm_integration import _build_obs
    from backend.services.orchestrator import SATQueryOrchestrator
    
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2A_10TFK_20210708_0_L2A", ObservationRole.T1, "2021-07-08", "a1"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TFK_20211001_0_L2A", ObservationRole.T2, "2021-10-01", "a2")
    ]
    
    runner = RealToolRunner(observations=obs, vlm_client=client)
    runner.execute("validate_inputs", {})
    ndvi_res = runner.execute("ndvi_delta", {"input_ids": ["a1", "a2"]})
    rs_res = runner.execute("change_statistics", {"delta_map": ndvi_res.output["delta_map"], "valid_mask": ndvi_res.output["valid_mask"], "input_ids": ["a1", "a2"]})
    vlm_res = runner.execute("run_rs_vlm", {"query": "Has vegetation decreased?"})
    
    assert vlm_res.status.name == "SUCCESS", f"VLM API Error: {vlm_res.error}"
    
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare("vegetation_decrease", [
        ndvi_res.output["evidence"], rs_res.output["evidence"], vlm_res.output["evidence"]
    ])
    
    print(f"Case A Real VLM output: {vlm_res.output['evidence'].value}")
    print(f"Case A Comparator Status: {comp_result.status}")

@pytest.mark.skipif(not os.getenv("RUN_REAL_VLM_TESTS"), reason="RUN_REAL_VLM_TESTS not set")
def test_real_gemini_case_b():
    """Case B: Stable forest."""
    client = GeminiVLMClient()
    from agents.real_runner import RealToolRunner
    from tests.test_vlm_integration import _build_obs
    from backend.services.orchestrator import SATQueryOrchestrator
    
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20210706_1_L2A", ObservationRole.T1, "2021-07-06", "b1"),
        _build_obs("datasets/golden_fixtures/raw/S2B_10TDL_20211014_1_L2A", ObservationRole.T2, "2021-10-14", "b2")
    ]
    
    runner = RealToolRunner(observations=obs, vlm_client=client)
    runner.execute("validate_inputs", {})
    ndvi_res = runner.execute("ndvi_delta", {"input_ids": ["b1", "b2"]})
    rs_res = runner.execute("change_statistics", {"delta_map": ndvi_res.output["delta_map"], "valid_mask": ndvi_res.output["valid_mask"], "input_ids": ["b1", "b2"]})
    vlm_res = runner.execute("run_rs_vlm", {"query": "Has vegetation decreased?"})
    
    assert vlm_res.status.name == "SUCCESS", f"VLM API Error: {vlm_res.error}"
    
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare("vegetation_decrease", [
        ndvi_res.output["evidence"], rs_res.output["evidence"], vlm_res.output["evidence"]
    ])
    
    print(f"Case B Real VLM output: {vlm_res.output['evidence'].value}")
    print(f"Case B Comparator Status: {comp_result.status}")

@pytest.mark.skipif(not os.getenv("RUN_REAL_VLM_TESTS"), reason="RUN_REAL_VLM_TESTS not set")
def test_real_gemini_case_c():
    """Case C: Agricultural confounder."""
    client = GeminiVLMClient()
    from agents.real_runner import RealToolRunner
    from tests.test_vlm_integration import _build_obs
    from backend.services.orchestrator import SATQueryOrchestrator
    
    obs = [
        _build_obs("datasets/golden_fixtures/raw/S2A_10SGF_20210705_2_L2A", ObservationRole.T1, "2021-07-05", "c1"),
        _build_obs("datasets/golden_fixtures/raw/S2A_10SGF_20211013_0_L2A", ObservationRole.T2, "2021-10-13", "c2")
    ]
    
    runner = RealToolRunner(observations=obs, vlm_client=client)
    runner.execute("validate_inputs", {})
    ndvi_res = runner.execute("ndvi_delta", {"input_ids": ["c1", "c2"]})
    rs_res = runner.execute("change_statistics", {"delta_map": ndvi_res.output["delta_map"], "valid_mask": ndvi_res.output["valid_mask"], "input_ids": ["c1", "c2"]})
    vlm_res = runner.execute("run_rs_vlm", {"query": "Has there been permanent deforestation?"})
    
    assert vlm_res.status.name == "SUCCESS", f"VLM API Error: {vlm_res.error}"
    
    orch = SATQueryOrchestrator(runner=runner)
    comp_result = orch.comparator.compare("vegetation_decrease", [
        ndvi_res.output["evidence"], rs_res.output["evidence"], vlm_res.output["evidence"]
    ])
    
    print(f"Case C Real VLM output: {vlm_res.output['evidence'].value}")
    print(f"Case C Comparator Status: {comp_result.status}")
