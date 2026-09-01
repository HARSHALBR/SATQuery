"""Phase 12 Unified Mock End-to-End Workflow Integration Tests."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from schemas.query import QueryInput, Modality, ObservationInput, ObservationRole, ImageMetadata
from schemas.response import EvidenceStatus, FinalResponse
from agents.mock_tools import MockScenario
from backend.services.orchestrator import SATQueryOrchestrator


client = TestClient(app)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optical_obs(obs_id="obs_1", role=ObservationRole.T1) -> ObservationInput:
    return ObservationInput(
        observation_id=obs_id,
        image_path="/path/to/img.tif",
        role=role,
        metadata=ImageMetadata(modality=Modality.OPTICAL, bands=["red", "nir"])
    )

def _make_sar_obs(obs_id="obs_sar", role=ObservationRole.SAR_T1) -> ObservationInput:
    return ObservationInput(
        observation_id=obs_id,
        image_path="/path/to/sar.tif",
        role=role,
        metadata=ImageMetadata(modality=Modality.SAR, bands=["vv", "vh"])
    )

def _build_query(query_text: str, obs_list: list, dev_scenario: str = None) -> dict:
    q = {
        "query": query_text,
        "observations": [o.model_dump() for o in obs_list],
    }
    if dev_scenario:
        q["metadata"] = {"dev_scenario": dev_scenario}
    return q


# ---------------------------------------------------------------------------
# ACCEPTANCE SCENARIO A — SUPPORTED
# ---------------------------------------------------------------------------
def test_e2e_scenario_a_supported():
    """Verify SUPPORTED workflow end-to-end."""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1", ObservationRole.T1), _make_optical_obs("o2", ObservationRole.T2)],
        MockScenario.VEGETATION_DECREASE.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == EvidenceStatus.SUPPORTED.value
    assert data["task"] == "vegetation_change"
    
    # Evidence verification
    assert len(data["evidence"]) >= 2
    evidence_types = {e["type"] for e in data["evidence"]}
    assert "change_quantification" in evidence_types
    assert "vegetation_change" in evidence_types
    
    # Trace verification
    assert data["trace_id"] is not None
    assert len(data["execution_trace"]) > 0
    
    # Answer verification (human-readable + supporting IDs)
    answer = data["answer"]
    assert "supported" in answer.lower()
    assert "Reason:" in answer
    assert "Supporting Evidence IDs:" in answer
    assert "Conflicting Evidence IDs:" not in answer
    
    # TraceStore integration verification
    # Need to instantiate the orchestrator directly for internal state verification
    orch = SATQueryOrchestrator(MockScenario.VEGETATION_DECREASE)
    orch_resp = orch.analyze(QueryInput(**payload))
    trace = orch.trace_store.get(orch_resp.trace_id)
    assert trace is not None
    assert trace.trace_id == orch_resp.trace_id


# ---------------------------------------------------------------------------
# ACCEPTANCE SCENARIO B — UNCERTAIN
# ---------------------------------------------------------------------------
def test_e2e_scenario_b_uncertain():
    """Verify UNCERTAIN workflow end-to-end with conflicting evidence."""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1", ObservationRole.T1), _make_optical_obs("o2", ObservationRole.T2)],
        MockScenario.CONFLICTING_EVIDENCE.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == EvidenceStatus.UNCERTAIN.value
    
    answer = data["answer"]
    assert "conflicting" in answer.lower() or "uncertain" in answer.lower()
    assert "Conflicting Evidence IDs:" in answer
    
    # Must have both VLM and NDVI in evidence output
    evidence_types = {e["type"] for e in data["evidence"]}
    assert "change_quantification" in evidence_types
    assert "vegetation_change" in evidence_types
    
    # Trace available
    assert data["trace_id"] is not None
    assert len(data["execution_trace"]) > 0


# ---------------------------------------------------------------------------
# ACCEPTANCE SCENARIO C — INSUFFICIENT
# ---------------------------------------------------------------------------
def test_e2e_scenario_c_insufficient():
    """Verify LOW_QUALITY translates to INSUFFICIENT with quality notes."""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1"), _make_optical_obs("o2")],
        MockScenario.LOW_QUALITY.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == EvidenceStatus.INSUFFICIENT.value
    
    # Quality info preserved in evidence
    low_qual_evidence = [e for e in data["evidence"] if e["quality"]["valid_pixel_fraction"] < 0.5]
    assert len(low_qual_evidence) > 0
    
    # Answer and limitations are coherent
    assert "insufficient" in data["answer"].lower()
    assert len(data["limitations"]) > 0
    assert data["status"] != EvidenceStatus.SUPPORTED.value


# ---------------------------------------------------------------------------
# ACCEPTANCE SCENARIO D — CAPABILITY FAILURE
# ---------------------------------------------------------------------------
def test_e2e_scenario_d_capability_failure():
    """Verify missing modalities fail gracefully during planning."""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_sar_obs("s1"), _make_sar_obs("s2")],
        MockScenario.NORMAL.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Must be INSUFFICIENT, not HTTP 500, not SUPPORTED
    assert data["status"] == EvidenceStatus.INSUFFICIENT.value
    
    # No false vegetation evidence
    assert len(data["evidence"]) == 0
    
    # Limitations explain why
    assert len(data["limitations"]) > 0
    assert "missing" in data["limitations"][0].lower() or "optical" in data["limitations"][0].lower()


# ---------------------------------------------------------------------------
# ISOLATION & DETERMINISM
# ---------------------------------------------------------------------------
def test_e2e_request_isolation():
    """Verify independent sequential requests do not leak evidence or traces."""
    # Request A - SUPPORTED
    payload_a = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1"), _make_optical_obs("o2")],
        MockScenario.VEGETATION_DECREASE.value
    )
    resp_a = client.post("/api/v1/analyze", json=payload_a).json()
    
    # Request B - UNCERTAIN
    payload_b = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o3"), _make_optical_obs("o4")],
        MockScenario.CONFLICTING_EVIDENCE.value
    )
    resp_b = client.post("/api/v1/analyze", json=payload_b).json()
    
    assert resp_a["trace_id"] != resp_b["trace_id"]
    
    # Extract evidence IDs
    ev_ids_a = {e["evidence_id"] for e in resp_a["evidence"]}
    ev_ids_b = {e["evidence_id"] for e in resp_b["evidence"]}
    
    # Strictly isolated
    assert ev_ids_a.isdisjoint(ev_ids_b)

def test_e2e_determinism():
    """Verify same request yields same semantic output over multiple runs."""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1"), _make_optical_obs("o2")],
        MockScenario.VEGETATION_DECREASE.value
    )
    
    r1 = client.post("/api/v1/analyze", json=payload).json()
    r2 = client.post("/api/v1/analyze", json=payload).json()
    r3 = client.post("/api/v1/analyze", json=payload).json()
    
    assert r1["status"] == r2["status"] == r3["status"] == EvidenceStatus.SUPPORTED.value
    assert r1["task"] == r2["task"] == r3["task"] == "vegetation_change"
    
    # Semantic answer must be identical (ignoring dynamic evidence IDs)
    assert r1["answer"].split('\n')[0] == r2["answer"].split('\n')[0] == r3["answer"].split('\n')[0]
    
    # Length of evidence arrays should match deterministically
    assert len(r1["evidence"]) == len(r2["evidence"]) == len(r3["evidence"])
    
    # Trace IDs will differ per run (uuids)
    assert r1["trace_id"] != r2["trace_id"]

def test_e2e_malformed_request_rejected():
    """Verify FastAPI correctly blocks invalid schemas."""
    response = client.post("/api/v1/analyze", json={"bad_field": 123})
    assert response.status_code == 422
