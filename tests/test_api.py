"""Tests for Phase 11 FastAPI Backend."""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from schemas.query import QueryInput, Modality, ObservationInput, ObservationRole, ImageMetadata
from schemas.response import EvidenceStatus, FinalResponse
from agents.mock_tools import MockScenario
from backend.services.orchestrator import SATQueryOrchestrator
from unittest.mock import patch


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
# Basic API & Health
# ---------------------------------------------------------------------------

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "satquery-ai"}

def test_missing_query():
    # Body completely missing
    response = client.post("/api/v1/analyze")
    assert response.status_code == 422  # Validation Error

def test_invalid_request():
    # Missing required 'query' string field
    response = client.post("/api/v1/analyze", json={"observations": []})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# End-to-End Scenarios
# ---------------------------------------------------------------------------

def test_analyze_supported():
    """Scenario 1: SUPPORTED"""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1", ObservationRole.T1), _make_optical_obs("o2", ObservationRole.T2)],
        MockScenario.VEGETATION_DECREASE.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    
    # Domain successful
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == EvidenceStatus.SUPPORTED.value
    assert len(data["evidence"]) > 0
    assert len(data["execution_trace"]) > 0
    assert data["trace_id"] is not None

def test_analyze_uncertain():
    """Scenario 2: CONFLICTING_EVIDENCE -> UNCERTAIN"""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1"), _make_optical_obs("o2")],
        MockScenario.CONFLICTING_EVIDENCE.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == EvidenceStatus.UNCERTAIN.value

def test_analyze_insufficient():
    """Scenario 3: LOW_QUALITY -> INSUFFICIENT"""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1"), _make_optical_obs("o2")],
        MockScenario.LOW_QUALITY.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    
    # HTTP 200 but domain INSUFFICIENT
    assert response.status_code == 200
    assert response.json()["status"] == EvidenceStatus.INSUFFICIENT.value


# ---------------------------------------------------------------------------
# Failure Handling
# ---------------------------------------------------------------------------

def test_sar_only_capability_failure():
    """Scenario 4: Vegetation query with SAR-only context fails applicability gracefully."""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_sar_obs("s1"), _make_sar_obs("s2")],
        MockScenario.NORMAL.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == EvidenceStatus.INSUFFICIENT.value
    assert "missing" in data["limitations"][0].lower() or "optical" in data["limitations"][0].lower() or "missing required" in data["limitations"][0].lower()

def test_execution_failure_handling():
    """TOOL_FAILURE scenario causes execution engine to fail."""
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1"), _make_optical_obs("o2")],
        MockScenario.TOOL_FAILURE.value
    )
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == EvidenceStatus.INSUFFICIENT.value
    assert "error" in response.json()["limitations"][0].lower() or "execution failed" in response.json()["answer"].lower()

@patch('backend.routes.analyze.SATQueryOrchestrator.analyze')
def test_unexpected_exception_handling(mock_analyze):
    """Ensure unexpected internal errors map to HTTP 500 without leaking stack traces."""
    mock_analyze.side_effect = RuntimeError("Database exploded")
    payload = _build_query("Test?", [])
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 500
    assert "Database exploded" not in response.json()["detail"]


# ---------------------------------------------------------------------------
# State Isolation & Determinism
# ---------------------------------------------------------------------------

def test_deterministic_repeated_request():
    payload = _build_query(
        "Has vegetation decreased?", 
        [_make_optical_obs("o1"), _make_optical_obs("o2")],
        MockScenario.VEGETATION_DECREASE.value
    )
    r1 = client.post("/api/v1/analyze", json=payload)
    r2 = client.post("/api/v1/analyze", json=payload)
    
    # trace_ids will differ (uuid per request), but core analysis fields match
    d1, d2 = r1.json(), r2.json()
    assert d1["status"] == d2["status"]
    # Answer text includes generated IDs, so compare just the first line (the semantic part)
    assert d1["answer"].split('\n')[0] == d2["answer"].split('\n')[0]
    assert len(d1["evidence"]) == len(d2["evidence"])

def test_request_isolation():
    """Verify that multiple orchestrator instantiations do not share state."""
    orch1 = SATQueryOrchestrator()
    orch2 = SATQueryOrchestrator()
    
    q_input = QueryInput(**_build_query("Has vegetation decreased?", [_make_optical_obs(), _make_optical_obs()]))
    
    orch1.analyze(q_input)
    assert len(orch1.trace_store.list()) == 1
    # orch2 should be clean
    assert len(orch2.trace_store.list()) == 0


# ---------------------------------------------------------------------------
# Internal TraceStore Integration Check
# ---------------------------------------------------------------------------

def test_trace_store_integration_retrieval():
    """Verify that the returned trace_id can be used internally to retrieve the trace."""
    orchestrator = SATQueryOrchestrator(MockScenario.VEGETATION_DECREASE)
    q_input = QueryInput(**_build_query("Has vegetation decreased?", [_make_optical_obs(), _make_optical_obs()]))
    
    response: FinalResponse = orchestrator.analyze(q_input)
    
    # Retrievable from the store
    trace = orchestrator.trace_store.get(response.trace_id)
    assert trace is not None
    assert trace.trace_id == response.trace_id
    assert len(trace.steps) == len(response.execution_trace)
