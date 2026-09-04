"""Analyze endpoint for GeoVision."""

import os
from fastapi import APIRouter, HTTPException
from schemas.query import QueryInput
from schemas.response import FinalResponse
from agents.mock_tools import MockScenario
from backend.services.orchestrator import GeoVisionOrchestrator
from agents.real_runner import RealToolRunner

router = APIRouter()

@router.post("/analyze", response_model=FinalResponse)
def analyze_query(query: QueryInput):
    """
    Process a natural language query over satellite imagery observations.
    
    Returns a FinalResponse containing the extracted answer, evidence status,
    collected evidence, and execution trace.
    """
    # Extract optional test scenario from metadata
    scenario = MockScenario.NORMAL
    if query.metadata and "dev_scenario" in query.metadata:
        try:
            scenario_val = query.metadata["dev_scenario"]
            if isinstance(scenario_val, str):
                scenario_val = scenario_val.upper()
            scenario = MockScenario(scenario_val)
        except ValueError:
            # Safe fallback if scenario is invalid
            pass

    # Check execution mode configuration
    # If a dev_scenario is explicitly specified, use MockToolRunner for backward compatibility
    has_dev_scenario = bool(query.metadata and query.metadata.get("dev_scenario"))
    has_uploaded_obs = any("demo_uploads" in (obs.image_path or "") for obs in query.observations)
    mock_rs = os.getenv("MOCK_RS_TOOLS", "false").lower() == "true"
    
    # Explicitly uploaded observations must ALWAYS use RealToolRunner
    if has_dev_scenario or (mock_rs and not has_uploaded_obs):
        runner = None
    else:
        runner = RealToolRunner(observations=query.observations, query_text=query.query)

    # Instantiate orchestrator for request isolation
    orchestrator = GeoVisionOrchestrator(scenario=scenario, runner=runner)
    
    try:
        response = orchestrator.analyze(query)
        return response
    except Exception as e:
        # Unexpected internal errors map to HTTP 500
        # Expected domain failures (INSUFFICIENT) are handled safely inside analyze()
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal error occurred during analysis."
        )
