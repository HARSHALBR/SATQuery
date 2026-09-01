"""Analyze endpoint for SATQuery AI."""

import os
from fastapi import APIRouter, HTTPException
from schemas.query import QueryInput
from schemas.response import FinalResponse
from agents.mock_tools import MockScenario
from backend.services.orchestrator import SATQueryOrchestrator
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
    # If a dev_scenario is explicitly specified, always use MockToolRunner for backward compatibility
    has_dev_scenario = query.metadata and "dev_scenario" in query.metadata
    mock_rs = os.getenv("MOCK_RS_TOOLS", "true").lower() == "true"
    if mock_rs or has_dev_scenario:
        runner = None
    else:
        runner = RealToolRunner(observations=query.observations, query_text=query.query)

    # Instantiate orchestrator for request isolation
    orchestrator = SATQueryOrchestrator(scenario=scenario, runner=runner)
    
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
