import json
from typing import Protocol, List
from schemas.vlm import VLMResult, VLMClaimType, VLMContext

class VLMClient(Protocol):
    """Abstract interface for a Vision-Language Model client."""
    def analyze(self, image_paths: List[str], query: str, context: VLMContext) -> VLMResult:
        ...

class MockVLMClient(VLMClient):
    """Deterministic Mock VLM for testing scenarios without external API calls."""
    
    def __init__(self, scenario: str = "VEGETATION_DECREASE"):
        self.scenario = scenario.upper()

    def analyze(self, image_paths: List[str], query: str, context: VLMContext) -> VLMResult:
        if self.scenario == "VEGETATION_DECREASE":
            return VLMResult(
                claim=VLMClaimType.VEGETATION_DECREASE,
                confidence=0.91,
                reasoning="Visible reduction in vegetation consistent with a severe disturbance like a fire scar."
            )
        elif self.scenario == "VEGETATION_INCREASE":
            return VLMResult(
                claim=VLMClaimType.VEGETATION_INCREASE,
                confidence=0.85,
                reasoning="Visible greening and growth in the vegetation."
            )
        elif self.scenario == "CONFLICTING_EVIDENCE":
            # For testing comparator logic where RS says one thing and VLM says another
            return VLMResult(
                claim=VLMClaimType.VEGETATION_INCREASE,
                confidence=0.88,
                reasoning="VLM hallucinates or observes growth conflicting with NDVI."
            )
        elif self.scenario == "AGRICULTURAL_HARVEST":
            return VLMResult(
                claim=VLMClaimType.VEGETATION_CHANGE,
                confidence=0.85,
                reasoning="agricultural harvest"
            )
        elif self.scenario == "NO_CHANGE":
            return VLMResult(
                claim=VLMClaimType.VEGETATION_CHANGE,
                confidence=0.9,
                reasoning="No meaningful change detected."
            )
        elif self.scenario == "LOW_CONFIDENCE":
            return VLMResult(
                claim=VLMClaimType.VEGETATION_DECREASE,
                confidence=0.30, # Low confidence boundary
                reasoning="Heavy cloud cover obscures most of the area, making interpretation difficult."
            )
        elif self.scenario == "MALFORMED_RESPONSE":
            # Simulate a provider returning bad JSON that crashes the Pydantic parser
            raise ValueError("VLM returned malformed JSON output.")
        else:
            return VLMResult(
                claim=VLMClaimType.GENERAL_CHANGE,
                confidence=0.75,
                reasoning=f"Generic change detected for scenario: {self.scenario}"
            )
