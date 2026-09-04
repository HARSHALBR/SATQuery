import json
from typing import Protocol, List, Any, Optional
from schemas.vlm import VLMResult, VLMClaimType, VLMContext

class VLMClient(Protocol):
    """Abstract interface for a Vision-Language Model client."""
    def analyze(self, image_paths: List[str], query: str, context: VLMContext) -> VLMResult:
        ...

    def analyze_observation(
        self,
        image_s1: Any,
        image_s2: Any,
        query: str,
        context: Optional[VLMContext] = None,
    ) -> VLMResult:
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

    def analyze_observation(
        self,
        image_s1: Any,
        image_s2: Any,
        query: str,
        context: Optional[VLMContext] = None,
    ) -> VLMResult:
        if self.scenario == "MALFORMED_RESPONSE":
            raise ValueError("VLM returned malformed JSON output.")

        lower = (query or "").lower()
        if "vegetation" in lower or "forest" in lower:
            claim = VLMClaimType.VEGETATION_CHANGE
            reasoning = "Vegetation and canopy features observed across the observation patch."
        elif "urban" in lower or "built-up" in lower or "built up" in lower:
            claim = VLMClaimType.BUILT_UP_CHANGE
            reasoning = "Built-up structures and urban surface features identified in patch."
        elif "water" in lower:
            claim = VLMClaimType.GENERAL_CHANGE
            reasoning = "Water body extent observed in patch."
        else:
            claim = VLMClaimType.GENERAL_CHANGE
            reasoning = f"Semantic scene interpretation for: {query}"

        conf = 0.30 if self.scenario == "LOW_CONFIDENCE" else 0.88
        return VLMResult(
            claim=claim,
            confidence=conf,
            reasoning=reasoning,
        )

