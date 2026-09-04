from enum import Enum
from pydantic import BaseModel, Field

class VLMClaimType(str, Enum):
    """Allowed semantic claims the VLM can return."""
    VEGETATION_CHANGE = "vegetation_change"
    VEGETATION_DECREASE = "vegetation_decrease"
    VEGETATION_INCREASE = "vegetation_increase"
    BUILT_UP_CHANGE = "built_up_change"
    BUILT_UP_DECREASE = "built_up_decrease"
    BUILT_UP_INCREASE = "built_up_increase"
    GENERAL_CHANGE = "general_change"
    SAR_CROSS_CHECK = "sar_cross_check"

class VLMResult(BaseModel):
    """Strict structured output required from the VLM."""
    claim: VLMClaimType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(..., min_length=5, description="Detailed reasoning for the claim")
    
class VLMContext(BaseModel):
    """Contextual information passed to the VLM."""
    query: str
    t1_date: str
    t2_date: str
    region_info: str = ""
