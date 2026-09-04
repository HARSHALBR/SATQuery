"""Health check endpoint for GeoVision."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    service: str

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Simple health check endpoint."""
    return HealthResponse(status="ok", service="geovision-ai")
