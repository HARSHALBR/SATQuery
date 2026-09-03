"""Response-related schemas for SATQuery AI.

Defines the evidence status, comparison result, and the final response
returned to the user.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from schemas.evidence import EvidenceRecord
from schemas.trace import TraceStep


class EvidenceStatus(str, Enum):
    """Final evidence status for a query."""

    SUPPORTED = "SUPPORTED"
    UNCERTAIN = "UNCERTAIN"
    INSUFFICIENT = "INSUFFICIENT"


class ComparisonResult(BaseModel):
    """Result of comparing a VLM claim against collected evidence."""

    status: EvidenceStatus
    reason: str
    supporting_evidence: list[str] = Field(default_factory=list)
    conflicting_evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class BoundsWGS84(BaseModel):
    """Geographic bounding box in WGS84 (EPSG:4326)."""
    west: float
    south: float
    east: float
    north: float


class GeoCenter(BaseModel):
    """Geographic center point."""
    lat: float
    lon: float


class SpatialChangeRegion(BaseModel):
    """A region extracted from a deterministic raster change mask."""

    region_id: str
    change_type: Optional[str] = None
    geometry: Optional[dict] = None
    changed_pixel_count: int
    area_m2: Optional[float] = None
    area_ha: Optional[float] = None
    change_percent: Optional[float] = None
    metrics: dict = Field(default_factory=dict)


class SpatialEvidence(BaseModel):
    """Spatial extent and geographic metadata for the analyzed region."""
    available: bool
    crs: Optional[str] = None
    bounds_wgs84: Optional[BoundsWGS84] = None
    center: Optional[GeoCenter] = None
    reason: Optional[str] = None   # Set when available=False
    spatial_grounding: Optional[str] = None
    observation_extent: Optional[dict] = None
    change_regions: list[SpatialChangeRegion] = Field(default_factory=list)


class FinalResponse(BaseModel):
    """Final response returned to the user."""

    trace_id: str
    task: str
    answer: str
    status: EvidenceStatus
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    execution_trace: list[TraceStep] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    spatial_evidence: Optional[SpatialEvidence] = None
