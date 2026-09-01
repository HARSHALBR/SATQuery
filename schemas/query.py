"""Query-related schemas for SATQuery AI.

Defines the structured representations for user queries, image metadata,
observation inputs, and the parsed query that drives the planner.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Supported task types in the SATQuery system."""

    SINGLE_IMAGE_VQA = "single_image_vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    BI_TEMPORAL_CHANGE = "bi_temporal_change"
    VEGETATION_CHANGE = "vegetation_change"
    BUILT_UP_CHANGE = "built_up_change"
    OPTICAL_SAR_CROSS_CHECK = "optical_sar_cross_check"
    SPATIAL_MEASUREMENT = "spatial_measurement"
    INSUFFICIENT_CAPABILITY = "insufficient_capability"


class Modality(str, Enum):
    """Supported image modalities."""

    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"


class ObservationRole(str, Enum):
    """Role of an uploaded observation in the analysis."""

    T1 = "t1"
    T2 = "t2"
    SAR_T1 = "sar_t1"
    SAR_T2 = "sar_t2"
    SINGLE = "single"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ImageMetadata(BaseModel):
    """Metadata for a single satellite image."""

    sensor: Optional[str] = None
    modality: Modality
    bands: list[str] = Field(default_factory=list)
    acquisition_date: Optional[date] = None
    crs: Optional[str] = None
    resolution_m: Optional[float] = None
    bounds: Optional[dict] = None
    cloud_cover: Optional[float] = None
    nodata_value: Optional[float] = None
    # Phase 13C Provenance Fields
    product_id: Optional[str] = None
    processing_level: Optional[str] = None
    cloud_mask_band: Optional[str] = None
    stac_item_id: Optional[str] = None
    catalog_url: Optional[str] = None


class ObservationInput(BaseModel):
    """A single uploaded observation — image file reference plus metadata."""

    observation_id: str
    image_path: str
    role: ObservationRole
    metadata: ImageMetadata


class ParsedQuery(BaseModel):
    """Structured representation of a parsed natural-language query.

    Produced by the query-understanding layer and consumed by the planner.
    """

    raw_query: str
    task: TaskType
    intent: str
    claim: Optional[str] = None
    observations: int = 1
    temporal: bool = False
    spatial: bool = False
    quantification: bool = False
    modalities: list[Modality] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    candidate_tools: list[str] = Field(default_factory=list)


class QueryInput(BaseModel):
    """Top-level input combining the raw query string and uploaded observations."""

    query: str
    observations: list[ObservationInput] = Field(default_factory=list)
    metadata: Optional[dict] = None
