"""Evidence-related schemas for GeoVision.

Defines the evidence record, quality report, and provenance models
used throughout the evidence pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class QualityReport(BaseModel):
    """Quality assessment of the data underlying an evidence record."""

    valid_pixel_fraction: Optional[float] = None
    registration_ok: Optional[bool] = None
    cloud_cover: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    """Provenance chain for a single evidence record."""

    input_ids: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_utcnow)
    tool: str
    tool_version: str


class EvidenceRecord(BaseModel):
    """A single piece of evidence produced by a tool.

    Evidence records are the primary currency of the comparator.
    Each record is independently inspectable.
    """

    evidence_id: str
    type: str  # e.g. "ndvi_delta", "change_statistics", "sar_change"
    source: Optional[str] = None
    tool_version: str
    value: Optional[Any] = None
    region: Optional[dict] = None
    quality: QualityReport = Field(default_factory=QualityReport)
    provenance: Provenance
