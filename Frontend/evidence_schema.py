"""
evidence_schema.py — Pydantic/dataclass schemas for SatQuery evidence items.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvidenceItem:
    """A single piece of satellite evidence supporting or refuting a claim."""
    type: str                          # e.g. 'ndvi_decrease', 'ndbi_increase', 'vv_decrease'
    source: str                        # Sensor name, e.g. 'Sentinel-2', 'Sentinel-1'
    region_id: Optional[int] = None
    value: Optional[float] = None      # Measured change value
    threshold: Optional[float] = None  # Threshold used for classification
    supports_claim: bool = False       # Does this item support the hypothesis?
    description: str = ""
    confidence: float = 0.0            # 0.0 – 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "source": self.source,
            "region_id": self.region_id,
            "value": self.value,
            "threshold": self.threshold,
            "supports_claim": self.supports_claim,
            "description": self.description,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
