"""Evidence Comparator for GeoVision.

Determines whether collected evidence supports, contradicts, or is
insufficient for the claim extracted by the TaskClassifier.

The comparator uses transparent, deterministic rules — no LLM, no
majority voting, no arbitrary confidence fusion.

Responsibilities:
    EvidenceStore:    "What evidence exists?"
    EvidenceComparator: "Does the evidence support the claim?"

The comparator does NOT execute tools, call VLMs, or modify evidence.
"""

from __future__ import annotations

from typing import Optional

from schemas.evidence import EvidenceRecord
from schemas.response import ComparisonResult, EvidenceStatus


# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------

_MIN_VALID_PIXEL_FRACTION = 0.50
_MAX_CLOUD_COVER = 0.50


# ---------------------------------------------------------------------------
# Claim → compatible evidence type mapping
# ---------------------------------------------------------------------------

# Maps claim keywords to the evidence types that can directly address them.
# Evidence types not in this mapping for a given claim are treated as
# "not directly applicable" and cannot promote a result to SUPPORTED.

_CLAIM_EVIDENCE_MAP: dict[str, list[str]] = {
    # Vegetation claims require NDVI evidence + change quantification.
    "vegetation_decrease": ["vlm_interpretation", "vegetation_change", "change_quantification", "spatial_grounding"],
    "vegetation_increase": ["vlm_interpretation", "vegetation_change", "change_quantification", "spatial_grounding"],
    "vegetation_change":   ["vlm_interpretation", "vegetation_change", "change_quantification", "spatial_grounding"],
    # Built-up claims require NDBI evidence + change quantification.
    "built_up_increase":   ["vlm_interpretation", "built_up_change", "change_quantification", "spatial_grounding"],
    "built_up_decrease":   ["vlm_interpretation", "built_up_change", "change_quantification", "spatial_grounding"],
    "built_up_change":     ["vlm_interpretation", "built_up_change", "change_quantification", "spatial_grounding"],
    # SAR cross-check claims require SAR amplitude change evidence.
    "sar_cross_check":     ["vlm_interpretation", "sar_amplitude_change", "spatial_grounding"],
    # Generic change claims.
    "change_detected":     ["vlm_interpretation", "change_quantification", "spatial_grounding"],
    "general_change":      ["vlm_interpretation", "change_quantification", "spatial_grounding"],
}

_CLAIM_OPTIONAL_EVIDENCE_MAP: dict[str, list[str]] = {
    "vegetation_decrease": ["vlm_interpretation", "spatial_grounding", "spatial_measurement"],
    "vegetation_increase": ["vlm_interpretation", "spatial_grounding", "spatial_measurement"],
    "vegetation_change":   ["vlm_interpretation", "spatial_grounding", "spatial_measurement"],
    "built_up_increase":   ["vlm_interpretation", "spatial_grounding", "spatial_measurement"],
    "built_up_decrease":   ["vlm_interpretation", "spatial_grounding", "spatial_measurement"],
    "built_up_change":     ["vlm_interpretation", "spatial_grounding", "spatial_measurement"],
    "sar_cross_check":     ["spatial_grounding", "change_quantification"],
    "change_detected":     ["spatial_grounding", "spatial_measurement"],
    "general_change":      ["spatial_grounding", "spatial_measurement"],
}

# The minimum set of evidence types that must ALL be present (and pass
# quality) to reach SUPPORTED for a given claim family.
_REQUIRED_EVIDENCE_FOR_SUPPORTED: dict[str, list[str]] = {
    "vegetation_decrease": ["vegetation_change", "change_quantification"],
    "vegetation_increase": ["vegetation_change", "change_quantification"],
    "vegetation_change":   ["vegetation_change", "change_quantification"],
    "built_up_increase":   ["built_up_change", "change_quantification"],
    "built_up_decrease":   ["built_up_change", "change_quantification"],
    "built_up_change":     ["built_up_change", "change_quantification"],
    "sar_cross_check":     ["vlm_interpretation", "sar_amplitude_change"],
    "change_detected":     ["vlm_interpretation", "change_quantification"],
    "general_change":      ["vlm_interpretation", "change_quantification"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_claim(claim: Optional[str]) -> str:
    """Normalize claim string to a canonical lookup key."""
    if not claim:
        return ""
    return claim.strip().lower().replace(" ", "_").replace("-", "_")


def _passes_quality(record: EvidenceRecord) -> bool:
    """Check whether an evidence record meets minimum quality standards.

    Conservative: unknown quality fields are treated as failing.
    """
    q = record.quality

    # Valid pixel fraction check.
    if q.valid_pixel_fraction is not None:
        if q.valid_pixel_fraction < _MIN_VALID_PIXEL_FRACTION:
            return False

    # Registration check.
    if q.registration_ok is not None and not q.registration_ok:
        return False

    # Cloud cover check.
    if q.cloud_cover is not None:
        if q.cloud_cover > _MAX_CLOUD_COVER:
            return False

    return True


def _extract_direction(record: EvidenceRecord) -> Optional[str]:
    """Extract a directional indicator from the evidence value dict."""
    if not isinstance(record.value, dict):
        return None

    # Vegetation / NDBI direction field.
    direction = record.value.get("direction")
    if direction:
        return str(direction).lower()

    # VLM interpretation text.
    interp = record.value.get("interpretation")
    if interp:
        text = str(interp).lower()
        if "decrease" in text or "loss" in text or "reduced" in text:
            return "decrease"
        if "increase" in text or "growth" in text or "expanded" in text:
            return "increase"

    return None


def _claim_direction(claim_key: str) -> Optional[str]:
    """Extract the expected direction from a normalized claim key."""
    if "decrease" in claim_key or "loss" in claim_key:
        return "decrease"
    if "increase" in claim_key or "growth" in claim_key:
        return "increase"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class EvidenceComparator:
    """Compares collected evidence against a claim to produce a ComparisonResult.

    Rules (transparent, deterministic):

    1. If no evidence exists, result is INSUFFICIENT.
    2. If the claim cannot be mapped to required evidence types, result
       is INSUFFICIENT (no compatible evidence definition).
    3. If required evidence types are missing, result is INSUFFICIENT.
    4. If any required evidence fails quality checks, result is
       INSUFFICIENT.
    5. If directional evidence contradicts the claim direction, result
       is UNCERTAIN.
    6. If multiple directional evidence sources disagree with each
       other, result is UNCERTAIN.
    7. If all required evidence is present, passes quality, and is
       directionally consistent, result is SUPPORTED.

    Limitation: region/temporal consistency is checked only when
    EvidenceRecord.region contains an 'id' field. Advanced GIS geometry
    comparison is not implemented in this phase.
    """

    def compare(
        self,
        claim: Optional[str],
        evidence: list[EvidenceRecord],
    ) -> ComparisonResult:
        """Compare evidence against a claim.

        Args:
            claim:    The claim string from ParsedQuery.claim.
            evidence: List of EvidenceRecord objects to evaluate.

        Returns:
            A ComparisonResult with status, reason, and evidence IDs.
        """
        claim_key = _normalize_claim(claim)
        limitations: list[str] = []

        # -- Gate 1: no evidence at all -----------------------------------
        if not evidence:
            return ComparisonResult(
                status=EvidenceStatus.INSUFFICIENT,
                reason="No evidence available to evaluate the claim.",
                limitations=["No evidence records were provided."],
            )

        # -- Gate 2: unmapped claim ---------------------------------------
        compatible_types = _CLAIM_EVIDENCE_MAP.get(claim_key)
        required_types = _REQUIRED_EVIDENCE_FOR_SUPPORTED.get(claim_key)

        if compatible_types is None or required_types is None:
            return ComparisonResult(
                status=EvidenceStatus.INSUFFICIENT,
                reason=f"No evidence requirements defined for claim '{claim_key}'.",
                limitations=[
                    f"Claim '{claim_key}' does not map to known evidence types."
                ],
            )

        # -- Partition evidence by compatibility --------------------------
        compatible_evidence: list[EvidenceRecord] = []
        incompatible_evidence: list[EvidenceRecord] = []

        for ev in evidence:
            if ev.type in compatible_types:
                compatible_evidence.append(ev)
            else:
                incompatible_evidence.append(ev)

        if incompatible_evidence:
            limitations.append(
                f"{len(incompatible_evidence)} evidence record(s) are not "
                f"directly applicable to claim '{claim_key}'."
            )

        # -- Gate 3: missing required evidence types ----------------------
        present_types = {ev.type for ev in compatible_evidence}
        missing_types = [t for t in required_types if t not in present_types]

        if missing_types:
            return ComparisonResult(
                status=EvidenceStatus.INSUFFICIENT,
                reason=(
                    f"Required evidence type(s) missing: "
                    f"{', '.join(sorted(missing_types))}."
                ),
                supporting_evidence=[],
                conflicting_evidence=[],
                limitations=limitations + [
                    f"Missing: {', '.join(sorted(missing_types))}."
                ],
            )

        # -- Gate 4: quality check on compatible evidence -----------------
        quality_failed: list[EvidenceRecord] = []
        quality_passed: list[EvidenceRecord] = []

        for ev in compatible_evidence:
            if _passes_quality(ev):
                quality_passed.append(ev)
            else:
                quality_failed.append(ev)

        if quality_failed:
            # Check if any required type now has NO quality-passing records.
            passed_types = {ev.type for ev in quality_passed}
            quality_missing = [
                t for t in required_types if t not in passed_types
            ]
            if quality_missing:
                return ComparisonResult(
                    status=EvidenceStatus.INSUFFICIENT,
                    reason=(
                        f"Evidence quality is insufficient for type(s): "
                        f"{', '.join(sorted(quality_missing))}."
                    ),
                    supporting_evidence=[],
                    conflicting_evidence=[
                        ev.evidence_id for ev in quality_failed
                    ],
                    limitations=limitations + [
                        f"Quality failure in: "
                        f"{', '.join(ev.evidence_id for ev in quality_failed)}."
                    ],
                )

        # -- Gate 5: directional consistency ------------------------------
        expected_direction = _claim_direction(claim_key)
        supporting_ids: list[str] = []
        conflicting_ids: list[str] = []

        for ev in quality_passed:
            ev_direction = _extract_direction(ev)
            if ev_direction is None:
                # Non-directional evidence (e.g. change_quantification)
                # counts as supporting if it passed quality.
                supporting_ids.append(ev.evidence_id)
                continue

            if expected_direction is not None:
                if ev_direction == expected_direction:
                    supporting_ids.append(ev.evidence_id)
                else:
                    conflicting_ids.append(ev.evidence_id)
            else:
                # No expected direction (generic claim) — any directional
                # evidence is considered supporting.
                supporting_ids.append(ev.evidence_id)

        # -- Gate 6: cross-source directional disagreement ----------------
        # Even if the claim has no expected direction, check whether
        # directional evidence sources disagree with each other.
        directions_seen: set[str] = set()
        for ev in quality_passed:
            d = _extract_direction(ev)
            if d is not None:
                directions_seen.add(d)

        if len(directions_seen) > 1:
            return ComparisonResult(
                status=EvidenceStatus.UNCERTAIN,
                reason=(
                    "Evidence sources disagree on direction: "
                    f"{', '.join(sorted(directions_seen))}."
                ),
                supporting_evidence=sorted(supporting_ids),
                conflicting_evidence=sorted(conflicting_ids),
                limitations=limitations,
            )

        if conflicting_ids:
            return ComparisonResult(
                status=EvidenceStatus.UNCERTAIN,
                reason="Evidence contradicts the claimed direction.",
                supporting_evidence=sorted(supporting_ids),
                conflicting_evidence=sorted(conflicting_ids),
                limitations=limitations,
            )

        # -- Gate 7: region consistency (best-effort) ---------------------
        regions_seen: set[str] = set()
        for ev in quality_passed:
            if ev.region and isinstance(ev.region, dict):
                rid = ev.region.get("id")
                if rid:
                    regions_seen.add(str(rid))

        if len(regions_seen) > 1:
            limitations.append(
                f"Evidence spans multiple regions: "
                f"{', '.join(sorted(regions_seen))}. "
                f"Cross-region corroboration may not be reliable."
            )

        # -- All gates passed → SUPPORTED ---------------------------------
        return ComparisonResult(
            status=EvidenceStatus.SUPPORTED,
            reason="All required evidence is present, passes quality, and is consistent.",
            supporting_evidence=sorted(supporting_ids),
            conflicting_evidence=[],
            limitations=limitations,
        )
