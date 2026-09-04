"""
evidence_status.py — Status constants and helper functions for SatQuery evidence pipeline.
"""

# Status constants
SUPPORTED = "SUPPORTED"
UNCERTAIN = "UNCERTAIN"
CONFLICTING = "CONFLICTING"
INSUFFICIENT = "INSUFFICIENT"


def determine_region_status(region: dict) -> str:
    """
    Determine the evidence status for a single region based on
    optical (ndvi/ndbi) and SAR (vv/vh) evidence counts.
    """
    score = region.get("evidence_score", 0)
    changes = region.get("changes", {})

    optical_signals = 0
    sar_signals = 0

    ndvi = changes.get("ndvi")
    ndbi = changes.get("ndbi")
    vv = changes.get("vv")
    vh = changes.get("vh")

    if ndvi is not None and ndvi < -0.20:
        optical_signals += 1
    if ndbi is not None and ndbi > 0.20:
        optical_signals += 1
    if vv is not None and vv < -0.07:
        sar_signals += 1
    if vh is not None and vh < -0.02:
        sar_signals += 1

    if score >= 3 and optical_signals >= 1 and sar_signals >= 1:
        return SUPPORTED
    if score >= 2 and (optical_signals >= 2 or sar_signals >= 2):
        return UNCERTAIN
    if optical_signals >= 1 and sar_signals >= 1 and score < 3:
        return CONFLICTING
    if score == 0 or (optical_signals == 0 and sar_signals == 0):
        return INSUFFICIENT
    return UNCERTAIN


def determine_overall_status(regions: list) -> str:
    """
    Aggregate region-level statuses into a single overall status.
    Priority: SUPPORTED > CONFLICTING > UNCERTAIN > INSUFFICIENT
    """
    if not regions:
        return INSUFFICIENT

    statuses = [r.get("status", INSUFFICIENT) for r in regions]
    status_set = set(statuses)

    if SUPPORTED in status_set:
        return SUPPORTED
    if CONFLICTING in status_set:
        return CONFLICTING
    if UNCERTAIN in status_set:
        return UNCERTAIN
    return INSUFFICIENT


def format_status_explanation(
    status: str,
    regions_count: int = 0,
    supported_count: int = 0,
    uncertain_count: int = 0,
    conflicting_count: int = 0,
) -> str:
    """Return a human-readable explanation string for the given status."""
    region_word = "region" if regions_count == 1 else "regions"

    if status == SUPPORTED:
        return (
            f"Multi-sensor analysis confirmed change across {regions_count} {region_word}. "
            f"Both optical (Sentinel-2) and SAR (Sentinel-1) evidence agree "
            f"({supported_count} supported, {uncertain_count} uncertain)."
        )
    if status == UNCERTAIN:
        return (
            f"Change patterns were detected across {regions_count} {region_word}, "
            "but the evidence does not meet the multi-sensor confirmation threshold. "
            "Only one sensor type shows a significant signal."
        )
    if status == CONFLICTING:
        return (
            f"Optical and SAR sensors disagree in {conflicting_count} {region_word}. "
            "This may indicate cloud contamination, a transient event, or a data artefact."
        )
    return (
        "Insufficient evidence to determine change status. "
        "No regions exceeded the minimum evidence threshold. "
        "Check that valid GeoTIFF inputs were provided."
    )
