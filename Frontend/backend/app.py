"""
SatQuery Backend API
Handles image uploads and provides change detection analysis.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import shutil
from pipeline_config import BEFORE_YEAR, AFTER_YEAR, DATA_DIR
from workspace import AnalysisWorkspace

# Pipeline Modules — same functions used by both real-upload and demo paths
from calculate_ndvi import calculate_ndvi_for_year
from calculate_ndbi import calculate_ndbi_for_year
from calculate_change import calculate_change
from calculate_sar_change import calculate_sar_change
from geospatial.alignment import create_analysis_grid, align_raster
from create_evidence_score import create_evidence_score
from cluster_evidence import cluster_evidence
from create_region_report import create_region_report
from convert_regions_to_geojson import convert_regions_to_geojson
from create_evidence_summary import create_evidence_summary
from create_evidence_record import create_evidence_record
from evidence_status import (
    SUPPORTED, UNCERTAIN, CONFLICTING, INSUFFICIENT,
    determine_overall_status, format_status_explanation
)

# ---------------------------------------------------------------------------
# Demo fixture paths — source change rasters only.
# These are used when demo_mode=True. We run the SAME pipeline functions;
# only the inputs differ (pre-computed change rasters vs. live uploads).
# We align them to a common grid before passing to evidence scoring.
# ---------------------------------------------------------------------------
SAMPLE_DIR = DATA_DIR
DEMO_NDVI_CHANGE = SAMPLE_DIR / "S2_NDVI_change_2017_2018.tif"
DEMO_NDBI_CHANGE = SAMPLE_DIR / "S2_NDBI_change_2017_2018.tif"
DEMO_VV_CHANGE   = SAMPLE_DIR / "S1_vv_change_2017_2018.tif"
DEMO_VH_CHANGE   = SAMPLE_DIR / "S1_vh_change_2017_2018.tif"

app = FastAPI(title="SatQuery API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: derive final human-readable answer from intent + status + evidence
# Never hardcoded — built from actual API response fields.
# ---------------------------------------------------------------------------
def build_final_answer(intent: str, status: str, evidence_record: dict | None) -> str:
    """
    Derive a natural-language answer from:
      - intent   (VEGETATION | URBAN | GENERAL_CHANGE)
      - status   (SUPPORTED | UNCERTAIN | CONFLICTING | INSUFFICIENT)
      - evidence_record (contains ndvi/score data for direction)
    """
    strong_evidence = evidence_record.get("evidence", {}).get("strong_evidence", {}) if evidence_record else {}
    strong_pixels = strong_evidence.get("pixels", 0)
    regions = evidence_record.get("regions", {}).get("strong_evidence_regions", []) if evidence_record else []
    region_count = len(regions)

    # Determine NDVI direction from evidence items
    ndvi_decreased = False
    ndbi_increased = False
    if evidence_record:
        for item in evidence_record.get("evidence_items", []):
            if item.get("type") == "ndvi_decrease" and item.get("supports_claim"):
                ndvi_decreased = True
            if item.get("type") == "ndbi_increase" and item.get("supports_claim"):
                ndbi_increased = True

    region_str = f"{region_count} region{'s' if region_count != 1 else ''}"

    if status == SUPPORTED:
        if intent == "VEGETATION":
            direction = "decreased" if ndvi_decreased else "changed"
            return (
                f"YES — Vegetation {direction} between T1 and T2. "
                f"Evidence is SUPPORTED by multi-sensor analysis across {region_str} "
                f"({strong_pixels:,} high-confidence pixels). "
                "Both optical (NDVI/NDBI) and SAR (VV/VH) sensors agree."
            )
        if intent == "URBAN":
            direction = "increased" if ndbi_increased else "changed"
            return (
                f"YES — Built-up area {direction} between T1 and T2. "
                f"Evidence is SUPPORTED across {region_str}."
            )
        return (
            f"YES — Change is detected and SUPPORTED by multi-sensor satellite evidence "
            f"across {region_str} ({strong_pixels:,} high-confidence pixels)."
        )

    if status == UNCERTAIN:
        return (
            "UNCERTAIN — Change patterns were detected, but the evidence is not strong enough "
            "to confirm with high confidence. Only one sensor type shows significant signal, "
            "or the signal strength is below the confirmation threshold."
        )

    if status == CONFLICTING:
        return (
            "CONFLICTING — Optical (Sentinel-2) and SAR (Sentinel-1) sensors disagree. "
            "The change detected by one sensor is not corroborated by the other. "
            "This could indicate a data artefact, cloud contamination, or a transient event."
        )

    if status == INSUFFICIENT:
        return (
            "INSUFFICIENT — The required satellite evidence could not be computed from "
            "the available data. This may be due to missing bands, invalid file format, "
            "or no detectable change above the evidence threshold."
        )

    return "Analysis status unknown. Please check the processing trace for details."


# ---------------------------------------------------------------------------
# Shared pipeline executor: runs evidence scoring → clustering → regions →
# GeoJSON → summary → record from aligned change rasters.
# Used by BOTH real-upload and demo paths — no duplicate logic.
# ---------------------------------------------------------------------------
def run_evidence_pipeline(
    ndvi_change: Path,
    ndbi_change: Path,
    vv_change: Path,
    vh_change: Path,
    workspace: AnalysisWorkspace,
    before_year: int,
    after_year: int,
    trace: list,
    create_insufficient_response,
):
    """
    Given 4 aligned change rasters, run the evidence pipeline and return
    (geojson_data, record_data, overall_status, regions) or an error response dict.
    Returns None for error_response on success.
    """
    d_dir = workspace.get_dir("derived")

    def add_trace(step, status, reason=None):
        entry = {"step": step, "status": status}
        if reason:
            entry["reason"] = reason
        trace.append(entry)

    # ------------------------------------------------------------------
    # STEP 1: Evidence Score
    # ------------------------------------------------------------------
    add_trace("evidence_scoring", "in_progress")
    try:
        evidence_score_file = d_dir / "evidence_score.tif"
        classification_file = d_dir / "classification.tif"
        create_evidence_score(
            ndvi_change, ndbi_change, vv_change, vh_change,
            evidence_score_file, classification_file
        )
        add_trace("evidence_scoring", "completed")
    except Exception as e:
        add_trace("evidence_scoring", "failed", str(e))
        return None, None, None, None, create_insufficient_response(
            f"Evidence scoring failed: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 2: Clustering
    # ------------------------------------------------------------------
    add_trace("region_clustering", "in_progress")
    try:
        clusters_file = d_dir / "clusters.tif"
        cluster_evidence(evidence_score_file, clusters_file)
        add_trace("region_clustering", "completed")
    except Exception as e:
        add_trace("region_clustering", "failed", str(e))
        return None, None, None, None, create_insufficient_response(
            f"Region clustering failed: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 3: Region Reporting
    # ------------------------------------------------------------------
    add_trace("region_analysis", "in_progress")
    try:
        report_file = workspace.get_dir("regions") / "regions.json"
        create_region_report(
            clusters_file, ndvi_change, ndbi_change, vv_change, vh_change,
            report_file
        )
        add_trace("region_analysis", "completed")
    except Exception as e:
        add_trace("region_analysis", "failed", str(e))
        return None, None, None, None, create_insufficient_response(
            f"Region analysis failed: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 4: GeoJSON Generation
    # ------------------------------------------------------------------
    add_trace("geojson_generation", "in_progress")
    try:
        geojson_file = workspace.get_dir("regions") / "regions.geojson"
        geojson_data = convert_regions_to_geojson(report_file, clusters_file, geojson_file)
        add_trace("geojson_generation", "completed")
    except Exception as e:
        add_trace("geojson_generation", "failed", str(e))
        return None, None, None, None, create_insufficient_response(
            f"GeoJSON generation failed: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 5: Evidence Summary
    # ------------------------------------------------------------------
    add_trace("evidence_summary", "in_progress")
    try:
        summary_file = workspace.get_dir("evidence") / "summary.json"
        create_evidence_summary(
            evidence_score_file, ndvi_change, ndbi_change, vv_change, vh_change,
            summary_file
        )
        add_trace("evidence_summary", "completed")
    except Exception as e:
        add_trace("evidence_summary", "failed", str(e))
        return None, None, None, None, create_insufficient_response(
            f"Evidence summary failed: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 6: Evidence Record
    # ------------------------------------------------------------------
    add_trace("evidence_record", "in_progress")
    try:
        record_file = workspace.get_dir("evidence") / "record.json"
        record_data = create_evidence_record(
            summary_file, report_file, record_file, before_year, after_year
        )
        add_trace("evidence_record", "completed")
    except Exception as e:
        add_trace("evidence_record", "failed", str(e))
        return None, None, None, None, create_insufficient_response(
            f"Evidence record creation failed: {str(e)}"
        )

    # ------------------------------------------------------------------
    # STEP 7: Status Determination
    # ------------------------------------------------------------------
    try:
        regions = record_data.get("regions", {}).get("strong_evidence_regions", [])
        overall_status = determine_overall_status(regions) if regions else INSUFFICIENT
        add_trace("status_determination", "completed", f"Status: {overall_status}")
        if overall_status == INSUFFICIENT:
            return None, None, None, None, create_insufficient_response(
                "No regions with sufficient evidence were detected"
            )
    except Exception as e:
        add_trace("status_determination", "failed", str(e))
        return None, None, None, None, create_insufficient_response(
            f"Status determination failed: {str(e)}"
        )

    return geojson_data, record_data, overall_status, regions, None


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------
async def validate_uploaded_image(file: UploadFile, label: str, workspace: AnalysisWorkspace) -> dict:
    """Validate uploaded image format, save it, and extract metadata."""
    if not file:
        raise HTTPException(status_code=400, detail=f"Missing required file: {label}")

    temp_dir = workspace.get_dir("metadata") / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        entry = workspace.add_input_file(label, temp_path)
    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise HTTPException(
            status_code=400,
            detail=f"{label} is invalid: {str(e)}. (Note: True analysis requires scientific GeoTIFFs, not RGB PNGs)"
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

    return entry


# ===========================================================================
# HEALTH CHECK
# ===========================================================================
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "SatQuery Backend"}


# ===========================================================================
# GET /evidence — Serve the pre-computed demo evidence record
# ===========================================================================
@app.get("/evidence")
async def get_evidence():
    """
    Return the pre-computed evidence record for the default demo dataset (2017→2018).
    This endpoint powers the dashboard's initial state without requiring an upload.
    The GeoJSON is also returned via /data served by Vite; this endpoint provides
    the evidence statistics and region metadata.
    """
    record_file = SAMPLE_DIR / "evidence_record.json"
    summary_file = SAMPLE_DIR / "evidence_summary.json"

    if not record_file.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Demo evidence record not found. "
                "Run restore_demo_data.py to regenerate it."
            )
        )

    with open(record_file) as f:
        record = json.load(f)

    # Also merge summary classification counts if available
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        record["classification"] = summary.get("classification", record.get("classification", {}))

    return record


# ===========================================================================
# POST /analyze — Main analysis endpoint
# ===========================================================================
@app.post("/analyze")
async def analyze_change(
    question: str = Form(...),
    demo_mode: bool = Form(False),
    # All 10 band uploads are optional — required only when demo_mode=False
    before_b04: Optional[UploadFile] = File(None),
    before_b08: Optional[UploadFile] = File(None),
    before_b11: Optional[UploadFile] = File(None),
    before_vv:  Optional[UploadFile] = File(None),
    before_vh:  Optional[UploadFile] = File(None),
    after_b04:  Optional[UploadFile] = File(None),
    after_b08:  Optional[UploadFile] = File(None),
    after_b11:  Optional[UploadFile] = File(None),
    after_vv:   Optional[UploadFile] = File(None),
    after_vh:   Optional[UploadFile] = File(None),
    before_year: int = Form(BEFORE_YEAR),
    after_year:  int = Form(AFTER_YEAR),
):
    """
    Analyze satellite imagery for change detection.

    demo_mode=True: Uses pre-computed change rasters from data/sample/ as inputs
    and runs the SAME evidence pipeline (no separate implementation).

    demo_mode=False: Requires 10 GeoTIFF uploads (B04, B08, B11, VV, VH × 2 epochs).

    Returns a structured response with:
    - status: SUPPORTED | UNCERTAIN | CONFLICTING | INSUFFICIENT
    - final_answer: dynamic natural-language answer (never hardcoded)
    - geojson: detected change regions
    - evidence_record: full evidence data
    - trace: processing steps executed
    """
    workspace = AnalysisWorkspace()
    trace = []

    def add_trace(step: str, status: str, reason: str = None):
        entry = {"step": step, "status": status}
        if reason:
            entry["reason"] = reason
        trace.append(entry)

    def create_insufficient_response(reason: str):
        return {
            "question": question,
            "intent": "INSUFFICIENT_DATA",
            "analysis_id": workspace.analysis_id,
            "status": INSUFFICIENT,
            "final_answer": (
                f"INSUFFICIENT — {reason}"
            ),
            "reason": reason,
            "regions": [],
            "geojson": {"type": "FeatureCollection", "features": []},
            "evidence_record": None,
            "trace": trace,
            "provenance": {},
            "limitations": [reason],
            "demo_mode": demo_mode,
        }

    try:
        # ------------------------------------------------------------------
        # INTENT DETECTION (used for final answer)
        # ------------------------------------------------------------------
        intent = "GENERAL_CHANGE"
        q_lower = question.lower()
        if any(w in q_lower for w in ["vegetation", "forest", "green", "tree", "ndvi", "crop"]):
            intent = "VEGETATION"
        elif any(w in q_lower for w in ["built-up", "construction", "urban", "building", "city"]):
            intent = "URBAN"

        # ------------------------------------------------------------------
        # SELECT CHANGE RASTERS: demo fixtures or from uploaded bands
        # ------------------------------------------------------------------
        if demo_mode:
            # ---- DEMO MODE: use pre-computed change rasters as pipeline inputs ----
            add_trace("input_validation", "in_progress")
            for label, path in [
                ("NDVI change", DEMO_NDVI_CHANGE),
                ("NDBI change", DEMO_NDBI_CHANGE),
                ("VV change",   DEMO_VV_CHANGE),
                ("VH change",   DEMO_VH_CHANGE),
            ]:
                if not path.exists():
                    add_trace("input_validation", "failed", f"Missing demo fixture: {path.name}")
                    return create_insufficient_response(
                        f"Demo fixture not found: {path.name}. "
                        "Run restore_demo_data.py to regenerate fixtures."
                    )
            add_trace("input_validation", "completed",
                      "Using pre-computed 2017→2018 change rasters as pipeline inputs")

            ndvi_change = DEMO_NDVI_CHANGE
            ndbi_change = DEMO_NDBI_CHANGE
            vv_change   = DEMO_VV_CHANGE
            vh_change   = DEMO_VH_CHANGE

            # Align demo rasters to a common grid (VV patch as reference)
            # The optical and SAR rasters have different dimensions
            add_trace("spatial_alignment", "in_progress")
            try:
                a_dir = workspace.get_dir("aligned")
                ndvi_aligned = a_dir / "ndvi_change_aligned.tif"
                ndbi_aligned = a_dir / "ndbi_change_aligned.tif"
                vv_aligned   = a_dir / "vv_change_aligned.tif"
                vh_aligned   = a_dir / "vh_change_aligned.tif"
                ref_grid = create_analysis_grid(vv_change)
                align_raster(ndvi_change, ref_grid, ndvi_aligned)
                align_raster(ndbi_change, ref_grid, ndbi_aligned)
                align_raster(vv_change,   ref_grid, vv_aligned)
                align_raster(vh_change,   ref_grid, vh_aligned)
                ndvi_change, ndbi_change, vv_change, vh_change = (
                    ndvi_aligned, ndbi_aligned, vv_aligned, vh_aligned
                )
                add_trace("spatial_alignment", "completed")
            except Exception as e:
                add_trace("spatial_alignment", "failed", str(e))
                return create_insufficient_response(
                    f"Demo raster alignment failed: {str(e)}"
                )

        else:
            # ---- REAL UPLOAD MODE: process 10 GeoTIFFs through full pipeline ----
            add_trace("input_validation", "in_progress")
            required = {
                "before_b04": before_b04, "before_b08": before_b08, "before_b11": before_b11,
                "before_vv": before_vv,   "before_vh": before_vh,
                "after_b04":  after_b04,  "after_b08":  after_b08,  "after_b11":  after_b11,
                "after_vv":  after_vv,    "after_vh":  after_vh,
            }
            for name, f in required.items():
                if f is None:
                    add_trace("input_validation", "failed", f"Missing required band: {name}")
                    return create_insufficient_response(
                        f"Missing required upload: {name}. "
                        "Upload all 10 bands or use demo_mode=true."
                    )
            try:
                b_b04 = await validate_uploaded_image(before_b04, "before_b04", workspace)
                b_b08 = await validate_uploaded_image(before_b08, "before_b08", workspace)
                b_b11 = await validate_uploaded_image(before_b11, "before_b11", workspace)
                b_vv  = await validate_uploaded_image(before_vv,  "before_vv",  workspace)
                b_vh  = await validate_uploaded_image(before_vh,  "before_vh",  workspace)
                a_b04 = await validate_uploaded_image(after_b04,  "after_b04",  workspace)
                a_b08 = await validate_uploaded_image(after_b08,  "after_b08",  workspace)
                a_b11 = await validate_uploaded_image(after_b11,  "after_b11",  workspace)
                a_vv  = await validate_uploaded_image(after_vv,   "after_vv",   workspace)
                a_vh  = await validate_uploaded_image(after_vh,   "after_vh",   workspace)
                add_trace("input_validation", "completed")
            except HTTPException as e:
                add_trace("input_validation", "failed", str(e.detail))
                return create_insufficient_response(f"Input validation failed: {e.detail}")
            except Exception as e:
                add_trace("input_validation", "failed", str(e))
                return create_insufficient_response(f"Input validation failed: {str(e)}")

            d_dir = workspace.get_dir("derived")
            a_dir = workspace.get_dir("aligned")

            # Calculate NDVI
            add_trace("ndvi_calculation", "in_progress")
            try:
                ndvi_before_f = d_dir / "ndvi_before.tif"
                calculate_ndvi_for_year(before_year, red_input=b_b04["path"],
                                        nir_input=b_b08["path"], output_name=ndvi_before_f,
                                        data_dir=None, is_absolute=True)
                ndvi_after_f = d_dir / "ndvi_after.tif"
                calculate_ndvi_for_year(after_year, red_input=a_b04["path"],
                                        nir_input=a_b08["path"], output_name=ndvi_after_f,
                                        data_dir=None, is_absolute=True)
                add_trace("ndvi_calculation", "completed")
            except Exception as e:
                add_trace("ndvi_calculation", "failed", str(e))
                return create_insufficient_response(f"NDVI calculation failed: {str(e)}")

            # Calculate NDBI
            add_trace("ndbi_calculation", "in_progress")
            try:
                ndbi_before_f = d_dir / "ndbi_before.tif"
                calculate_ndbi_for_year(before_year, nir_input=b_b08["path"],
                                        swir_input=b_b11["path"], output_name=ndbi_before_f,
                                        data_dir=None, is_absolute=True)
                ndbi_after_f = d_dir / "ndbi_after.tif"
                calculate_ndbi_for_year(after_year, nir_input=a_b08["path"],
                                        swir_input=a_b11["path"], output_name=ndbi_after_f,
                                        data_dir=None, is_absolute=True)
                add_trace("ndbi_calculation", "completed")
            except Exception as e:
                add_trace("ndbi_calculation", "failed", str(e))
                return create_insufficient_response(f"NDBI calculation failed: {str(e)}")

            # Calculate change
            add_trace("change_calculation", "in_progress")
            try:
                ndvi_change_raw = d_dir / "ndvi_change_raw.tif"
                ndbi_change_raw = d_dir / "ndbi_change_raw.tif"
                calculate_change(ndvi_before_f, ndvi_after_f, ndvi_change_raw)
                calculate_change(ndbi_before_f, ndbi_after_f, ndbi_change_raw)
                add_trace("change_calculation", "completed")
            except Exception as e:
                add_trace("change_calculation", "failed", str(e))
                return create_insufficient_response(f"Change calculation failed: {str(e)}")

            # Calculate SAR change
            add_trace("sar_calculation", "in_progress")
            try:
                vv_change_raw = d_dir / "vv_change_raw.tif"
                vh_change_raw = d_dir / "vh_change_raw.tif"
                calculate_sar_change(b_vv["path"], a_vv["path"],
                                     b_vh["path"], a_vh["path"],
                                     vv_change_raw, vh_change_raw)
                add_trace("sar_calculation", "completed")
            except Exception as e:
                add_trace("sar_calculation", "failed", str(e))
                return create_insufficient_response(f"SAR calculation failed: {str(e)}")

            # Spatial alignment
            add_trace("spatial_alignment", "in_progress")
            try:
                ndvi_change = a_dir / "ndvi_change_aligned.tif"
                ndbi_change = a_dir / "ndbi_change_aligned.tif"
                vv_change   = a_dir / "vv_change_aligned.tif"
                vh_change   = a_dir / "vh_change_aligned.tif"
                ref_grid = create_analysis_grid(ndvi_change_raw)
                align_raster(ndvi_change_raw, ref_grid, ndvi_change)
                align_raster(ndbi_change_raw, ref_grid, ndbi_change)
                align_raster(vv_change_raw,   ref_grid, vv_change)
                align_raster(vh_change_raw,   ref_grid, vh_change)
                add_trace("spatial_alignment", "completed")
            except Exception as e:
                add_trace("spatial_alignment", "failed", str(e))
                return create_insufficient_response(
                    f"Spatial alignment failed (CRS/bounds mismatch): {str(e)}"
                )

        # ------------------------------------------------------------------
        # EVIDENCE PIPELINE — same for both demo and real-upload paths
        # ------------------------------------------------------------------
        geojson_data, record_data, overall_status, regions, error_response = (
            run_evidence_pipeline(
                ndvi_change, ndbi_change, vv_change, vh_change,
                workspace, before_year, after_year, trace,
                create_insufficient_response,
            )
        )

        if error_response is not None:
            return error_response

        # ------------------------------------------------------------------
        # BUILD FINAL ANSWER — dynamically from intent + status + evidence
        # ------------------------------------------------------------------
        final_answer = build_final_answer(intent, overall_status, record_data)

        # ------------------------------------------------------------------
        # STATUS EXPLANATION
        # ------------------------------------------------------------------
        explanation = format_status_explanation(
            overall_status,
            regions_count=len(regions),
            supported_count=sum(1 for r in regions if r.get("status") == SUPPORTED),
            uncertain_count=sum(1 for r in regions if r.get("status") == UNCERTAIN),
            conflicting_count=sum(1 for r in regions if r.get("status") == CONFLICTING),
        )

        response_payload = {
            "question": question,
            "intent": intent,
            "analysis_id": workspace.analysis_id,
            "status": overall_status,
            "final_answer": final_answer,
            "status_explanation": explanation,
            "detected_regions": len(geojson_data.get("features", [])),
            "geojson": geojson_data,
            "evidence_record": record_data,
            "trace": trace,
            "demo_mode": demo_mode,
            "provenance": {
                "pipeline_version": "1.0",
                "demo_fixture_backed": demo_mode,
                "temporal_range": f"{before_year} to {after_year}",
                "timestamp": datetime.utcnow().isoformat(),
            },
            "limitations": [],
        }

        with open(workspace.get_dir("metadata") / "api_response.json", "w") as f:
            json.dump(response_payload, f, indent=2)

        return response_payload

    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "question": question,
            "intent": "ERROR",
            "analysis_id": workspace.analysis_id if workspace else "unknown",
            "status": INSUFFICIENT,
            "final_answer": f"INSUFFICIENT — Unexpected error during analysis: {str(e)}",
            "reason": f"Unexpected error: {str(e)}",
            "regions": [],
            "geojson": {"type": "FeatureCollection", "features": []},
            "evidence_record": None,
            "trace": trace,
            "demo_mode": demo_mode,
            "provenance": {},
            "limitations": ["Unexpected error during processing"],
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
