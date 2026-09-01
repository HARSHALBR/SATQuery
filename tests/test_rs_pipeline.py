import pytest
import numpy as np
import datetime
import rasterio
from affine import Affine
from tools.rs import (
    validate_observations, RSValidationError, align_rasters,
    create_valid_mask, combined_valid_mask, compute_ndvi,
    compute_ndvi_delta, compute_change_statistics
)
from schemas.evidence import EvidenceRecord, QualityReport, Provenance

# ---------------------------------------------------------
# Test 1-3: NDVI math, Zero denom, NaN/NoData
# ---------------------------------------------------------
def test_basic_ndvi_mathematics():
    # NDVI = (NIR - RED) / (NIR + RED)
    red = np.array([1000, 2000, 0])
    nir = np.array([3000, 2000, 5000])
    
    ndvi = compute_ndvi(red, nir)
    
    assert np.isclose(ndvi[0], (3000-1000)/(3000+1000)) # 0.5
    assert np.isclose(ndvi[1], 0.0) # (2000-2000)/(2000+2000)
    assert np.isclose(ndvi[2], 1.0) # (5000-0)/5000

def test_zero_denominator():
    red = np.array([0, 0])
    nir = np.array([0, 0])
    ndvi = compute_ndvi(red, nir)
    # Should be NaN
    assert np.isnan(ndvi).all()

def test_nan_nodata_handling():
    red = np.array([1000, 1000])
    nir = np.array([3000, 3000])
    mask = np.array([True, False]) # Pixel 2 is invalid
    
    ndvi = compute_ndvi(red, nir, valid_mask=mask)
    assert np.isclose(ndvi[0], 0.5)
    assert np.isnan(ndvi[1])

# ---------------------------------------------------------
# Test 4-5: Cloud masking, Both-date validity
# ---------------------------------------------------------
def test_cloud_masking():
    scl = np.array([4, 9, 3, 5]) # Veg, Cloud High, Cloud Shadow, Not Veg
    # Valid should be 4, 5
    mask = create_valid_mask(scl)
    assert mask[0] == True
    assert mask[1] == False
    assert mask[2] == False
    assert mask[3] == True

def test_both_date_validity():
    # T1 valid: P1, P2. T2 valid: P1, P3. Both valid: P1 only.
    t1_scl = np.array([4, 4, 9])
    t2_scl = np.array([4, 9, 4])
    mask = combined_valid_mask(t1_scl, t2_scl)
    assert mask[0] == True
    assert mask[1] == False
    assert mask[2] == False

# ---------------------------------------------------------
# Test 6-10: Validation (Temporal, missing band, CRS, overlap)
# ---------------------------------------------------------
def test_temporal_ordering():
    t1_date = datetime.datetime(2021, 10, 1)
    t2_date = datetime.datetime(2021, 9, 1)
    with pytest.raises(RSValidationError, match="Invalid temporal ordering"):
        validate_observations({}, {}, t1_date, t2_date)

def test_missing_band_failure():
    t1_date = datetime.datetime(2021, 8, 1)
    t2_date = datetime.datetime(2021, 9, 1)
    t1_paths = {"red": "r.tif"} # missing nir, scl
    t2_paths = {"red": "r2.tif"}
    with pytest.raises(RSValidationError, match="T1 is missing required band"):
        validate_observations(t1_paths, t2_paths, t1_date, t2_date)

# Synthetic tiff creator for CRS/Grid tests
import tempfile
import os

def from_origin(west, north, xsize, ysize):
    return Affine.translation(west, north) * Affine.scale(xsize, -ysize)

def create_synthetic_tiff(path, crs_str="EPSG:4326", transform=None, width=10, height=10):
    if transform is None:
        transform = from_origin(0, 10, 1, 1)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype=str(rasterio.uint16), crs=crs_str, transform=transform
    ) as dst:
        dst.write(np.ones((height, width), dtype=rasterio.uint16), 1)

def test_validation_and_crs_overlap(tmp_path):
    # Test zero spatial overlap
    p1 = str(tmp_path / "t1_red.tif")
    p2 = str(tmp_path / "t2_red.tif")
    # T1 at x=0..10
    create_synthetic_tiff(p1, transform=from_origin(0, 10, 1, 1))
    # T2 at x=20..30 -> no overlap
    create_synthetic_tiff(p2, transform=from_origin(20, 10, 1, 1))
    
    t1_paths = {"red": p1, "nir": p1, "scl": p1}
    t2_paths = {"red": p2, "nir": p2, "scl": p2}
    t1_date = datetime.datetime(2021, 8, 1)
    t2_date = datetime.datetime(2021, 9, 1)
    
    with pytest.raises(RSValidationError, match="Zero spatial overlap"):
        validate_observations(t1_paths, t2_paths, t1_date, t2_date)

def test_grid_mismatch_resampling(tmp_path):
    # T1 standard 10x10
    p1 = str(tmp_path / "t1.tif")
    create_synthetic_tiff(p1, transform=from_origin(0, 10, 1, 1), width=10, height=10)
    
    # T2 offset and different resolution
    p2 = str(tmp_path / "t2.tif")
    create_synthetic_tiff(p2, transform=from_origin(0, 10, 2, 2), width=5, height=5)
    
    arr, resampled = align_rasters(p2, p1)
    assert resampled is True
    assert arr.shape == (10, 10)

# ---------------------------------------------------------
# Test 11-15: Statistics (Stable, Decrease, Increase, Thresholds)
# ---------------------------------------------------------
def test_stable_pixels():
    # NDVI t1=0.6, t2=0.6 -> delta=0.0
    delta = np.array([0.0, 0.0, 0.0])
    mask = np.array([True, True, True])
    stats = compute_change_statistics(delta, mask, threshold=-0.15, change_type="decrease")
    assert stats["decrease_pixel_fraction"] == 0.0
    assert stats["increase_pixel_fraction"] == 0.0
    assert stats["mean_delta"] == 0.0

def test_vegetation_decrease():
    # NDVI t1=0.8, t2=0.3 -> delta = -0.5
    delta = np.array([-0.5, -0.2, 0.0])
    mask = np.array([True, True, True])
    stats = compute_change_statistics(delta, mask, threshold=-0.15, change_type="decrease")
    # <= -0.15 matches the first two
    assert np.isclose(stats["decrease_pixel_fraction"], 2/3)

def test_vegetation_increase():
    # NDVI t1=0.2, t2=0.8 -> delta = +0.6
    delta = np.array([0.6, 0.2, 0.0])
    mask = np.array([True, True, True])
    stats = compute_change_statistics(delta, mask, threshold=0.15, change_type="increase")
    # >= 0.15 matches the first two
    assert np.isclose(stats["increase_pixel_fraction"], 2/3)

def test_threshold_boundary():
    delta = np.array([-0.15])
    mask = np.array([True])
    stats = compute_change_statistics(delta, mask, threshold=-0.15, change_type="decrease")
    # If delta <= threshold is used, it should be matched.
    assert stats["decrease_pixel_fraction"] == 1.0

# ---------------------------------------------------------
# Test 16-18: Evidence Generation, Provenance, Quality
# ---------------------------------------------------------
def test_evidence_generation_provenance():
    # Verify EvidenceRecord can properly store the RS payload
    prov = Provenance(
        tool="change_statistics", 
        tool_version="1.0", 
        input_ids=["S2A_10TFK_20210708_0_L2A", "S2B_10TFK_20211001_0_L2A"]
    )
    q_rep = QualityReport(cloud_cover=0.0, valid_pixel_fraction=1.0)
    
    val = {
        "total_valid_pixels": 100,
        "decrease_pixel_fraction": 0.5,
        "increase_pixel_fraction": 0.0,
        "mean_delta": -0.2,
        "threshold_used": -0.15
    }
    
    er = EvidenceRecord(
        evidence_id="ev_123",
        type="change_quantification",
        tool_version="rs-1.0",
        value=val,
        quality=q_rep,
        provenance=prov
    )
    
    assert er.value["mean_delta"] == -0.2
    assert er.quality.cloud_cover == 0.0
    assert er.provenance.tool == "change_statistics"
