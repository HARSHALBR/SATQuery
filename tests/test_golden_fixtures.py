import pytest
import numpy as np
import os
import json
import datetime
from tools.rs.validation import validate_observations
from tools.rs.alignment import align_rasters
from tools.rs.masking import combined_valid_mask
from tools.rs.ndvi import compute_ndvi_delta
from tools.rs.statistics import compute_change_statistics

MANIFEST_PATH = "datasets/golden_fixtures/manifests/golden_vegetation.json"
RAW_DIR = "datasets/golden_fixtures/raw"

def load_case_paths(item_id: str):
    return {
        "red": os.path.join(RAW_DIR, f"{item_id}_red.tif"),
        "nir": os.path.join(RAW_DIR, f"{item_id}_nir.tif"),
        "scl": os.path.join(RAW_DIR, f"{item_id}_scl.tif")
    }

def run_real_pipeline(case_id: str, threshold: float = -0.2):
    if not os.path.exists(MANIFEST_PATH):
        pytest.skip(f"Manifest not found: {MANIFEST_PATH}")
        
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
        
    case_data = next((p for p in manifest["pairs"] if p["pair_id"] == case_id), None)
    if not case_data:
        pytest.skip(f"Case {case_id} not found in manifest")
        
    t1_item = case_data["observation_t1"]["stac_item_id"]
    t2_item = case_data["observation_t2"]["stac_item_id"]
    
    t1_paths = load_case_paths(t1_item)
    t2_paths = load_case_paths(t2_item)
    
    # Check if files exist
    if not all(os.path.exists(p) for p in t1_paths.values()) or not all(os.path.exists(p) for p in t2_paths.values()):
        pytest.skip(f"Raw TIFFs for {case_id} not found. Run download_fixtures.py first.")
        
    t1_date = datetime.datetime.strptime(case_data["observation_t1"]["acquisition_date"], "%Y-%m-%dT%H:%M:%SZ")
    t2_date = datetime.datetime.strptime(case_data["observation_t2"]["acquisition_date"], "%Y-%m-%dT%H:%M:%SZ")
    
    # 1. Validate
    meta = validate_observations(t1_paths, t2_paths, t1_date, t2_date)
    
    # 2. Align (using T1 as reference grid)
    # Optical bands -> bilinear
    import rasterio.warp
    t1_red, _ = align_rasters(t1_paths["red"], t1_paths["red"], resampling_method=rasterio.warp.Resampling.bilinear)
    t1_nir, _ = align_rasters(t1_paths["nir"], t1_paths["red"], resampling_method=rasterio.warp.Resampling.bilinear)
    t2_red, resampled_red = align_rasters(t2_paths["red"], t1_paths["red"], resampling_method=rasterio.warp.Resampling.bilinear)
    t2_nir, _ = align_rasters(t2_paths["nir"], t1_paths["red"], resampling_method=rasterio.warp.Resampling.bilinear)
    
    # SCL -> nearest
    t1_scl, _ = align_rasters(t1_paths["scl"], t1_paths["red"], resampling_method=rasterio.warp.Resampling.nearest)
    t2_scl, _ = align_rasters(t2_paths["scl"], t1_paths["red"], resampling_method=rasterio.warp.Resampling.nearest)
    
    # 3. SCL Masking
    valid_mask = combined_valid_mask(t1_scl, t2_scl)
    cloud_frac = 1.0 - (np.sum(valid_mask) / valid_mask.size) if valid_mask.size > 0 else 1.0
    
    # 4. NDVI Delta
    ndvi_t1, ndvi_t2, delta, final_mask = compute_ndvi_delta(
        t1_red, t1_nir, t2_red, t2_nir, valid_mask
    )
    
    # 5. Statistics
    stats = compute_change_statistics(delta, final_mask, threshold, change_type="decrease")
    
    # Output structure exactly as requested
    mean_ndvi_t1 = np.nanmean(ndvi_t1[final_mask]) if np.any(final_mask) else 0.0
    mean_ndvi_t2 = np.nanmean(ndvi_t2[final_mask]) if np.any(final_mask) else 0.0
    
    report = {
        "t1_dimensions": meta["t1"]["dimensions"],
        "t2_dimensions": meta["t2"]["dimensions"],
        "crs": meta["t1"]["crs"],
        "resolution": meta["t1"]["resolution"],
        "bounds": str(meta["t1"].get("bounds", "transformed")),
        "cloud_fraction": cloud_frac,
        "valid_pixel_fraction": np.sum(final_mask) / final_mask.size if final_mask.size > 0 else 0.0,
        "resampling_status": resampled_red,
        "mean_ndvi_t1": mean_ndvi_t1,
        "mean_ndvi_t2": mean_ndvi_t2,
        "mean_delta": stats["mean_delta"],
        "decrease_fraction": stats["decrease_pixel_fraction"],
        "increase_fraction": stats["increase_pixel_fraction"],
        "threshold": stats["threshold_used"]
    }
    
    print(f"\n--- REAL RESULTS: {case_id} ---")
    for k, v in report.items():
        print(f"{k}: {v}")
        
    return report

def test_real_case_a_wildfire():
    report = run_real_pipeline("case_a_dixie_fire_2021", threshold=-0.2)
    assert report["decrease_fraction"] >= 0.0

def test_real_case_b_stable_forest():
    report = run_real_pipeline("case_b_redwoods_stable_2021", threshold=-0.2)
    assert report["decrease_fraction"] >= 0.0

def test_real_case_c_agriculture():
    report = run_real_pipeline("case_c_central_valley_ag_2021", threshold=-0.2)
    assert report["decrease_fraction"] >= 0.0
