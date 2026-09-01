import rasterio
from rasterio.enums import Resampling
import numpy as np
from rasterio.windows import from_bounds

from pathlib import Path
from geospatial.alignment import check_alignment

def create_evidence_score(
    ndvi_change_path,
    ndbi_change_path,
    vv_change_path,
    vh_change_path,
    score_output_path,
    classification_output_path,
    *,
    ndvi_threshold=-0.20,
    ndbi_threshold=0.20,
    vv_threshold=-0.07333232,
    vh_threshold=-0.02110010,
    expected_nodata=255
):
    """
    Calculate dynamic evidence score and classification from aligned change rasters.
    """
    # Verify exact alignment between all four inputs
    align1 = check_alignment(ndvi_change_path, ndbi_change_path)
    align2 = check_alignment(ndvi_change_path, vv_change_path)
    align3 = check_alignment(ndvi_change_path, vh_change_path)

    if not (align1["aligned"] and align2["aligned"] and align3["aligned"]):
        raise ValueError("Inputs are not perfectly spatially aligned. Run Step 6 alignment first.")

    with rasterio.open(ndvi_change_path) as ndvi_src, \
         rasterio.open(ndbi_change_path) as ndbi_src, \
         rasterio.open(vv_change_path) as vv_src, \
         rasterio.open(vh_change_path) as vh_src:

        ndvi = ndvi_src.read(1).astype(np.float32)
        ndbi = ndbi_src.read(1).astype(np.float32)
        vv = vv_src.read(1).astype(np.float32)
        vh = vh_src.read(1).astype(np.float32)

        ndvi_nodata = ndvi_src.nodata if ndvi_src.nodata is not None else -9999.0
        ndbi_nodata = ndbi_src.nodata if ndbi_src.nodata is not None else -9999.0
        vv_nodata = vv_src.nodata if vv_src.nodata is not None else -9999.0
        vh_nodata = vh_src.nodata if vh_src.nodata is not None else -9999.0

        profile = ndvi_src.profile.copy()

    # Create a strict validity mask
    valid = (
        (ndvi != ndvi_nodata) & (ndbi != ndbi_nodata) &
        (vv != vv_nodata) & (vh != vh_nodata) &
        np.isfinite(ndvi) & np.isfinite(ndbi) &
        np.isfinite(vv) & np.isfinite(vh)
    )

    # Calculate evidence
    ndvi_evidence = valid & (ndvi < ndvi_threshold)
    ndbi_evidence = valid & (ndbi > ndbi_threshold)
    vv_evidence = valid & (vv < vv_threshold)
    vh_evidence = valid & (vh < vh_threshold)

    score = np.full(ndvi.shape, expected_nodata, dtype=np.uint8)
    
    # Calculate score only where valid
    valid_score = np.zeros(ndvi.shape, dtype=np.uint8)
    valid_score[ndvi_evidence] += 1
    valid_score[ndbi_evidence] += 1
    valid_score[vv_evidence] += 1
    valid_score[vh_evidence] += 1
    
    score[valid] = valid_score[valid]

    # Save Score
    profile.update(
        dtype="uint8",
        nodata=expected_nodata,
        count=1
    )
    with rasterio.open(score_output_path, "w", **profile) as dst:
        dst.write(score, 1)

    # Calculate Classification
    optical_count = np.zeros(ndvi.shape, dtype=np.uint8)
    optical_count[valid] = ndvi_evidence[valid].astype(np.uint8) + ndbi_evidence[valid].astype(np.uint8)
    
    sar_count = np.zeros(ndvi.shape, dtype=np.uint8)
    sar_count[valid] = vv_evidence[valid].astype(np.uint8) + vh_evidence[valid].astype(np.uint8)

    classification = np.full(ndvi.shape, expected_nodata, dtype=np.uint8)

    # Strong consensus
    strong_consensus = valid & (optical_count >= 1) & (sar_count >= 1) & (valid_score >= 3)
    
    # Uncertain
    uncertain = valid & (
        ((optical_count == 2) & (sar_count == 0)) |
        ((optical_count == 0) & (sar_count == 2))
    )

    # 0 = OTHER (valid pixels that aren't strong or uncertain)
    classification[valid] = 0
    classification[strong_consensus] = 2
    classification[uncertain] = 1

    with rasterio.open(classification_output_path, "w", **profile) as dst:
        dst.write(classification, 1)

    valid_pixels = int(valid.sum())
    strong_pixels = int(strong_consensus.sum())
    uncertain_pixels = int(uncertain.sum())
    other_pixels = valid_pixels - strong_pixels - uncertain_pixels

    return {
        "score_output": str(score_output_path),
        "classification_output": str(classification_output_path),
        "valid_pixels": valid_pixels,
        "ndvi_evidence_pixels": int(ndvi_evidence.sum()),
        "ndbi_evidence_pixels": int(ndbi_evidence.sum()),
        "vv_evidence_pixels": int(vv_evidence.sum()),
        "vh_evidence_pixels": int(vh_evidence.sum()),
        "score_counts": {
            str(value): int(np.sum(valid_score[valid] == value))
            for value in range(5)
        },
        "strong_evidence_percentage": (float(strong_pixels / valid_pixels * 100) if valid_pixels > 0 else 0.0),
        "classification": {
            "strong": strong_pixels,
            "uncertain": uncertain_pixels,
            "other": other_pixels
        },
        "thresholds": {
            "ndvi": ndvi_threshold,
            "ndbi": ndbi_threshold,
            "vv": vv_threshold,
            "vh": vh_threshold
        }
    }

if __name__ == "__main__":
    import sys
    print("create_evidence_score.py should be invoked via orchestration.")
    sys.exit(1)