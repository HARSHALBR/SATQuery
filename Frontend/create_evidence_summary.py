import json
import rasterio
import numpy as np
from rasterio.enums import Resampling

def create_evidence_summary(
    evidence_score_file,
    ndvi_change_file,
    ndbi_change_file,
    vv_change_file,
    vh_change_file,
    output_summary_file,
    ndvi_threshold=-0.20,
    ndbi_threshold=0.20,
    vv_threshold=-0.07333232,
    vh_threshold=-0.02110010,
    processor_name="SatQuery Dynamic Processor"
):
    """
    Dynamically create a summary of the evidence score raster and classification.
    """
    # Read evidence score raster
    with rasterio.open(evidence_score_file) as src:
        data = src.read(1)
        nodata = src.nodata
        # Capture metadata for later use
        crs = str(src.crs)
        width = src.width
        height = src.height
        res = list(src.res)
        score_valid = data != nodata
        valid_data = data[score_valid]

    # Load change rasters
    with rasterio.open(ndvi_change_file) as src_ndvi:
        ndvi = src_ndvi.read(1)
        ndvi_nodata = src_ndvi.nodata
    with rasterio.open(ndbi_change_file) as src_ndbi:
        ndbi = src_ndbi.read(1)
        ndbi_nodata = src_ndbi.nodata
    with rasterio.open(vv_change_file) as src_vv:
        vv = src_vv.read(1)
        vv_nodata = src_vv.nodata
    with rasterio.open(vh_change_file) as src_vh:
        vh = src_vh.read(1)
        vh_nodata = src_vh.nodata

    # Resample if dimensions differ
    if ndvi.shape != vv.shape:
        # Resample optical rasters to SAR shape using nearest neighbor
        with rasterio.open(ndvi_change_file) as src_ndvi:
            ndvi = src_ndvi.read(1, out_shape=vv.shape, resampling=Resampling.nearest)
        with rasterio.open(ndbi_change_file) as src_ndbi:
            ndbi = src_ndbi.read(1, out_shape=vv.shape, resampling=Resampling.nearest)

    # Valid mask across all rasters
    valid = ((ndvi != ndvi_nodata) & (ndbi != ndbi_nodata) &
             (vv != vv_nodata) & (vh != vh_nodata) &
             np.isfinite(ndvi) & np.isfinite(ndbi) &
             np.isfinite(vv) & np.isfinite(vh))

    # Evidence per indicator
    ndvi_evidence = valid & (ndvi < ndvi_threshold)
    ndbi_evidence = valid & (ndbi > ndbi_threshold)
    vv_evidence   = valid & (vv < vv_threshold)
    vh_evidence   = valid & (vh < vh_threshold)

    optical_count = ndvi_evidence.astype(np.uint8) + ndbi_evidence.astype(np.uint8)
    sar_count = vv_evidence.astype(np.uint8) + vh_evidence.astype(np.uint8)

    # Classification logic
    strong_consensus = valid & (optical_count >= 1) & (sar_count >= 1) & (data >= 3)
    uncertain = valid & (((optical_count == 2) & (sar_count == 0)) |
                        ((optical_count == 0) & (sar_count == 2)))

    strong_consensus_pixels = int(strong_consensus.sum())
    uncertain_pixels = int(uncertain.sum())
    other_pixels = int(valid.sum() - strong_consensus_pixels - uncertain_pixels)

    # Build summary
    summary = {
        "source": str(evidence_score_file),
        "provenance": {
            "processor": processor_name,
            "evidence_method": "optical_sar_score",
            "tool_version": "1.0"
        },
        "crs": crs,
        "width": width,
        "height": height,
        "resolution": res,
        "valid_pixels": int(valid.sum()),
        "nodata_pixels": int((~valid).sum()),
        "score_counts": {
            str(score): int(np.sum(valid_data == score))
            for score in range(5)
        },
        "strong_evidence": {
            "threshold": 3,
            "pixels": int(np.sum(valid_data >= 3)),
            "percentage": float(np.mean(valid_data >= 3) * 100) if len(valid_data) > 0 else 0.0
        },
        "classification": {
            "strong": strong_consensus_pixels,
            "uncertain": uncertain_pixels,
            "other": other_pixels
        }
    }

    with open(output_summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    import sys
    print("create_evidence_summary.py should be invoked via orchestration.")
    sys.exit(1)