import rasterio
import numpy as np
from pathlib import Path
from pipeline_config import DATA_DIR, BEFORE_YEAR, AFTER_YEAR

from geospatial.alignment import check_alignment

def calculate_sar_change(
    vv_before_path: Path,
    vv_after_path: Path,
    vh_before_path: Path,
    vh_after_path: Path,
    vv_output_path: Path,
    vh_output_path: Path
):
    """
    Calculate dynamic SAR change for VV and VH bands.
    Validates input alignment and writes to current workspace.
    """
    def process_band(before_path, after_path, output_path, band_name):
        with rasterio.open(before_path) as old_src, rasterio.open(after_path) as new_src:
            align_check = check_alignment(before_path, after_path)
            if not align_check["valid"] or not align_check["aligned"]:
                raise ValueError(f"{band_name} alignment failed: {align_check['error'] or 'Geospatial mismatch (CRS, dimensions, transform, or bounds)'}")
            
            old = old_src.read(1).astype(np.float32)
            new = new_src.read(1).astype(np.float32)

            valid = (
                (old != old_src.nodata) &
                (new != new_src.nodata) &
                np.isfinite(old) &
                np.isfinite(new)
            )

            change = np.full(
                old.shape,
                -9999.0,
                dtype=np.float32
            )

            change[valid] = new[valid] - old[valid]

            profile = old_src.profile.copy()
            profile.update(
                dtype="float32",
                nodata=-9999.0,
                count=1
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(change, 1)

            values = change[valid]
            
            return {
                "valid_pixels": int(valid.sum()),
                "min": float(values.min()) if valid.sum() > 0 else 0.0,
                "max": float(values.max()) if valid.sum() > 0 else 0.0
            }
            
    # Validate cross-alignment between VV and VH
    cross_align_before = check_alignment(vv_before_path, vh_before_path)
    cross_align_after = check_alignment(vv_after_path, vh_after_path)
    
    if not cross_align_before["aligned"] or not cross_align_after["aligned"]:
        raise ValueError("VV and VH must be spatially compatible (cross-alignment failed).")
        
    vv_stats = process_band(vv_before_path, vv_after_path, vv_output_path, "VV")
    vh_stats = process_band(vh_before_path, vh_after_path, vh_output_path, "VH")
    
    return {
        "vv": vv_stats,
        "vh": vh_stats,
        "alignment": {
            "vv_before_after": "PASS",
            "vh_before_after": "PASS",
            "vv_vh_cross": "PASS"
        }
    }

if __name__ == "__main__":
    import sys
    print("calculate_sar_change.py should be invoked via orchestration.")
    sys.exit(1)