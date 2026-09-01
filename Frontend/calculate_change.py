import rasterio
import numpy as np


from pathlib import Path
from geospatial.alignment import check_alignment

def calculate_change(before_path, after_path, output_path, *, expected_nodata=-9999.0):
    """
    Calculate dynamic change (after - before) between two aligned rasters.
    """
    with rasterio.open(before_path) as before_src, rasterio.open(after_path) as after_src:

        # Check alignment strictly
        align_check = check_alignment(before_path, after_path)
        if not align_check["valid"] or not align_check["aligned"]:
            raise ValueError(f"Rasters are not spatially aligned: {align_check['error'] or 'Mismatch'}")

        before = before_src.read(1).astype(np.float32)
        after = after_src.read(1).astype(np.float32)

        before_nodata = before_src.nodata if before_src.nodata is not None else expected_nodata
        after_nodata = after_src.nodata if after_src.nodata is not None else expected_nodata

        valid = (
            (before != before_nodata) &
            (after != after_nodata) &
            np.isfinite(before) &
            np.isfinite(after)
        )

        change = np.full(
            before.shape,
            expected_nodata,
            dtype=np.float32
        )

        change[valid] = after[valid] - before[valid]

        profile = before_src.profile.copy()

        profile.update(
            dtype="float32",
            nodata=expected_nodata,
            count=1
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(change, 1)

        values = change[valid]

        return {
            "output": str(output_path),
            "valid_pixels": int(valid.sum()),
            "min_change": float(values.min()) if valid.sum() > 0 else 0.0,
            "max_change": float(values.max()) if valid.sum() > 0 else 0.0,
            "mean_change": float(values.mean()) if valid.sum() > 0 else 0.0,
            "median_change": float(np.median(values)) if valid.sum() > 0 else 0.0,
            "nodata": expected_nodata,
            "formula": "after - before"
        }