import numpy as np
from typing import Dict, Any, Tuple

def align_rasters(src_path: str, ref_path: str, resampling_method=None):
    import rasterio
    from rasterio.warp import reproject, Resampling
    """
    Reads src_path and ref_path. If their grid/CRS mismatch, resamples src onto ref's grid.
    Returns the aligned src numpy array and a boolean indicating if resampling occurred.
    """
    if resampling_method is None:
        resampling_method = Resampling.nearest

    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
        
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_shape = (src.height, src.width)
        
        # Check if resampling is needed
        needs_resampling = False
        if src_crs != ref_crs or src_transform != ref_transform or src_shape != ref_shape:
            needs_resampling = True
            
        if not needs_resampling:
            return src.read(1), False
            
        # Allocate destination array using source nodata if available, else 0
        fill_value = src.nodata if src.nodata is not None else 0
        dest_array = np.full(ref_shape, fill_value, dtype=src.dtypes[0])
        
        reproject(
            source=rasterio.band(src, 1),
            destination=dest_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling_method
        )
        
        return dest_array, True
