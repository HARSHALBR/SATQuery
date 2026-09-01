import rasterio
import numpy as np
from scipy import ndimage

def cluster_evidence(
    score_file_path,
    output_file_path,
    score_threshold=3,
    min_pixels=5,
    nodata_val=255
):
    """
    Dynamically perform DBSCAN/connected-components clustering on a score raster.
    """
    with rasterio.open(score_file_path) as src:
        score = src.read(1)
        profile = src.profile.copy()
        
    # Valid pixels
    valid = score != nodata_val
    
    # Strong evidence pixels
    strong = valid & (score >= score_threshold)
    
    # 8-neighbour connectivity
    structure = np.ones((3, 3), dtype=np.uint8)
    
    labels, number_of_regions = ndimage.label(
        strong,
        structure=structure
    )
    
    # Calculate region sizes
    sizes = np.bincount(labels.ravel())
    
    # Ignore background label 0
    if len(sizes) > 1:
        region_sizes = sizes[1:]
    else:
        region_sizes = np.array([])
        
    # Remove tiny regions
    filtered = np.zeros_like(labels, dtype=np.uint16)
    new_id = 1
    
    for old_id, size in enumerate(region_sizes, start=1):
        if size >= min_pixels:
            filtered[labels == old_id] = new_id
            new_id += 1
            
    final_regions = new_id - 1
    
    # Save clustered raster
    profile.update(
        dtype="uint16",
        count=1,
        nodata=0
    )
    
    with rasterio.open(output_file_path, "w", **profile) as dst:
        dst.write(filtered, 1)
        
    return {
        "cluster_output": str(output_file_path),
        "total_initial_regions": int(number_of_regions),
        "total_final_regions": int(final_regions),
        "min_pixels_threshold": min_pixels,
        "score_threshold": score_threshold,
        "largest_region_pixels": int(region_sizes.max()) if len(region_sizes) else 0,
        "strong_pixels_total": int(strong.sum())
    }

if __name__ == "__main__":
    import sys
    print("cluster_evidence.py should be invoked via orchestration.")
    sys.exit(1)
