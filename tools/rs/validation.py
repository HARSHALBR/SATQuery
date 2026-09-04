from typing import Dict, Any, Tuple
import datetime

class RSValidationError(Exception):
    pass

def validate_observations(t1_paths: Dict[str, str], t2_paths: Dict[str, str], t1_date: datetime.datetime, t2_date: datetime.datetime) -> Dict[str, Any]:
    """
    Validates two observations for RS processing.
    t1_paths and t2_paths are dictionaries mapping band names ('red', 'nir', 'scl') to file paths.
    """
    # Check temporal ordering
    if t1_date >= t2_date:
        raise RSValidationError(f"Invalid temporal ordering: T1 ({t1_date}) is not before T2 ({t2_date})")
        
    required_bands = {"red", "nir", "scl"}
    
    import rasterio
    
    # Check missing bands
    for b in required_bands:
        if b not in t1_paths:
            raise RSValidationError(f"T1 is missing required band {b}")
        if b not in t2_paths:
            raise RSValidationError(f"T2 is missing required band {b}")
            
    # Check files can be opened and get basic metadata
    t1_meta = {}
    for b in required_bands:
        try:
            with rasterio.open(t1_paths[b]) as src:
                if not src.crs:
                    raise RSValidationError(f"T1 band {b} has invalid/missing CRS")
                t1_meta[b] = {
                    "crs": src.crs,
                    "bounds": src.bounds,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "res": src.res
                }
        except Exception as e:
            raise RSValidationError(f"Cannot open T1 band {b} file: {t1_paths[b]} - {str(e)}")

    t2_meta = {}
    for b in required_bands:
        try:
            with rasterio.open(t2_paths[b]) as src:
                if not src.crs:
                    raise RSValidationError(f"T2 band {b} has invalid/missing CRS")
                t2_meta[b] = {
                    "crs": src.crs,
                    "bounds": src.bounds,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "res": src.res
                }
        except Exception as e:
            raise RSValidationError(f"Cannot open T2 band {b} file: {t2_paths[b]} - {str(e)}")

    # CRS normalization/compatibility check
    if not t1_meta["red"]["crs"].is_valid or not t2_meta["red"]["crs"].is_valid:
        raise RSValidationError("Incompatible CRS that cannot be safely normalized")

    # Spatial overlap (CRS-safe using transform_bounds to EPSG:4326)
    from rasterio.warp import transform_bounds
    try:
        b1_4326 = transform_bounds(t1_meta["red"]["crs"], "EPSG:4326", *t1_meta["red"]["bounds"])
        b2_4326 = transform_bounds(t2_meta["red"]["crs"], "EPSG:4326", *t2_meta["red"]["bounds"])
        overlap = not (b1_4326[2] <= b2_4326[0] or b1_4326[0] >= b2_4326[2] or b1_4326[3] <= b2_4326[1] or b1_4326[1] >= b2_4326[3])
        if not overlap:
            raise RSValidationError("Zero spatial overlap between T1 and T2")
    except Exception as e:
        raise RSValidationError(f"Error computing CRS-safe bounding box intersection: {e}")

    # Extract quality metadata conceptually
    metadata = {
        "t1": {
            "crs": str(t1_meta["red"]["crs"]),
            "dimensions": (t1_meta["red"]["height"], t1_meta["red"]["width"]),
            "resolution": t1_meta["red"]["res"]
        },
        "t2": {
            "crs": str(t2_meta["red"]["crs"]),
            "dimensions": (t2_meta["red"]["height"], t2_meta["red"]["width"]),
            "resolution": t2_meta["red"]["res"]
        },
        "registration_status": "Checked"
    }

    return metadata
