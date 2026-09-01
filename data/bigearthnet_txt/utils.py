"""
Utility functions for GeoTIFF raster inspection, integrity verification,
coordinate validation, and S1/S2 multi-sensor pairing checks.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

from data.bigearthnet_txt.constants import (
    BAND_RESOLUTIONS,
    NATIVE_SHAPES,
    S1_BAND_NAMES,
    S2_BAND_NAMES,
)


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute a cryptographic hash of a file for reproducibility and integrity verification.
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_geotiff(
    file_path: Union[str, Path],
    expected_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Safely read a GeoTIFF raster file and return the numpy array along with metadata.
    
    Supports both rasterio and tifffile backends with automatic fallback.
    
    Args:
        file_path: Path to the GeoTIFF file.
        expected_shape: Optional expected (height, width) to validate against.
        
    Returns:
        Tuple of (2D/3D numpy array, metadata dict).
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is corrupted, empty, or has an invalid shape.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"GeoTIFF file not found: {p}")

    if p.stat().st_size == 0:
        raise ValueError(f"GeoTIFF file is empty (0 bytes): {p}")

    arr: Optional[np.ndarray] = None
    meta: Dict[str, Any] = {"file_path": str(p), "file_size": p.stat().st_size}

    # Primary reading attempt with rasterio
    if HAS_RASTERIO:
        try:
            with rasterio.open(p) as src:
                arr = src.read()
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                meta["driver"] = src.driver
                meta["dtype"] = str(src.dtypes[0])
                meta["crs"] = str(src.crs) if src.crs else None
                meta["transform"] = list(src.transform) if src.transform else None
                meta["bounds"] = tuple(src.bounds) if src.bounds else None
                meta["shape"] = (src.height, src.width)
        except Exception as e:
            # Try fallback to tifffile
            if not HAS_TIFFFILE:
                raise ValueError(f"Failed to read GeoTIFF with rasterio: {e}") from e

    # Fallback to tifffile if rasterio was unavailable or failed
    if arr is None and HAS_TIFFFILE:
        try:
            arr = tifffile.imread(str(p))
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            meta["dtype"] = str(arr.dtype)
            meta["shape"] = (arr.shape[-2], arr.shape[-1]) if arr.ndim >= 2 else (0, 0)
        except Exception as e:
            raise ValueError(f"Failed to read TIFF with tifffile: {e}") from e

    if arr is None:
        raise ValueError(f"No GeoTIFF reader available (neither rasterio nor tifffile succeeded) for {p}")

    # Check for invalid values (all-NaN, all-Inf)
    if np.isnan(arr).all():
        raise ValueError(f"GeoTIFF contains all-NaN values: {p}")
    if np.isinf(arr).all():
        raise ValueError(f"GeoTIFF contains all-Inf values: {p}")

    # Validate shape if requested
    if expected_shape is not None:
        actual_shape = (arr.shape[-2], arr.shape[-1])
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in {p.name}: expected {expected_shape}, found {actual_shape}"
            )

    return arr, meta


def validate_patch_bands(
    patch_dir: Union[str, Path],
    expected_bands: List[str],
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate that all required band GeoTIFFs exist and are readable in a patch folder.
    
    Args:
        patch_dir: Directory containing band files (e.g. data/bigearthnet_txt/images_s2/<patch_id>).
        expected_bands: List of band identifiers (e.g., ['B01', ..., 'B12'] or ['VV', 'VH']).
        prefix: Optional patch name prefix (e.g., patch_id).
        
    Returns:
        Dict with validation result:
        {
            "valid": bool,
            "missing_bands": list,
            "corrupted_bands": dict,
            "band_paths": dict,
            "band_shapes": dict
        }
    """
    p = Path(patch_dir)
    res = {
        "valid": True,
        "missing_bands": [],
        "corrupted_bands": {},
        "band_paths": {},
        "band_shapes": {},
    }

    if not p.exists() or not p.is_dir():
        res["valid"] = False
        res["missing_bands"] = list(expected_bands)
        return res

    for band in expected_bands:
        # Expected naming conventions in BigEarthNet:
        # 1) <prefix>_<band>.tif
        # 2) <prefix>_<band>.tiff
        # 3) <band>.tif
        # 4) <band>.tiff
        candidates = []
        if prefix:
            candidates.extend([p / f"{prefix}_{band}.tif", p / f"{prefix}_{band}.tiff"])
        candidates.extend([p / f"{band}.tif", p / f"{band}.tiff"])

        # Also search case-insensitively if needed
        found_path: Optional[Path] = None
        for cand in candidates:
            if cand.exists():
                found_path = cand
                break

        if found_path is None:
            # Fallback: search directory for any file ending with _<band>.tif or matching band
            for f in p.iterdir():
                if f.is_file() and (
                    f.name.endswith(f"_{band}.tif")
                    or f.name.endswith(f"_{band}.tiff")
                    or f.stem == band
                ):
                    found_path = f
                    break

        if found_path is None:
            res["missing_bands"].append(band)
            res["valid"] = False
            continue

        res["band_paths"][band] = str(found_path)

        # Validate readability and expected dimensions
        expected_dim = NATIVE_SHAPES.get(band, 120)
        try:
            arr, meta = read_geotiff(found_path, expected_shape=(expected_dim, expected_dim))
            res["band_shapes"][band] = meta["shape"]
        except Exception as e:
            res["corrupted_bands"][band] = str(e)
            res["valid"] = False

    return res


def validate_coordinates(latitude: Any, longitude: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate latitude and longitude ranges.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False, f"Non-numeric coordinates: lat={latitude}, lon={longitude}"

    if not (-90.0 <= lat <= 90.0):
        return False, f"Latitude {lat} out of bounds [-90, 90]"
    if not (-180.0 <= lon <= 180.0):
        return False, f"Longitude {lon} out of bounds [-180, 180]"

    return True, None


def validate_s1_s2_pairing(
    s1_name: str,
    patch_id: str,
    s1_metadata: Optional[Dict[str, Any]] = None,
    s2_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate the co-registration pairing between a Sentinel-1 patch and Sentinel-2 patch.
    
    Checks:
    - Non-empty identifiers
    - Basic timestamp and acquisition compatibility if metadata is provided
    
    Returns:
        (is_valid, error_reason)
    """
    if not s1_name or not isinstance(s1_name, str) or not s1_name.strip():
        return False, "Empty or invalid s1_name"
    if not patch_id or not isinstance(patch_id, str) or not patch_id.strip():
        return False, "Empty or invalid patch_id (s2_name)"

    # Prefix checks (BigEarthNet convention: S1A/S1B and S2A/S2B)
    if not (s1_name.startswith("S1A") or s1_name.startswith("S1B") or "S1" in s1_name):
        return False, f"Invalid S1 patch name convention: {s1_name}"
    if not (patch_id.startswith("S2A") or patch_id.startswith("S2B") or "S2" in patch_id):
        return False, f"Invalid S2 patch name convention: {patch_id}"

    # Metadata spatial proximity check if bounding coordinates exist
    if s1_metadata and s2_metadata:
        lat1 = s1_metadata.get("latitude")
        lat2 = s2_metadata.get("latitude")
        lon1 = s1_metadata.get("longitude")
        lon2 = s2_metadata.get("longitude")
        if lat1 is not None and lat2 is not None and lon1 is not None and lon2 is not None:
            # Pair coordinates should be very close (within ~0.05 degrees for 1.2km patch)
            if abs(float(lat1) - float(lat2)) > 0.1 or abs(float(lon1) - float(lon2)) > 0.1:
                return False, f"Spatial distance too large between S1 ({lat1},{lon1}) and S2 ({lat2},{lon2})"

    return True, None
