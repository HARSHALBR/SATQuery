"""
geospatial/alignment.py — Spatial alignment utilities for SatQuery.

Functions:
  check_alignment       — verify two rasters share CRS, shape, and transform
  calculate_geographic_overlap — compute % geographic overlap between two rasters
  create_analysis_grid  — build a canonical grid definition from a reference raster
  verify_pixel_alignment — strict pixel-level alignment check
  align_raster          — reproject/resample one raster to match a reference grid
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# check_alignment
# ---------------------------------------------------------------------------

def check_alignment(path_a: PathLike, path_b: PathLike) -> dict:
    """
    Check whether two rasters are spatially aligned (same CRS, shape, transform).

    Returns a dict::
        {
            "valid":   bool,   # both files opened without error
            "aligned": bool,   # True when CRS + shape + transform all match
            "error":   str | None,
            "details": dict,
        }
    """
    try:
        with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
            crs_match = a.crs == b.crs
            shape_match = (a.width == b.width) and (a.height == b.height)
            # Allow small floating-point tolerance in transform comparison
            t_a = np.array(a.transform)[:6]
            t_b = np.array(b.transform)[:6]
            transform_match = bool(np.allclose(t_a, t_b, rtol=1e-5, atol=1e-5))

            aligned = crs_match and shape_match and transform_match
            return {
                "valid": True,
                "aligned": aligned,
                "error": None if aligned else "CRS/shape/transform mismatch",
                "details": {
                    "crs_match": crs_match,
                    "shape_match": shape_match,
                    "transform_match": transform_match,
                    "crs_a": str(a.crs),
                    "crs_b": str(b.crs),
                    "shape_a": (a.height, a.width),
                    "shape_b": (b.height, b.width),
                },
            }
    except Exception as exc:
        return {
            "valid": False,
            "aligned": False,
            "error": str(exc),
            "details": {},
        }


# ---------------------------------------------------------------------------
# calculate_geographic_overlap
# ---------------------------------------------------------------------------

def calculate_geographic_overlap(path_a: PathLike, path_b: PathLike) -> dict:
    """
    Calculate the geographic (bounding-box) overlap between two rasters.

    Returns a dict with ``overlap_fraction`` (0.0–1.0) and bbox coordinates.
    """
    try:
        with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
            # Bounds in native CRS
            ba = a.bounds
            bb = b.bounds

            # Intersection
            ix_left   = max(ba.left,   bb.left)
            ix_bottom = max(ba.bottom, bb.bottom)
            ix_right  = min(ba.right,  bb.right)
            ix_top    = min(ba.top,    bb.top)

            if ix_right <= ix_left or ix_top <= ix_bottom:
                overlap_fraction = 0.0
            else:
                area_intersection = (ix_right - ix_left) * (ix_top - ix_bottom)
                area_a = (ba.right - ba.left) * (ba.top - ba.bottom)
                area_b = (bb.right - bb.left) * (bb.top - bb.bottom)
                area_union = area_a + area_b - area_intersection
                overlap_fraction = area_intersection / area_union if area_union > 0 else 0.0

            return {
                "overlap_fraction": overlap_fraction,
                "intersection": {
                    "left": ix_left, "bottom": ix_bottom,
                    "right": ix_right, "top": ix_top,
                },
                "bounds_a": dict(ba._asdict()),
                "bounds_b": dict(bb._asdict()),
            }
    except Exception as exc:
        return {"overlap_fraction": 0.0, "error": str(exc)}


# ---------------------------------------------------------------------------
# create_analysis_grid
# ---------------------------------------------------------------------------

def create_analysis_grid(reference_path: PathLike) -> dict:
    """
    Build a canonical grid definition (CRS, transform, width, height) from
    a reference raster.  All rasters aligned to this grid will be pixel-perfect.

    Returns::
        {
            "crs":       CRS object,
            "transform": Affine transform,
            "width":     int,
            "height":    int,
            "bounds":    BoundingBox,
            "nodata":    float | None,
        }
    """
    with rasterio.open(reference_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }


# ---------------------------------------------------------------------------
# verify_pixel_alignment
# ---------------------------------------------------------------------------

def verify_pixel_alignment(path_a: PathLike, path_b: PathLike) -> bool:
    """
    Strict pixel-level alignment check: same CRS, width, height, *and* transform.
    Returns True only when all four match exactly (within floating-point tolerance).
    """
    result = check_alignment(path_a, path_b)
    return result["valid"] and result["aligned"]


# ---------------------------------------------------------------------------
# align_raster
# ---------------------------------------------------------------------------

def align_raster(
    src_path: PathLike,
    grid: dict,
    dst_path: PathLike,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Reproject and resample *src_path* so that it exactly matches *grid*
    (as returned by :func:`create_analysis_grid`).

    Writes the result to *dst_path* and returns its ``Path``.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata if src.nodata is not None else -9999.0
        src_dtype = src.dtypes[0]
        band_count = src.count

        profile = src.profile.copy()
        profile.update(
            crs=grid["crs"],
            transform=grid["transform"],
            width=grid["width"],
            height=grid["height"],
            nodata=src_nodata,
            dtype=src_dtype,
            count=band_count,
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, band_count + 1):
                src_data = src.read(band_idx)
                dst_data = np.full(
                    (grid["height"], grid["width"]),
                    src_nodata,
                    dtype=np.dtype(src_dtype),
                )
                reproject(
                    source=src_data,
                    destination=dst_data,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=grid["transform"],
                    dst_crs=grid["crs"],
                    resampling=resampling,
                    src_nodata=src_nodata,
                    dst_nodata=src_nodata,
                )
                dst.write(dst_data, band_idx)

    return dst_path
