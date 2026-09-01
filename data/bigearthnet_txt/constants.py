"""
Constants and definitions for BigEarthNet.txt and BigEarthNet-MM / BigEarthNet v2.0.

Provides standard band names, resolutions, spectral properties, official normalization statistics,
task categories, and split configurations.
"""

from typing import Dict, List

# Sentinel-1 SAR Bands (Polarizations)
S1_BAND_NAMES: List[str] = ["VV", "VH"]

# Sentinel-2 ALL L2A Bands — 12 raw bands from the GeoTIFF tiles
# Ordered as stored: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
S2_BAND_NAMES: List[str] = [
    "B01",  # idx 0  — Coastal aerosol (60m)  — EXCLUDED from model
    "B02",  # idx 1  — Blue (10m)
    "B03",  # idx 2  — Green (10m)
    "B04",  # idx 3  — Red (10m)
    "B05",  # idx 4  — Red Edge 1 (20m)
    "B06",  # idx 5  — Red Edge 2 (20m)
    "B07",  # idx 6  — Red Edge 3 (20m)
    "B08",  # idx 7  — NIR Broad (10m)
    "B8A",  # idx 8  — Narrow NIR (20m)
    "B09",  # idx 9  — Water vapour (60m)     — EXCLUDED from model
    "B11",  # idx 10 — SWIR 1 (20m)
    "B12",  # idx 11 — SWIR 2 (20m)
]

# -----------------------------------------------------------------------
# Single source of truth: the 10 Sentinel-2 bands consumed by S2Encoder.
# Excludes 60 m coarse bands B01 (Coastal aerosol) and B09 (Water vapour).
# Channel order inside S2Encoder input tensor:
#   0 → B02  (Blue,       10m)
#   1 → B03  (Green,      10m)
#   2 → B04  (Red,        10m)
#   3 → B05  (Red Edge 1, 20m)
#   4 → B06  (Red Edge 2, 20m)
#   5 → B07  (Red Edge 3, 20m)
#   6 → B08  (NIR Broad,  10m)
#   7 → B8A  (Narrow NIR, 20m)
#   8 → B11  (SWIR 1,     20m)
#   9 → B12  (SWIR 2,     20m)
# -----------------------------------------------------------------------
MODEL_S2_BANDS: List[str] = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

# Mapping: raw 12-band index → model 10-band index, for bands included in MODEL_S2_BANDS.
# raw index is the position in S2_BAND_NAMES.
# B01 (raw idx 0) and B09 (raw idx 9) are absent — they are excluded.
RAW_S2_TO_MODEL_INDEX: Dict[str, int] = {
    band: MODEL_S2_BANDS.index(band)
    for band in MODEL_S2_BANDS
}

# Predefined Band Combinations
PREDEFINED_BAND_COMBINATIONS: Dict[str, List[str]] = {
    "S1": S1_BAND_NAMES,
    "RGB": ["B04", "B03", "B02"],
    "S2-10m": ["B02", "B03", "B04", "B08"],
    "S2-10m20m": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "S2-all": S2_BAND_NAMES,
    "S1S2-10m20m": ["VV", "VH", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    "all": S1_BAND_NAMES + S2_BAND_NAMES,
}

# Native Spatial Resolutions (meters per pixel)
BAND_RESOLUTIONS: Dict[str, int] = {
    "VV": 10,
    "VH": 10,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B08": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B8A": 20,
    "B11": 20,
    "B12": 20,
    "B01": 60,
    "B09": 60,
}

# Standard Pixel Dimensions at Native Resolution for 1.2km x 1.2km patch
NATIVE_SHAPES: Dict[str, int] = {
    "VV": 120,
    "VH": 120,
    "B02": 120,
    "B03": 120,
    "B04": 120,
    "B08": 120,
    "B05": 60,
    "B06": 60,
    "B07": 60,
    "B8A": 60,
    "B11": 60,
    "B12": 60,
    "B01": 20,
    "B09": 20,
}

# Default Standardized Spatial Dimension for Multi-modal alignment
DEFAULT_IMG_SIZE: int = 120

# Official BigEarthNet Band Means (calculated across official training split)
BAND_MEANS: Dict[str, float] = {
    "B01": 361.0767822265625,
    "B02": 438.3720703125,
    "B03": 614.0556640625,
    "B04": 588.4096069335938,
    "B05": 942.8433227539062,
    "B06": 1769.931640625,
    "B07": 2049.551513671875,
    "B08": 2193.2919921875,
    "B09": 2241.455322265625,
    "B11": 1568.226806640625,
    "B12": 997.7324829101562,
    "B8A": 2235.556640625,
    "VH": -19.352558135986328,
    "VV": -12.643863677978516,
}

# Official BigEarthNet Band Standard Deviations
BAND_STDS: Dict[str, float] = {
    "B01": 575.0687255859375,
    "B02": 607.02685546875,
    "B03": 603.2968139648438,
    "B04": 684.56884765625,
    "B05": 738.4326782226562,
    "B06": 1100.4560546875,
    "B07": 1275.805419921875,
    "B08": 1369.3717041015625,
    "B09": 1316.393310546875,
    "B11": 1070.1612548828125,
    "B12": 813.5276489257812,
    "B8A": 1356.5440673828125,
    "VH": 5.590505599975586,
    "VV": 5.133493900299072,
}

# BigEarthNet.txt Task Types & Categories
VALID_TASK_TYPES = {
    "binary",
    "mcq",
    "captioning",
    "bounding box",
}

VALID_CATEGORIES = {
    "presence",
    "area",
    "counting",
    "adjacency",
    "relative position",
    "country",
    "season",
    "climate zone",
}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
    "bench",
}

EXPECTED_METADATA_COLUMNS = {
    "ID",
    "patch_id",
    "s1_name",
    "input",
    "output",
    "type",
    "category",
    "split",
    "latitude",
    "longitude",
    "country",
    "season",
    "climate_zone",
}
