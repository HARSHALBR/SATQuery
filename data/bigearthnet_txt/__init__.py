"""
BigEarthNet.txt multimodal data engineering package.
"""

from data.bigearthnet_txt.constants import (
    BAND_MEANS,
    BAND_RESOLUTIONS,
    BAND_STDS,
    DEFAULT_IMG_SIZE,
    EXPECTED_METADATA_COLUMNS,
    MODEL_S2_BANDS,
    NATIVE_SHAPES,
    PREDEFINED_BAND_COMBINATIONS,
    RAW_S2_TO_MODEL_INDEX,
    S1_BAND_NAMES,
    S2_BAND_NAMES,
    VALID_CATEGORIES,
    VALID_SPLITS,
    VALID_TASK_TYPES,
)
from data.bigearthnet_txt.dataset import BigEarthNetDataset, collate_bigearthnet
from data.bigearthnet_txt.parser import BigEarthNetParser
from data.bigearthnet_txt.transforms import (
    MultiBandNormalize,
    MultiBandResize,
    RandomSpatialAugmentation,
    build_transform,
)
from data.bigearthnet_txt.utils import (
    compute_file_hash,
    read_geotiff,
    validate_coordinates,
    validate_patch_bands,
    validate_s1_s2_pairing,
)

__all__ = [
    "BigEarthNetDataset",
    "collate_bigearthnet",
    "BigEarthNetParser",
    "MultiBandNormalize",
    "MultiBandResize",
    "RandomSpatialAugmentation",
    "build_transform",
    "read_geotiff",
    "validate_patch_bands",
    "validate_s1_s2_pairing",
    "validate_coordinates",
    "compute_file_hash",
    "S1_BAND_NAMES",
    "S2_BAND_NAMES",
    "MODEL_S2_BANDS",
    "RAW_S2_TO_MODEL_INDEX",
    "PREDEFINED_BAND_COMBINATIONS",
    "BAND_RESOLUTIONS",
    "NATIVE_SHAPES",
    "DEFAULT_IMG_SIZE",
    "BAND_MEANS",
    "BAND_STDS",
    "VALID_TASK_TYPES",
    "VALID_CATEGORIES",
    "VALID_SPLITS",
    "EXPECTED_METADATA_COLUMNS",
]
