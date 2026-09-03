"""Satellite band loading and tensor preparation for RS-InternVL."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import torch
import torch.nn.functional as F


S1_BANDS: List[str] = ["VV", "VH"]

S2_BANDS: List[str] = [
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

MODEL_IMAGE_SIZE = 120


def build_band_paths(
    image_path: str,
    bands: List[str],
) -> Dict[str, str]:
    """Build the expected GeoTIFF path for each band."""

    base = Path(image_path)

    return {
        band: str(Path(f"{base}_{band}.tif"))
        for band in bands
    }


def load_band(path: str) -> np.ndarray:
    """Load the first raster band as float32."""

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"Required satellite band not found: {path_obj}"
        )

    with rasterio.open(path_obj) as src:
        if src.count < 1:
            raise ValueError(
                f"Raster contains no bands: {path_obj}"
            )

        array = src.read(1).astype(np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D raster for {path_obj}, got shape {array.shape}"
        )

    return array


def load_and_stack_bands(
    image_path: str,
    bands: List[str],
) -> np.ndarray:
    """
    Load and stack bands in the supplied canonical order.

    Returns:
        numpy array with shape [C,H,W].
    """

    paths = build_band_paths(image_path, bands)

    arrays: List[np.ndarray] = []

    reference_shape: Tuple[int, int] | None = None

    for band in bands:
        array = load_band(paths[band])

        if reference_shape is None:
            reference_shape = array.shape
        elif array.shape != reference_shape:
            raise ValueError(
                "Satellite bands have inconsistent spatial shapes: "
                f"{band}={array.shape}, expected={reference_shape}"
            )

        arrays.append(array)

    if not arrays:
        raise ValueError("No satellite bands were supplied")

    return np.stack(arrays, axis=0)


def resize_tensor(
    tensor: torch.Tensor,
    size: int = MODEL_IMAGE_SIZE,
) -> torch.Tensor:
    """
    Resize a [C,H,W] tensor to [C,size,size].

    Bilinear interpolation is used for continuous raster values.
    """

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected tensor with shape [C,H,W], got {tuple(tensor.shape)}"
        )

    tensor = tensor.unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )

    return tensor.squeeze(0)


def prepare_single_observation_tensors(
    image_path: str,
    image_size: int = MODEL_IMAGE_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare S1/S2 tensors for a single observation."""
    s1 = load_and_stack_bands(image_path, S1_BANDS)
    s2 = load_and_stack_bands(image_path, S2_BANDS)

    image_s1 = torch.from_numpy(s1).float()
    image_s2 = torch.from_numpy(s2).float()

    image_s1 = resize_tensor(image_s1, image_size)
    image_s2 = resize_tensor(image_s2, image_size)

    return image_s1, image_s2


def prepare_t1_t2_tensors(
    t1_image_path: str,
    t2_image_path: str,
    image_size: int = MODEL_IMAGE_SIZE,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Prepare the complete temporal S1/S2 inputs for RS-InternVL.
    Returns:
        Tuple[
            (t1_s1, t1_s2),
            (t2_s1, t2_s2)
        ]
    """
    t1_s1, t1_s2 = prepare_single_observation_tensors(
        t1_image_path,
        image_size,
    )

    t2_s1, t2_s2 = prepare_single_observation_tensors(
        t2_image_path,
        image_size,
    )

    return (t1_s1, t1_s2), (t2_s1, t2_s2)
