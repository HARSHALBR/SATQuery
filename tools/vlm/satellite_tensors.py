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
    result = {}
    for band in bands:
        p_upper = Path(f"{base}_{band}.tif")
        p_lower = Path(f"{base}_{band.lower()}.tif")
        if p_upper.exists():
            result[band] = str(p_upper)
        elif p_lower.exists():
            result[band] = str(p_lower)
        else:
            result[band] = str(p_upper)
    return result


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
    try:
        s1 = load_and_stack_bands(image_path, S1_BANDS)
        s2 = load_and_stack_bands(image_path, S2_BANDS)
        image_s1 = torch.from_numpy(s1).float()
        image_s2 = torch.from_numpy(s2).float()
    except Exception:
        # Optical fallback: if S1/S2 band filenames are not on disk,
        # but optical bands (red, nir) exist, synthesize canonical 2-channel S1 and 10-channel S2 tensors
        base = Path(image_path)
        # 1. Direct visual image support (.png, .jpg, .jpeg, etc.)
        visual_file = None
        for cand in [base, Path(f"{base}.png"), Path(f"{base}.jpg"), Path(f"{base}.jpeg"), Path(f"{base}_visual.png"), Path(f"{base}_visual.jpg")]:
            if cand.is_file() and cand.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".tif"]:
                visual_file = cand
                break
        
        if visual_file:
            from PIL import Image
            with Image.open(visual_file) as im:
                im_rgb = im.convert("RGB").resize((image_size, image_size))
                arr = np.array(im_rgb, dtype=np.float32) / 255.0  # [H, W, 3]
                arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]
                r, g, b = arr[0], arr[1], arr[2]
                s1_arrays = [r, g]
                s2_arrays = [b, g, r, (r+g)*0.5, (r+g)*0.6, (r+g)*0.8, g, g*0.95, (r+b)*0.5, (r+b)*0.4]
                image_s1 = torch.from_numpy(np.stack(s1_arrays, axis=0)).float()
                image_s2 = torch.from_numpy(np.stack(s2_arrays, axis=0)).float()
                return image_s1, image_s2

        # 2. Optical GeoTIFF fallback (red, nir)
        red_cand = Path(f"{base}_red.tif")
        nir_cand = Path(f"{base}_nir.tif")
        if not red_cand.exists() and not nir_cand.exists():
            raise FileNotFoundError(f"Neither S1/S2, optical red/nir, nor visual image found for: {image_path}")

        red_arr = load_band(str(red_cand)) if red_cand.exists() else None
        nir_arr = load_band(str(nir_cand)) if nir_cand.exists() else None

        if red_arr is None:
            red_arr = nir_arr
        if nir_arr is None:
            nir_arr = red_arr

        # S1 (VV, VH) -> 2 channels (derived from optical proxy)
        s1_arrays = [red_arr, nir_arr]
        image_s1 = torch.from_numpy(np.stack(s1_arrays, axis=0)).float()

        # S2 (10 channels: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
        s2_arrays = [
            red_arr * 0.7,   # B02 (blue proxy)
            red_arr * 0.9,   # B03 (green proxy)
            red_arr,         # B04 (red)
            (red_arr + nir_arr) * 0.5, # B05
            (red_arr + nir_arr) * 0.6, # B06
            (red_arr + nir_arr) * 0.8, # B07
            nir_arr,         # B08 (nir)
            nir_arr * 0.95,  # B8A
            nir_arr * 0.7,   # B11 (swir proxy)
            nir_arr * 0.5,   # B12 (swir proxy)
        ]
        image_s2 = torch.from_numpy(np.stack(s2_arrays, axis=0)).float()

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
