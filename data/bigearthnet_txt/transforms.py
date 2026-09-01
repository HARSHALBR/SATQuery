"""
Multi-modal transforms and normalization routines for Sentinel-1 (SAR)
and Sentinel-2 (multispectral) remote sensing imagery.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from data.bigearthnet_txt.constants import (
    BAND_MEANS,
    BAND_STDS,
    DEFAULT_IMG_SIZE,
    S1_BAND_NAMES,
    S2_BAND_NAMES,
)


class MultiBandNormalize:
    """
    Applies per-band Z-score normalization (x - mean) / std using standard BigEarthNet statistics.
    """

    def __init__(
        self,
        bands: List[str],
        means: Optional[Dict[str, float]] = None,
        stds: Optional[Dict[str, float]] = None,
    ):
        self.bands = bands
        means_dict = means if means is not None else BAND_MEANS
        stds_dict = stds if stds is not None else BAND_STDS

        mean_vals = [means_dict.get(b, 0.0) for b in bands]
        std_vals = [stds_dict.get(b, 1.0) for b in bands]

        # Shape: [C, 1, 1] for broadcasting across [C, H, W]
        self.mean_tensor = torch.tensor(mean_vals, dtype=torch.float32).view(-1, 1, 1)
        self.std_tensor = torch.tensor(std_vals, dtype=torch.float32).view(-1, 1, 1)
        # Avoid division by zero
        self.std_tensor = torch.where(
            self.std_tensor == 0.0, torch.tensor(1.0, dtype=torch.float32), self.std_tensor
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor of shape [C, H, W] or [B, C, H, W].
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)

        mean = self.mean_tensor.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std_tensor.to(device=tensor.device, dtype=tensor.dtype)

        if tensor.ndim == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return (tensor - mean) / std

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization: x * std + mean.
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)

        mean = self.mean_tensor.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std_tensor.to(device=tensor.device, dtype=tensor.dtype)

        if tensor.ndim == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return (tensor * std) + mean


class MultiBandResize:
    """
    Resizes multi-band image tensors to a target [H, W] resolution using specified interpolation.
    """

    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]] = DEFAULT_IMG_SIZE,
        mode: str = "nearest",
    ):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = tuple(target_size)
        self.mode = mode

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize tensor of shape [C, H, W] or [B, C, H, W] to target_size.
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)

        orig_ndim = tensor.ndim
        if orig_ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif orig_ndim == 3:
            tensor = tensor.unsqueeze(0)  # [1, C, H, W]

        curr_shape = (tensor.shape[-2], tensor.shape[-1])
        if curr_shape != self.target_size:
            align_corners = (
                True if self.mode in ("bilinear", "bicubic") else None
            )
            tensor = F.interpolate(
                tensor,
                size=self.target_size,
                mode=self.mode,
                align_corners=align_corners,
            )

        if orig_ndim == 2:
            return tensor.squeeze(0).squeeze(0)
        elif orig_ndim == 3:
            return tensor.squeeze(0)
        return tensor


class RandomSpatialAugmentation:
    """
    Random horizontal and vertical flips for remote sensing data augmentation.
    Remote sensing satellite imagery is invariant to horizontal/vertical orientation.
    """

    def __init__(self, p_hflip: float = 0.5, p_vflip: float = 0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)

        if torch.rand(1).item() < self.p_hflip:
            tensor = torch.flip(tensor, dims=[-1])
        if torch.rand(1).item() < self.p_vflip:
            tensor = torch.flip(tensor, dims=[-2])
        return tensor


def build_transform(
    bands: List[str],
    img_size: int = DEFAULT_IMG_SIZE,
    upsample_mode: str = "nearest",
    is_training: bool = False,
    normalize: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory function to create a composite transform pipeline for a given set of bands.
    
    Args:
        bands: List of band identifiers.
        img_size: Target spatial dimension (H=W=img_size).
        upsample_mode: Interpolation mode ('nearest', 'bilinear', 'bicubic').
        is_training: If True, applies random horizontal/vertical flips.
        normalize: If True, applies band-wise Z-score normalization.
        
    Returns:
        A callable transform taking a [C, H, W] tensor and returning a processed [C, H, W] tensor.
    """
    resizer = MultiBandResize(target_size=img_size, mode=upsample_mode)
    normalizer = MultiBandNormalize(bands=bands) if normalize else None
    augmenter = RandomSpatialAugmentation() if is_training else None

    def _transform(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        tensor = resizer(tensor)
        if augmenter is not None:
            tensor = augmenter(tensor)
        if normalizer is not None:
            tensor = normalizer(tensor)
        return tensor

    return _transform
