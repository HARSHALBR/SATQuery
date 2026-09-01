"""
Production PyTorch Dataset and DataLoader collation for BigEarthNet.txt.

Provides high-throughput multi-modal data loading for Sentinel-1 (SAR) and Sentinel-2 (Multispectral)
satellite imagery paired with instruction-driven natural language queries and answers.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.bigearthnet_txt.constants import (
    DEFAULT_IMG_SIZE,
    MODEL_S2_BANDS,
    PREDEFINED_BAND_COMBINATIONS,
    S1_BAND_NAMES,
    S2_BAND_NAMES,
)
from data.bigearthnet_txt.transforms import (
    MultiBandNormalize,
    MultiBandResize,
    build_transform,
)
from data.bigearthnet_txt.utils import read_geotiff

logger = logging.getLogger(__name__)


class BigEarthNetDataset(Dataset):
    """
    Production multi-modal PyTorch Dataset for BigEarthNet.txt.
    
    Exposes unified Sentinel-1 (SAR), Sentinel-2 (Multispectral), text prompt,
    target answer, metadata, and task attributes.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        samples: Optional[List[Dict[str, Any]]] = None,
        s1_bands: Optional[Union[List[str], str]] = None,
        s2_bands: Optional[Union[List[str], str]] = None,
        img_size: int = DEFAULT_IMG_SIZE,
        upsample_mode: str = "nearest",
        split: Optional[Union[str, List[str]]] = None,
        task_types: Optional[Iterable[str]] = None,
        task_categories: Optional[Iterable[str]] = None,
        countries: Optional[Iterable[str]] = None,
        seasons: Optional[Iterable[str]] = None,
        climate_zones: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
        sample_ratio: Optional[float] = None,
        seed: int = 42,
        is_training: bool = False,
        normalize: bool = True,
        transform_s1: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        transform_s2: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        strict: bool = True,
    ):
        """
        Initialize the BigEarthNet multi-modal dataset.
        
        Args:
            manifest_path: Path to manifest JSONL or Parquet file.
            data_root: Root directory where image files reside.
            samples: Pre-loaded list of sample dicts (optional alternative to manifest_path).
            s1_bands: S1 band list or preset (e.g. ['VV', 'VH']).
            s2_bands: S2 band list or preset (e.g. 'all', 'S2-10m20m', 'RGB').
            img_size: Standardized spatial size (H=W=img_size).
            upsample_mode: Interpolation method for multi-resolution bands ('nearest', 'bilinear', 'bicubic').
            split: Filter by split ('train', 'validation', 'test', 'bench', or list).
            task_types: Filter by task type ('binary', 'mcq', 'captioning', 'bounding box').
            task_categories: Filter by category ('presence', 'area', 'country', etc.).
            countries: Filter by country list.
            seasons: Filter by acquisition season list.
            climate_zones: Filter by Köppen-Geiger climate zones.
            max_samples: Limit total samples loaded (useful for rapid dev/debugging).
            sample_ratio: Fraction of dataset to retain (0.0 < sample_ratio <= 1.0).
            seed: Deterministic random seed for subset sampling.
            is_training: If True, applies training augmentations (spatial flips).
            normalize: If True, applies channel-wise Z-score normalization.
            transform_s1: Custom transform for Sentinel-1 tensor.
            transform_s2: Custom transform for Sentinel-2 tensor.
            strict: If True, raises exceptions on missing/corrupted files. If False, returns zero-filled tensors.
        """
        super().__init__()
        self.data_root = Path(data_root) if data_root else Path("data/bigearthnet_txt")
        self.img_size = img_size
        self.upsample_mode = upsample_mode
        self.strict = strict
        self.is_training = is_training

        # Resolve S1 bands
        if s1_bands is None:
            self.s1_bands = list(S1_BAND_NAMES)
        elif isinstance(s1_bands, str) and s1_bands in PREDEFINED_BAND_COMBINATIONS:
            self.s1_bands = [b for b in PREDEFINED_BAND_COMBINATIONS[s1_bands] if b in S1_BAND_NAMES]
        elif isinstance(s1_bands, list):
            self.s1_bands = list(s1_bands)
        else:
            raise ValueError(f"Invalid s1_bands: {s1_bands}")

        # Resolve S2 bands
        # Default to MODEL_S2_BANDS (10 bands: B02–B12 excluding B01/B09).
        # Use s2_bands='S2-all' or s2_bands=S2_BAND_NAMES explicitly to load all 12 raw bands.
        if s2_bands is None or s2_bands == "model" or s2_bands == "S2-10m20m":
            self.s2_bands = list(MODEL_S2_BANDS)
        elif s2_bands == "all" or s2_bands == "S2-all":
            self.s2_bands = list(S2_BAND_NAMES)
        elif isinstance(s2_bands, str) and s2_bands in PREDEFINED_BAND_COMBINATIONS:
            self.s2_bands = [b for b in PREDEFINED_BAND_COMBINATIONS[s2_bands] if b in S2_BAND_NAMES]
        elif isinstance(s2_bands, list):
            self.s2_bands = list(s2_bands)
        else:
            raise ValueError(f"Invalid s2_bands: {s2_bands}")

        # Transforms
        self.transform_s1 = (
            transform_s1
            if transform_s1 is not None
            else build_transform(
                bands=self.s1_bands,
                img_size=img_size,
                upsample_mode=upsample_mode,
                is_training=is_training,
                normalize=normalize,
            )
        )
        self.transform_s2 = (
            transform_s2
            if transform_s2 is not None
            else build_transform(
                bands=self.s2_bands,
                img_size=img_size,
                upsample_mode=upsample_mode,
                is_training=is_training,
                normalize=normalize,
            )
        )

        # Load samples
        if samples is not None:
            self.records = list(samples)
        elif manifest_path is not None:
            self.records = self._load_manifest(manifest_path)
        else:
            raise ValueError("Either `manifest_path` or `samples` must be provided.")

        # Apply filtering
        self.records = self._filter_records(
            records=self.records,
            split=split,
            task_types=task_types,
            task_categories=task_categories,
            countries=countries,
            seasons=seasons,
            climate_zones=climate_zones,
        )

        # Apply subsetting / sampling
        if sample_ratio is not None and 0.0 < sample_ratio < 1.0:
            rng = np.random.default_rng(seed)
            num_keep = max(1, int(len(self.records) * sample_ratio))
            indices = rng.choice(len(self.records), size=num_keep, replace=False)
            indices.sort()
            self.records = [self.records[i] for i in indices]

        if max_samples is not None and len(self.records) > max_samples:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(self.records), size=max_samples, replace=False)
            indices.sort()
            self.records = [self.records[i] for i in indices]

    def _load_manifest(self, manifest_path: Union[str, Path]) -> List[Dict[str, Any]]:
        p = Path(manifest_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {p}")

        suffix = p.suffix.lower()
        records: List[Dict[str, Any]] = []

        if suffix in (".jsonl", ".txt"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and "samples" in data:
                    records = data["samples"]
                else:
                    raise ValueError(f"Unexpected JSON format in manifest: {p}")
        elif suffix == ".parquet":
            df = pd.read_parquet(p)
            records = df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported manifest format: {suffix}")

        return records

    def _filter_records(
        self,
        records: List[Dict[str, Any]],
        split: Optional[Union[str, List[str]]],
        task_types: Optional[Iterable[str]],
        task_categories: Optional[Iterable[str]],
        countries: Optional[Iterable[str]],
        seasons: Optional[Iterable[str]],
        climate_zones: Optional[Iterable[str]],
    ) -> List[Dict[str, Any]]:
        filtered = records

        if split is not None:
            splits_set = {split} if isinstance(split, str) else set(split)
            filtered = [r for r in filtered if r.get("split") in splits_set]

        if task_types is not None:
            types_set = set(task_types)
            filtered = [r for r in filtered if r.get("task_type") in types_set or r.get("type") in types_set]

        if task_categories is not None:
            cats_set = set(task_categories)
            filtered = [r for r in filtered if r.get("task_category") in cats_set or r.get("category") in cats_set]

        if countries is not None:
            cnt_set = set(countries)
            filtered = [
                r for r in filtered
                if (r.get("metadata", {}).get("country") in cnt_set) or (r.get("country") in cnt_set)
            ]

        if seasons is not None:
            sea_set = set(seasons)
            filtered = [
                r for r in filtered
                if (r.get("metadata", {}).get("season") in sea_set) or (r.get("season") in sea_set)
            ]

        if climate_zones is not None:
            clt_set = set(climate_zones)
            filtered = [
                r for r in filtered
                if (r.get("metadata", {}).get("climate_zone") in clt_set) or (r.get("climate_zone") in clt_set)
            ]

        return filtered

    def __len__(self) -> int:
        return len(self.records)

    def _load_patch_tensor(
        self,
        patch_dir_rel: Optional[str],
        patch_name: str,
        bands: List[str],
    ) -> torch.Tensor:
        """
        Load and stack individual band GeoTIFFs into a [C, H, W] float32 tensor.
        """
        if not patch_dir_rel and not patch_name:
            if self.strict:
                raise ValueError("Both patch_dir_rel and patch_name are empty.")
            return torch.zeros((len(bands), self.img_size, self.img_size), dtype=torch.float32)

        # Resolve patch directory
        patch_dir: Optional[Path] = None
        if patch_dir_rel:
            p = self.data_root / patch_dir_rel
            if p.exists():
                patch_dir = p
            elif Path(patch_dir_rel).exists():
                patch_dir = Path(patch_dir_rel)

        if patch_dir is None or not patch_dir.exists():
            # Search common directory structures
            candidates = [
                self.data_root / "images_s1" / patch_name,
                self.data_root / "images_s2" / patch_name,
                self.data_root / "s1" / patch_name,
                self.data_root / "s2" / patch_name,
                self.data_root / patch_name,
            ]
            for cand in candidates:
                if cand.exists():
                    patch_dir = cand
                    break

        if patch_dir is None or not patch_dir.exists():
            if self.strict:
                raise FileNotFoundError(f"Patch folder not found on disk: {patch_name} (checked {patch_dir_rel})")
            return torch.zeros((len(bands), self.img_size, self.img_size), dtype=torch.float32)

        resizer = MultiBandResize(target_size=self.img_size, mode=self.upsample_mode)
        band_tensors: List[torch.Tensor] = []

        for band in bands:
            # Look for <patch_name>_<band>.tif or <band>.tif
            band_candidates = [
                patch_dir / f"{patch_name}_{band}.tif",
                patch_dir / f"{patch_name}_{band}.tiff",
                patch_dir / f"{band}.tif",
                patch_dir / f"{band}.tiff",
            ]
            band_file: Optional[Path] = None
            for cand in band_candidates:
                if cand.exists():
                    band_file = cand
                    break

            if band_file is None:
                # Fallback scan
                for f in patch_dir.iterdir():
                    if f.is_file() and (f.stem.endswith(f"_{band}") or f.stem == band):
                        band_file = f
                        break

            if band_file is None or not band_file.exists():
                if self.strict:
                    raise FileNotFoundError(f"Band {band} not found in patch {patch_dir}")
                band_tensors.append(torch.zeros((self.img_size, self.img_size), dtype=torch.float32))
                continue

            try:
                arr, _ = read_geotiff(band_file)
                t = torch.as_tensor(arr, dtype=torch.float32)
                t = resizer(t)
                band_tensors.append(t)
            except Exception as e:
                if self.strict:
                    raise RuntimeError(f"Error reading GeoTIFF {band_file}: {e}") from e
                band_tensors.append(torch.zeros((self.img_size, self.img_size), dtype=torch.float32))

        # Stack into [C, H, W]
        stacked = torch.stack(band_tensors, dim=0)
        return stacked

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve sample at index idx.
        
        Returns:
            {
                "image_s1": torch.Tensor,       # [2, H, W]
                "image_s2": torch.Tensor,       # [12, H, W] or [C, H, W]
                "text": str,                    # input query/instruction
                "target_text": str,             # target output/answer
                "metadata": dict,               # spatial/temporal metadata dict
                "image_id": str,                # patch_id
                "task": str,                    # type and category string
                "split": str                    # dataset split
            }
        """
        record = self.records[idx]

        # Extract fields
        patch_id = record.get("image_id") or record.get("patch_id", "")
        s1_name = record.get("s1_name", "")
        s1_path = record.get("s1_path")
        s2_path = record.get("s2_path")
        text_in = record.get("text_input") or record.get("input", "")
        text_out = record.get("text_output") or record.get("output", "")
        task_type = record.get("task_type") or record.get("type", "unspecified")
        task_cat = record.get("task_category") or record.get("category", "unspecified")
        split = record.get("split", "train")

        meta = record.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
        # Ensure base attributes present in metadata dict
        meta.setdefault("patch_id", patch_id)
        meta.setdefault("s1_name", s1_name)
        meta.setdefault("split", split)

        # Load S1 tensor [2, H, W]
        s1_tensor = self._load_patch_tensor(s1_path, s1_name, self.s1_bands)
        if self.transform_s1 is not None:
            s1_tensor = self.transform_s1(s1_tensor)

        # Load S2 tensor [12, H, W]
        s2_tensor = self._load_patch_tensor(s2_path, patch_id, self.s2_bands)
        if self.transform_s2 is not None:
            s2_tensor = self.transform_s2(s2_tensor)

        task_str = f"{task_type}:{task_cat}" if task_type and task_cat else task_type

        return {
            "image_s1": s1_tensor,
            "image_s2": s2_tensor,
            "text": text_in,
            "target_text": text_out,
            "metadata": meta,
            "image_id": patch_id,
            "task": task_str,
            "split": split,
            "s2_bands": self.s2_bands,  # expose band list for downstream validation
        }


def collate_bigearthnet(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for PyTorch DataLoader.
    
    Stacks image_s1 and image_s2 into batch tensors [B, C, H, W] and gathers
    text, target_text, metadata, image_id, and task attributes.
    """
    s1_list = []
    s2_list = []
    text_list = []
    target_list = []
    meta_list = []
    id_list = []
    task_list = []
    split_list = []

    for item in batch:
        s1_list.append(item["image_s1"])
        s2_list.append(item["image_s2"])
        text_list.append(item["text"])
        target_list.append(item["target_text"])
        meta_list.append(item["metadata"])
        id_list.append(item["image_id"])
        task_list.append(item["task"])
        split_list.append(item["split"])

    return {
        "image_s1": torch.stack(s1_list, dim=0),
        "image_s2": torch.stack(s2_list, dim=0),
        "text": text_list,
        "target_text": target_list,
        "metadata": meta_list,
        "image_id": id_list,
        "task": task_list,
        "split": split_list,
    }
