"""
Robust dataset parser for BigEarthNet.txt and BigEarthNet-MM / BigEarthNet v2.0.

Handles Parquet, JSONL, CSV, and raw directory hierarchies containing GeoTIFF files
and patch-level label metadata files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import pandas as pd

from data.bigearthnet_txt.constants import (
    EXPECTED_METADATA_COLUMNS,
    S1_BAND_NAMES,
    S2_BAND_NAMES,
    VALID_CATEGORIES,
    VALID_SPLITS,
    VALID_TASK_TYPES,
)
from data.bigearthnet_txt.utils import (
    validate_coordinates,
    validate_patch_bands,
    validate_s1_s2_pairing,
)

logger = logging.getLogger(__name__)


class BigEarthNetParser:
    """
    Production parser for BigEarthNet.txt multimodal dataset.
    
    Reads instruction-text metadata, scans Sentinel-1 SAR and Sentinel-2 multispectral
    patch folders, validates pairings, and produces structured sample records.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        metadata_file: Optional[Union[str, Path]] = None,
        s1_dir: Optional[Union[str, Path]] = None,
        s2_dir: Optional[Union[str, Path]] = None,
    ):
        self.data_root = Path(data_root)
        self.metadata_file = Path(metadata_file) if metadata_file else self._discover_metadata_file()
        self.s1_dir = Path(s1_dir) if s1_dir else self._discover_sensor_dir("s1")
        self.s2_dir = Path(s2_dir) if s2_dir else self._discover_sensor_dir("s2")

    def _discover_metadata_file(self) -> Optional[Path]:
        """Auto-discover metadata file in data_root if not explicitly provided."""
        candidates = [
            self.data_root / "BigEarthNet.txt.parquet",
            self.data_root / "BigEarthNet.txt.csv",
            self.data_root / "BigEarthNet.txt.jsonl",
            self.data_root / "metadata.parquet",
            self.data_root / "metadata.jsonl",
            self.data_root / "metadata.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _discover_sensor_dir(self, sensor_type: str) -> Optional[Path]:
        """Auto-discover Sentinel-1 or Sentinel-2 directory hierarchy."""
        sensor_type = sensor_type.lower()
        if sensor_type == "s1":
            names = ["images_s1", "s1", "BigEarthNet-S1", "BigEarthNet-v1.0-S1", "BigEarthNet-v2.0-S1", "S1"]
        else:
            names = ["images_s2", "s2", "BigEarthNet-S2", "BigEarthNet-v1.0-S2", "BigEarthNet-v2.0-S2", "S2"]

        for name in names:
            p = self.data_root / name
            if p.exists() and p.is_dir():
                return p
        return None

    def load_metadata_dataframe(self) -> pd.DataFrame:
        """
        Load textual instructions and annotations from Parquet, CSV, JSONL, or JSON.
        """
        if self.metadata_file is None or not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found in {self.data_root}. "
                "Expected BigEarthNet.txt.parquet, .csv, or .jsonl."
            )

        suffix = self.metadata_file.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(self.metadata_file)
        elif suffix == ".csv":
            df = pd.read_csv(self.metadata_file)
        elif suffix in (".jsonl", ".json"):
            df = pd.read_json(self.metadata_file, lines=(suffix == ".jsonl"))
        else:
            raise ValueError(f"Unsupported metadata file extension: {suffix}")

        # Normalize column names to lowercase
        col_map = {c: c.strip() for c in df.columns}
        df = df.rename(columns=col_map)

        # Standardize ID column
        if "id" in df.columns and "ID" not in df.columns:
            df["ID"] = df["id"]
        elif "ID" not in df.columns:
            df["ID"] = [f"ben_txt_{i:07d}" for i in range(len(df))]

        return df

    def scan_patch_directories(self) -> Tuple[Dict[str, Path], Dict[str, Path]]:
        """
        Scan disk for available S1 and S2 patch directories.
        
        Returns:
            Tuple of (s1_patches_dict, s2_patches_dict) mapping patch_name -> Path.
        """
        s1_patches: Dict[str, Path] = {}
        s2_patches: Dict[str, Path] = {}

        if self.s1_dir and self.s1_dir.exists():
            for item in self.s1_dir.iterdir():
                if item.is_dir():
                    s1_patches[item.name] = item

        if self.s2_dir and self.s2_dir.exists():
            for item in self.s2_dir.iterdir():
                if item.is_dir():
                    s2_patches[item.name] = item

        return s1_patches, s2_patches

    def parse_records(
        self,
        validate_files: bool = True,
        max_samples: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream parsed and validated records.
        
        Args:
            validate_files: If True, checks for existence and validity of S1 and S2 band files on disk.
            max_samples: Optional maximum number of samples to process.
            
        Yields:
            Dict containing unified sample fields.
        """
        df = self.load_metadata_dataframe()
        s1_patches, s2_patches = self.scan_patch_directories()

        count = 0
        for _, row in df.iterrows():
            if max_samples is not None and count >= max_samples:
                break

            record = self._row_to_record(row, s1_patches, s2_patches, validate_files=validate_files)
            yield record
            count += 1

    def _row_to_record(
        self,
        row: pd.Series,
        s1_patches: Dict[str, Path],
        s2_patches: Dict[str, Path],
        validate_files: bool = True,
    ) -> Dict[str, Any]:
        """
        Transform a single metadata row into a standardized sample record.
        """
        patch_id = str(row.get("patch_id", "")).strip()
        s1_name = str(row.get("s1_name", "")).strip()
        sample_id = str(row.get("ID", "")).strip()
        text_in = str(row.get("input", "")).strip()
        text_out = str(row.get("output", "")).strip()
        task_type = str(row.get("type", "unspecified")).strip()
        task_category = str(row.get("category", "unspecified")).strip()
        split = str(row.get("split", "train")).strip()

        # Metadata extraction
        metadata = {
            "latitude": float(row["latitude"]) if "latitude" in row and pd.notna(row["latitude"]) else None,
            "longitude": float(row["longitude"]) if "longitude" in row and pd.notna(row["longitude"]) else None,
            "country": str(row["country"]).strip() if "country" in row and pd.notna(row["country"]) else None,
            "season": str(row["season"]).strip() if "season" in row and pd.notna(row["season"]) else None,
            "climate_zone": str(row["climate_zone"]).strip() if "climate_zone" in row and pd.notna(row["climate_zone"]) else None,
            "patch_id": patch_id,
            "s1_name": s1_name,
        }

        # Resolve paths
        s1_dir_path = s1_patches.get(s1_name)
        if s1_dir_path is None and self.s1_dir:
            cand = self.s1_dir / s1_name
            if cand.exists():
                s1_dir_path = cand

        s2_dir_path = s2_patches.get(patch_id)
        if s2_dir_path is None and self.s2_dir:
            cand = self.s2_dir / patch_id
            if cand.exists():
                s2_dir_path = cand

        # Relative paths for portable manifests
        s1_rel = str(s1_dir_path.relative_to(self.data_root)) if s1_dir_path and s1_dir_path.is_relative_to(self.data_root) else (str(s1_dir_path) if s1_dir_path else None)
        s2_rel = str(s2_dir_path.relative_to(self.data_root)) if s2_dir_path and s2_dir_path.is_relative_to(self.data_root) else (str(s2_dir_path) if s2_dir_path else None)

        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_id": patch_id,
            "s1_name": s1_name,
            "s1_path": s1_rel,
            "s2_path": s2_rel,
            "text_input": text_in,
            "text_output": text_out,
            "task_type": task_type,
            "task_category": task_category,
            "split": split,
            "metadata": metadata,
            "is_valid": True,
            "validation_errors": [],
        }

        # Validate fields
        if not patch_id:
            record["is_valid"] = False
            record["validation_errors"].append("MISSING_PATCH_ID")
        if not s1_name:
            record["is_valid"] = False
            record["validation_errors"].append("MISSING_S1_NAME")
        if not text_in:
            record["is_valid"] = False
            record["validation_errors"].append("EMPTY_INPUT_TEXT")

        if metadata["latitude"] is not None and metadata["longitude"] is not None:
            coord_valid, coord_err = validate_coordinates(metadata["latitude"], metadata["longitude"])
            if not coord_valid:
                record["is_valid"] = False
                record["validation_errors"].append(f"INVALID_COORDINATES: {coord_err}")

        # Pairing verification
        pair_valid, pair_err = validate_s1_s2_pairing(s1_name, patch_id)
        if not pair_valid:
            record["is_valid"] = False
            record["validation_errors"].append(f"INVALID_PAIRING: {pair_err}")

        # File validation if requested
        if validate_files:
            if s1_dir_path is None or not s1_dir_path.exists():
                record["is_valid"] = False
                record["validation_errors"].append("MISSING_S1_DIR")
            else:
                s1_val = validate_patch_bands(s1_dir_path, S1_BAND_NAMES, prefix=s1_name)
                if not s1_val["valid"]:
                    record["is_valid"] = False
                    if s1_val["missing_bands"]:
                        record["validation_errors"].append(f"MISSING_S1_BANDS: {s1_val['missing_bands']}")
                    if s1_val["corrupted_bands"]:
                        record["validation_errors"].append(f"CORRUPTED_S1_BANDS: {s1_val['corrupted_bands']}")

            if s2_dir_path is None or not s2_dir_path.exists():
                record["is_valid"] = False
                record["validation_errors"].append("MISSING_S2_DIR")
            else:
                s2_val = validate_patch_bands(s2_dir_path, S2_BAND_NAMES, prefix=patch_id)
                if not s2_val["valid"]:
                    record["is_valid"] = False
                    if s2_val["missing_bands"]:
                        record["validation_errors"].append(f"MISSING_S2_BANDS: {s2_val['missing_bands']}")
                    if s2_val["corrupted_bands"]:
                        record["validation_errors"].append(f"CORRUPTED_S2_BANDS: {s2_val['corrupted_bands']}")

        return record
