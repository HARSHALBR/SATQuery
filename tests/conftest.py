from dotenv import load_dotenv

# Load environment variables from .env file for all tests
load_dotenv()

"""
Pytest configuration and synthetic fixtures for BigEarthNet unit & integration tests.
"""

import sys
from pathlib import Path
from typing import Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest
import tifffile

from data.bigearthnet_txt.constants import (
    NATIVE_SHAPES,
    S1_BAND_NAMES,
    S2_BAND_NAMES,
)


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Creates a temporary synthetic BigEarthNet dataset on disk with:
    - 5 co-registered S1 and S2 image patches with realistic multi-resolution GeoTIFFs
    - 1 intentionally corrupt patch (missing band and corrupt TIFF)
    - BigEarthNet.txt.parquet metadata file

    Returns:
        (data_root, s1_dir, s2_dir, metadata_parquet_path)
    """
    data_root = tmp_path / "bigearthnet_txt"
    s1_dir = data_root / "images_s1"
    s2_dir = data_root / "images_s2"
    s1_dir.mkdir(parents=True, exist_ok=True)
    s2_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []

    countries = ["Austria", "Belgium", "Finland", "Ireland", "Portugal"]
    seasons = ["Summer", "Winter", "Spring", "Fall", "Summer"]
    climates = ["Cfb", "Dfb", "Dfc", "Cfb", "Csa"]

    for i in range(5):
        patch_id = f"S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_22_{i:02d}"
        s1_name = f"S1A_IW_GRDH_1SDV_20170613T165043_33N_53E_{i:06x}"

        # Create S1 patch folder & bands (VV, VH)
        patch_s1_dir = s1_dir / s1_name
        patch_s1_dir.mkdir(exist_ok=True)

        for band in S1_BAND_NAMES:
            dim = NATIVE_SHAPES[band]
            arr = np.random.uniform(
                -25.0, -5.0, size=(dim, dim)
            ).astype(np.float32)

            tifffile.imwrite(
                str(patch_s1_dir / f"{s1_name}_{band}.tif"),
                arr,
            )

        # Create S2 patch folder & 12 bands (B01 - B12)
        patch_s2_dir = s2_dir / patch_id
        patch_s2_dir.mkdir(exist_ok=True)

        for band in S2_BAND_NAMES:
            dim = NATIVE_SHAPES[band]
            arr = np.random.uniform(
                200.0, 3500.0, size=(dim, dim)
            ).astype(np.float32)

            tifffile.imwrite(
                str(patch_s2_dir / f"{patch_id}_{band}.tif"),
                arr,
            )

        # Create metadata row
        row = {
            "ID": f"ben_txt_{i+1:06d}",
            "patch_id": patch_id,
            "s1_name": s1_name,
            "input": f"Is water body present in patch {i+1}?",
            "output": (
                "Yes, water body is present."
                if i % 2 == 0
                else "No water body."
            ),
            "type": "binary" if i % 2 == 0 else "mcq",
            "category": "presence",
            "split": (
                "train"
                if i < 3
                else ("validation" if i == 3 else "test")
            ),
            "latitude": 48.123 + i * 0.1,
            "longitude": 16.456 + i * 0.1,
            "country": countries[i],
            "season": seasons[i],
            "climate_zone": climates[i],
        }

        metadata_rows.append(row)

    # Add 1 corrupted patch for testing error handling & validation
    bad_patch_id = "S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_99_99"
    bad_s1_name = "S1A_IW_GRDH_1SDV_20170613T165043_33N_53E_bad001"

    bad_s1_dir = s1_dir / bad_s1_name
    bad_s1_dir.mkdir(exist_ok=True)

    # Write only VV, missing VH
    tifffile.imwrite(
        str(bad_s1_dir / f"{bad_s1_name}_VV.tif"),
        np.zeros((120, 120), dtype=np.float32),
    )

    bad_s2_dir = s2_dir / bad_patch_id
    bad_s2_dir.mkdir(exist_ok=True)

    # Write corrupted 0-byte TIFF file
    with open(bad_s2_dir / f"{bad_patch_id}_B02.tif", "wb") as f:
        f.write(b"")

    metadata_rows.append({
        "ID": "ben_txt_000099",
        "patch_id": bad_patch_id,
        "s1_name": bad_s1_name,
        "input": "Corrupted patch test?",
        "output": "Invalid.",
        "type": "binary",
        "category": "presence",
        "split": "train",
        "latitude": 48.0,
        "longitude": 16.0,
        "country": "Austria",
        "season": "Summer",
        "climate_zone": "Cfb",
    })

    # Save to Parquet
    df = pd.DataFrame(metadata_rows)
    metadata_parquet_path = data_root / "BigEarthNet.txt.parquet"
    df.to_parquet(metadata_parquet_path, index=False)

    return data_root, s1_dir, s2_dir, metadata_parquet_path