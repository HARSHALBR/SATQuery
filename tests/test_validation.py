"""
Unit tests for data validation, missing file detection, corruption handling, and audit reporting.
"""

import json
from pathlib import Path
import pytest

from data.bigearthnet_txt.utils import (
    read_geotiff,
    validate_coordinates,
    validate_patch_bands,
)
from scripts.validate_bigearthnet import run_validation


def test_corrupted_file_detection(tmp_path):
    # 0-byte file
    empty_file = tmp_path / "empty.tif"
    empty_file.touch()

    with pytest.raises(ValueError, match="empty"):
        read_geotiff(empty_file)

    # Missing file
    with pytest.raises(FileNotFoundError):
        read_geotiff(tmp_path / "nonexistent.tif")


def test_coordinate_validation():
    valid, err = validate_coordinates(45.0, 10.0)
    assert valid is True
    assert err is None

    valid, err = validate_coordinates(95.0, 10.0)
    assert valid is False
    assert "Latitude" in err

    valid, err = validate_coordinates(45.0, -195.0)
    assert valid is False
    assert "Longitude" in err

    valid, err = validate_coordinates("invalid", 10.0)
    assert valid is False
    assert "Non-numeric" in err


def test_missing_bands_detection(tmp_path):
    patch_dir = tmp_path / "test_patch"
    patch_dir.mkdir()

    # Create only VV, missing VH
    with open(patch_dir / "test_patch_VV.tif", "wb") as f:
        f.write(b"dummy")

    res = validate_patch_bands(patch_dir, ["VV", "VH"], prefix="test_patch")
    assert res["valid"] is False
    assert "VH" in res["missing_bands"]


def test_full_validation_run(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    report_out = tmp_path / "reports" / "validation_report.json"
    invalid_out = tmp_path / "reports" / "invalid_samples.jsonl"

    report = run_validation(
        data_root=str(data_root),
        report_output=str(report_out),
        invalid_log=str(invalid_out),
    )

    assert report["metrics"]["total_samples"] == 6
    assert report["metrics"]["valid_samples"] == 5
    assert report["metrics"]["invalid_samples"] == 1
    assert report["metrics"]["valid_percentage"] == pytest.approx(83.33, rel=1e-2)

    # Check report JSON file exists
    assert report_out.exists()
    assert invalid_out.exists()

    # Read invalid log
    with open(invalid_out, "r", encoding="utf-8") as f:
        invalid_entries = [json.loads(line) for line in f if line.strip()]

    assert len(invalid_entries) == 1
    assert invalid_entries[0]["image_id"] == "S2A_MSIL2A_20170613T101031_N0205_R022_T32ULD_99_99"
