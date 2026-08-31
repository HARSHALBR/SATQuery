"""
Unit tests for manifest generation, parsing, schema consistency, and loading.
"""

import json
from pathlib import Path
import pytest

from scripts.build_bigearthnet_manifest import build_manifest
from data.bigearthnet_txt.dataset import BigEarthNetDataset


def test_build_manifest_generation(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"

    summary = build_manifest(
        data_root=str(data_root),
        output_dir=str(manifest_out),
        seed=42,
        validate_files=True,
    )

    assert summary["total_samples"] == 6
    assert (manifest_out / "manifest_full.jsonl").exists()
    assert (manifest_out / "manifest_train.jsonl").exists()
    assert (manifest_out / "manifest_validation.jsonl").exists()
    assert (manifest_out / "manifest_test.jsonl").exists()
    assert (manifest_out / "manifest_summary.json").exists()

    # Read manifest_full.jsonl
    with open(manifest_out / "manifest_full.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    assert len(lines) == 6
    required_keys = {
        "sample_id",
        "image_id",
        "s1_name",
        "s1_path",
        "s2_path",
        "text_input",
        "text_output",
        "task_type",
        "task_category",
        "split",
        "metadata",
    }
    for item in lines:
        assert required_keys.issubset(set(item.keys()))
        assert isinstance(item["metadata"], dict)
        assert "country" in item["metadata"]
        assert "season" in item["metadata"]
        assert "climate_zone" in item["metadata"]


def test_load_manifest_in_dataset(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"

    build_manifest(
        data_root=str(data_root),
        output_dir=str(manifest_out),
        seed=42,
    )

    # Initialize Dataset pointing to manifest_train.jsonl
    ds = BigEarthNetDataset(
        manifest_path=manifest_out / "manifest_train.jsonl",
        data_root=data_root,
        strict=False,
    )
    assert len(ds) >= 1
    sample = ds[0]
    assert "image_s1" in sample
    assert "image_s2" in sample
    assert "text" in sample
    assert "target_text" in sample
    assert "metadata" in sample
