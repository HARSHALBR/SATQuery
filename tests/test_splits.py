"""
Unit tests for deterministic dataset splitting, seed stability, and isolation (zero data leakage).
"""

import json
from pathlib import Path
import pytest

from scripts.build_bigearthnet_manifest import deterministic_patch_split, build_manifest
from data.bigearthnet_txt.dataset import BigEarthNetDataset


def test_deterministic_split_stability():
    patch_ids = [f"patch_{i:04d}" for i in range(100)]

    split1 = deterministic_patch_split(patch_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    split2 = deterministic_patch_split(patch_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    split3 = deterministic_patch_split(patch_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=99)

    # Identical seed produces identical split
    assert split1 == split2

    # Different seed produces different split
    assert split1 != split3


def test_zero_leakage_and_isolation():
    patch_ids = [f"patch_{i:04d}" for i in range(500)]
    split_map = deterministic_patch_split(patch_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    train_patches = {p for p, s in split_map.items() if s == "train"}
    val_patches = {p for p, s in split_map.items() if s == "validation"}
    test_patches = {p for p, s in split_map.items() if s == "test"}

    # Total coverage
    assert len(train_patches) + len(val_patches) + len(test_patches) == 500

    # Zero overlap
    assert len(train_patches.intersection(val_patches)) == 0
    assert len(train_patches.intersection(test_patches)) == 0
    assert len(val_patches.intersection(test_patches)) == 0


def test_dataset_split_filtering(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"

    build_manifest(
        data_root=str(data_root),
        output_dir=str(manifest_out),
        seed=42,
    )

    full_manifest = manifest_out / "manifest_full.jsonl"

    train_ds = BigEarthNetDataset(manifest_path=full_manifest, data_root=data_root, split="train", strict=False)
    val_ds = BigEarthNetDataset(manifest_path=full_manifest, data_root=data_root, split="validation", strict=False)
    test_ds = BigEarthNetDataset(manifest_path=full_manifest, data_root=data_root, split="test", strict=False)

    for item in train_ds:
        assert item["split"] == "train"
    for item in val_ds:
        assert item["split"] == "validation"
    for item in test_ds:
        assert item["split"] == "test"
