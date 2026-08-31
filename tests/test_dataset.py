"""
Unit tests for PyTorch Dataset interface, multi-modal band loading, transforms, and DataLoader collation.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from data.bigearthnet_txt.constants import MODEL_S2_BANDS, S2_BAND_NAMES
from data.bigearthnet_txt.dataset import BigEarthNetDataset, collate_bigearthnet
from scripts.build_bigearthnet_manifest import build_manifest


def test_dataset_getitem_structure(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"

    build_manifest(
        data_root=str(data_root),
        output_dir=str(manifest_out),
        seed=42,
    )

    manifest_file = manifest_out / "manifest_full.jsonl"
    ds = BigEarthNetDataset(
        manifest_path=manifest_file,
        data_root=data_root,
        strict=False,  # Skip bad sample smoothly in test
    )

    assert len(ds) == 6
    sample = ds[0]

    # Required output structure
    assert isinstance(sample, dict)
    assert "image_s1" in sample
    assert "image_s2" in sample
    assert "text" in sample
    assert "target_text" in sample
    assert "metadata" in sample
    assert "image_id" in sample
    assert "task" in sample
    assert "split" in sample

    # Check tensors
    s1_tensor = sample["image_s1"]
    s2_tensor = sample["image_s2"]

    assert isinstance(s1_tensor, torch.Tensor)
    assert isinstance(s2_tensor, torch.Tensor)
    # Default is MODEL_S2_BANDS: 10 channels (B02–B12 excluding B01/B09)
    assert s1_tensor.shape == (2, 120, 120)
    assert s2_tensor.shape == (10, 120, 120), (
        f"Default dataset should produce 10-band S2 tensor, got {s2_tensor.shape}. "
        "MODEL_S2_BANDS excludes B01 (Coastal aerosol) and B09 (Water vapour)."
    )
    assert s1_tensor.dtype == torch.float32
    assert s2_tensor.dtype == torch.float32

    # Check no NaNs or Infs in normalized output
    assert not torch.isnan(s1_tensor).any()
    assert not torch.isnan(s2_tensor).any()


def test_dataset_custom_bands_and_resolution(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"

    build_manifest(
        data_root=str(data_root),
        output_dir=str(manifest_out),
        seed=42,
    )

    manifest_file = manifest_out / "manifest_full.jsonl"

    # RGB preset + 224x224 target resolution (for VLMs)
    ds = BigEarthNetDataset(
        manifest_path=manifest_file,
        data_root=data_root,
        s2_bands="RGB",
        img_size=224,
        strict=False,
    )

    sample = ds[0]
    assert sample["image_s1"].shape == (2, 224, 224)
    assert sample["image_s2"].shape == (3, 224, 224)  # B04, B03, B02


def test_dataloader_collation(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"

    build_manifest(
        data_root=str(data_root),
        output_dir=str(manifest_out),
        seed=42,
    )

    manifest_file = manifest_out / "manifest_full.jsonl"
    ds = BigEarthNetDataset(
        manifest_path=manifest_file,
        data_root=data_root,
        max_samples=4,
        strict=False,
    )

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_bigearthnet,
    )

    batch = next(iter(loader))
    assert batch["image_s1"].shape == (2, 2, 120, 120)
    # Default is MODEL_S2_BANDS — 10 channels
    assert batch["image_s2"].shape == (2, 10, 120, 120)
    assert len(batch["text"]) == 2
    assert len(batch["target_text"]) == 2
    assert len(batch["metadata"]) == 2
    assert len(batch["image_id"]) == 2
    assert len(batch["task"]) == 2


def test_dataset_filtering_and_sampling(synthetic_dataset_dir, tmp_path):
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"

    build_manifest(
        data_root=str(data_root),
        output_dir=str(manifest_out),
        seed=42,
    )

    manifest_file = manifest_out / "manifest_full.jsonl"

    # Filter by country
    ds_austria = BigEarthNetDataset(
        manifest_path=manifest_file,
        data_root=data_root,
        countries=["Austria"],
        strict=False,
    )
    for sample in ds_austria:
        assert sample["metadata"]["country"] == "Austria"

    # Filter by task type
    ds_binary = BigEarthNetDataset(
        manifest_path=manifest_file,
        data_root=data_root,
        task_types=["binary"],
        strict=False,
    )
    for sample in ds_binary:
        assert "binary" in sample["task"]

    # Deterministic max_samples subset
    ds_subset = BigEarthNetDataset(
        manifest_path=manifest_file,
        data_root=data_root,
        max_samples=2,
        seed=42,
        strict=False,
    )
    assert len(ds_subset) == 2


def test_s2_default_is_model_bands(synthetic_dataset_dir, tmp_path):
    """Regression: default dataset output must be MODEL_S2_BANDS (10 bands), not all 12 raw bands."""
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"
    build_manifest(data_root=str(data_root), output_dir=str(manifest_out), seed=42)

    ds = BigEarthNetDataset(
        manifest_path=manifest_out / "manifest_full.jsonl",
        data_root=data_root,
        strict=False,
    )

    sample = ds[0]
    img_s2 = sample["image_s2"]

    assert img_s2.shape[0] == len(MODEL_S2_BANDS), (
        f"Expected {len(MODEL_S2_BANDS)} channels, got {img_s2.shape[0]}. "
        "BigEarthNetDataset(s2_bands=None) must default to MODEL_S2_BANDS, not S2_BAND_NAMES."
    )
    assert ds.s2_bands == MODEL_S2_BANDS
    assert "B01" not in ds.s2_bands, "B01 (Coastal aerosol, 60m) must be excluded from model input."
    assert "B09" not in ds.s2_bands, "B09 (Water vapour, 60m) must be excluded from model input."


def test_s2_explicit_12_band_raw_still_works(synthetic_dataset_dir, tmp_path):
    """The raw 12-band capability must still work when explicitly requested."""
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"
    build_manifest(data_root=str(data_root), output_dir=str(manifest_out), seed=42)

    ds_raw = BigEarthNetDataset(
        manifest_path=manifest_out / "manifest_full.jsonl",
        data_root=data_root,
        s2_bands="S2-all",  # Explicit: load all 12 raw bands
        strict=False,
    )
    assert ds_raw.s2_bands == S2_BAND_NAMES
    img_s2 = ds_raw[0]["image_s2"]
    assert img_s2.shape[0] == 12, f"Explicit s2_bands='S2-all' should produce 12 channels, got {img_s2.shape[0]}"


def test_s2_model_band_exact_order(synthetic_dataset_dir, tmp_path):
    """Verify exact channel order: B02→ch0, B03→ch1, ..., B12→ch9."""
    data_root, _, _, _ = synthetic_dataset_dir
    manifest_out = tmp_path / "manifests"
    build_manifest(data_root=str(data_root), output_dir=str(manifest_out), seed=42)

    ds = BigEarthNetDataset(
        manifest_path=manifest_out / "manifest_full.jsonl",
        data_root=data_root,
        strict=False,
    )

    expected_order = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    assert ds.s2_bands == expected_order, (
        f"Band order mismatch.\nExpected: {expected_order}\nGot:      {ds.s2_bands}"
    )
    # Verify position by index
    for i, band in enumerate(expected_order):
        assert ds.s2_bands[i] == band, f"Channel {i} should be {band}, got {ds.s2_bands[i]}"


def test_s2_12_channel_regression_triggers_model_error():
    """Regression: a 12-channel S2 tensor must raise a clear ValueError in RSInternVL.encode_vision."""
    import torch
    from models.rs_internvl.config import RSInternVLConfig
    from models.rs_internvl.model import RSInternVL

    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )
    model = RSInternVL(config)

    s1_ok = torch.randn(1, 2, 120, 120)
    s2_wrong_12ch = torch.randn(1, 12, 120, 120)  # raw BigEarthNet — not model-ready

    with pytest.raises(ValueError, match="S2 channel mismatch"):
        model.encode_vision(s1_ok, s2_wrong_12ch)
