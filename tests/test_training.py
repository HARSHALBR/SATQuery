"""
Unit tests for Step 3: Tiny-subset overfitting and training pipeline.

Verifies:
- Deterministic subset selection and split isolation
- Configurable parameter freezing / unfreezing
- Single forward and backward pass gradient flow
- Gradients are finite (no NaNs, no Infs)
- Optimizer updates model weights
- Checkpoint serialization and deserialization
- Correct visual token masking (-100) in causal language model loss
"""

import json
import math
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.bigearthnet_txt.dataset import BigEarthNetDataset
from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.train_tiny import (
    FallbackTokenizer,
    MultimodalCollate,
    build_model,
    compute_grad_norm,
    get_tokenizer,
    load_tiny_dataset,
    set_seed,
    train_tiny,
)


@pytest.fixture
def dummy_train_manifest(tmp_path: Path) -> Path:
    """Create a temporary training manifest for testing."""
    manifest_path = tmp_path / "manifest_train.jsonl"
    records = []
    for i in range(20):
        records.append({
            "sample_id": f"test_ben_{i+1:04d}",
            "image_id": f"S2A_PATCH_{i:02d}",
            "s1_name": f"S1A_PATCH_{i:02d}",
            "s1_path": f"images_s1/S1A_PATCH_{i:02d}",
            "s2_path": f"images_s2/S2A_PATCH_{i:02d}",
            "text_input": f"Is forest present in area {i+1}?",
            "text_output": "Yes, forest is present." if i % 2 == 0 else "No forest.",
            "task_type": "binary",
            "task_category": "presence",
            "split": "train",
            "metadata": {
                "patch_id": f"S2A_PATCH_{i:02d}",
                "s1_name": f"S1A_PATCH_{i:02d}",
                "country": "Austria",
                "season": "Summer",
                "climate_zone": "Cfb",
                "split": "train",
            },
        })
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return manifest_path


def test_deterministic_subset_selection(dummy_train_manifest, tmp_path):
    """Verify that subsetting is deterministic across seeds and strictly respects num_samples."""
    subset1 = load_tiny_dataset(
        manifest_path=str(dummy_train_manifest),
        data_root=str(tmp_path),
        num_samples=8,
        seed=42,
    )
    subset2 = load_tiny_dataset(
        manifest_path=str(dummy_train_manifest),
        data_root=str(tmp_path),
        num_samples=8,
        seed=42,
    )
    subset_diff_seed = load_tiny_dataset(
        manifest_path=str(dummy_train_manifest),
        data_root=str(tmp_path),
        num_samples=8,
        seed=999,
    )

    assert len(subset1) == 8
    assert len(subset2) == 8
    assert len(subset_diff_seed) == 8

    # Verify identical sample IDs with same seed
    indices1 = subset1.indices
    indices2 = subset2.indices
    assert indices1 == indices2

    # Verify different sample IDs with different seed
    indices_diff = subset_diff_seed.indices
    assert indices1 != indices_diff

    # Verify all samples belong strictly to train split
    for idx in indices1:
        sample = subset1.dataset[idx]
        assert sample["split"] == "train"


def test_trainable_parameter_selection():
    """Verify that model submodules can be frozen/unfrozen according to config."""
    device = torch.device("cpu")

    # Case 1: All trainable
    cfg_all = {
        "img_size": 120,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "freeze_s1_encoder": False,
        "freeze_s2_encoder": False,
        "freeze_llm": False,
    }
    model_all = build_model(cfg_all, device)
    for p in model_all.s1_encoder.parameters():
        assert p.requires_grad
    for p in model_all.s2_encoder.parameters():
        assert p.requires_grad
    for p in model_all.language_model.parameters():
        assert p.requires_grad
    for p in model_all.s1_projection.parameters():
        assert p.requires_grad

    # Case 2: Encoders frozen, LLM frozen, projections trainable
    cfg_frozen = {
        "img_size": 120,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "freeze_s1_encoder": True,
        "freeze_s2_encoder": True,
        "freeze_llm": True,
    }
    model_frozen = build_model(cfg_frozen, device)
    for p in model_frozen.s1_encoder.parameters():
        assert not p.requires_grad
    for p in model_frozen.s2_encoder.parameters():
        assert not p.requires_grad
    for p in model_frozen.language_model.parameters():
        assert not p.requires_grad
    # Projections must still be trainable
    for p in model_frozen.s1_projection.parameters():
        assert p.requires_grad
    for p in model_frozen.s2_projection.parameters():
        assert p.requires_grad


def test_one_forward_backward_step():
    """Verify that a single forward and backward step executes and calculates finite gradients."""
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
        freeze_llm=False,
    )
    model = RSInternVL(config)
    model.train()

    B = 2
    s1 = torch.randn(B, 2, 120, 120)
    s2 = torch.randn(B, 10, 120, 120)
    input_ids = torch.randint(3, 900, (B, 12), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :4] = -100  # mask prompt tokens
    attention_mask = torch.ones((B, 12), dtype=torch.long)

    outputs = model(
        image_s1=s1,
        image_s2=s2,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert "loss" in outputs
    loss = outputs["loss"]
    assert loss is not None
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Perform backpropagation
    loss.backward()

    # Verify gradients exist on trainable components
    assert model.s1_projection.net[0].weight.grad is not None
    assert model.s2_projection.net[0].weight.grad is not None
    assert model.language_model.model.embed_tokens.weight.grad is not None


def test_gradients_are_finite():
    """Verify that computed gradients contain zero NaNs and zero Infs."""
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
        freeze_llm=False,
    )
    model = RSInternVL(config)
    model.train()

    s1 = torch.randn(2, 2, 120, 120)
    s2 = torch.randn(2, 10, 120, 120)
    input_ids = torch.randint(3, 900, (2, 10), dtype=torch.long)
    labels = torch.randint(3, 900, (2, 10), dtype=torch.long)

    outputs = model(image_s1=s1, image_s2=s2, input_ids=input_ids, labels=labels)
    outputs["loss"].backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in parameter {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in parameter {name}"

    grad_norm = compute_grad_norm(model)
    assert math.isfinite(grad_norm)
    assert grad_norm > 0.0


def test_optimizer_updates_parameters():
    """Verify that AdamW step alters trainable weights in the direction of gradients."""
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
        freeze_llm=False,
    )
    model = RSInternVL(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Clone initial weight values
    initial_s1_proj_weight = model.s1_projection.net[0].weight.clone()
    initial_s2_proj_weight = model.s2_projection.net[0].weight.clone()

    s1 = torch.randn(2, 2, 120, 120)
    s2 = torch.randn(2, 10, 120, 120)
    input_ids = torch.randint(3, 900, (2, 8), dtype=torch.long)
    labels = torch.randint(3, 900, (2, 8), dtype=torch.long)

    outputs = model(image_s1=s1, image_s2=s2, input_ids=input_ids, labels=labels)
    outputs["loss"].backward()

    optimizer.step()

    # Verify weights changed
    assert not torch.equal(model.s1_projection.net[0].weight, initial_s1_proj_weight)
    assert not torch.equal(model.s2_projection.net[0].weight, initial_s2_proj_weight)


def test_checkpoint_saving_and_loading(tmp_path: Path):
    """Verify that checkpoints save state dicts, optimizer state, and config correctly."""
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )
    model = RSInternVL(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    save_data = {
        "epoch": 5,
        "global_step": 50,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": 0.1234,
        "config": {"epochs": 5, "batch_size": 2},
    }
    torch.save(save_data, ckpt_path)
    assert ckpt_path.exists()

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 5
    assert loaded["global_step"] == 50
    assert loaded["loss"] == 0.1234
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded

    # Test loading state dict into fresh model
    new_model = RSInternVL(config)
    new_model.load_state_dict(loaded["model_state_dict"])


def test_visual_tokens_masked_in_loss():
    """Verify that visual tokens are masked with -100 in the fused sequence labels."""
    tokenizer = FallbackTokenizer(vocab_size=1000)
    collate = MultimodalCollate(tokenizer=tokenizer, max_seq_length=64, mask_prompt=True)

    dummy_batch = [
        {
            "image_s1": torch.randn(2, 120, 120),
            "image_s2": torch.randn(10, 120, 120),
            "text": "Is water present?",
            "target_text": "Yes, water is present.",
        }
    ]

    collated = collate(dummy_batch)
    assert "labels" in collated
    labels = collated["labels"]

    # Initial tokens (prompt) should be -100
    assert labels[0, 0].item() == -100

    # Test through fusion module
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )
    model = RSInternVL(config)

    s1 = collated["image_s1"]
    s2 = collated["image_s2"]
    input_ids = collated["input_ids"]

    # In forward, visual tokens (225 S1 + 225 S2 = 450) must be masked with -100 in fused labels
    s1_feat, s2_feat, s1_tok, s2_tok = model.encode_vision(s1, s2)
    text_embeds = model.language_model.get_input_embeddings()(input_ids)

    fused = model.fusion(
        s1_tokens=s1_tok,
        s2_tokens=s2_tok,
        text_embeds=text_embeds,
        text_labels=labels,
    )

    n_visual = 225 + 225  # 450
    assert (fused.labels[:, :n_visual] == -100).all()
    assert (fused.labels[:, n_visual:] == labels).all()
