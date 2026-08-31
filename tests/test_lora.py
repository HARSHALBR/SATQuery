"""
Unit and integration tests for Step 4: PEFT / LoRA Fine-Tuning Module and Pipeline.

Verifies:
- Dynamic target module discovery on Qwen2/InternVL architecture
- Clear failure on zero matched target modules
- Base LLM parameter freezing
- LoRA adapter parameters trainable
- Modality projections trainable
- S1/S2 encoder freezing configurability
- Forward and backward gradient propagation with optimizer step
- Modular checkpoint serialization and deserialization
- Reconstructed model inference equivalence
- Validation split isolation (loss computation without gradients)
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
from peft import PeftModel

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.lora import (
    apply_lora,
    audit_parameters,
    build_lora_config,
    find_target_modules,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from training.train_lora import FallbackTokenizer, MultimodalCollate, evaluate, set_seed


def get_lightweight_config(
    freeze_s1: bool = False,
    freeze_s2: bool = False,
) -> RSInternVLConfig:
    """Helper creating a lightweight RSInternVLConfig for rapid unit testing."""
    return RSInternVLConfig(
        img_size=120,
        s1_channels=2,
        s1_hidden_dim=512,
        s2_channels=10,
        s2_hidden_dim=768,
        projection_hidden_dim=1024,
        llm_hidden_dim=896,
        vocab_size=1000,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        freeze_s1_encoder=freeze_s1,
        freeze_s2_encoder=freeze_s2,
        freeze_llm=True,
    )


def test_lora_module_matching_and_discovery():
    """Verify that target module discovery inspects actual Qwen2 module hierarchy."""
    config = get_lightweight_config()
    model = RSInternVL(config)

    # Discover target modules in model.language_model
    matched = find_target_modules(model.language_model, ["q_proj", "v_proj", "nonexistent_proj"])
    assert "q_proj" in matched
    assert "v_proj" in matched
    assert "nonexistent_proj" not in matched
    assert len(matched) == 2


def test_zero_target_module_detection_fails_clearly():
    """Verify that specifying non-existent target module names raises a clear ValueError."""
    config = get_lightweight_config()
    model = RSInternVL(config)

    with pytest.raises(ValueError, match="Zero target modules matched candidate list"):
        find_target_modules(model.language_model, ["completely_fake_layer_name_xyz"])


def test_lora_adapter_insertion_and_freezing():
    """Verify that base LLM backbone is frozen while LoRA adapters are trainable."""
    config = get_lightweight_config()
    model = RSInternVL(config)

    adapted_model, audit = apply_lora(
        model=model,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
    )

    assert isinstance(adapted_model.language_model, PeftModel)

    # 1. Base LLM parameters must be FROZEN
    base_llm_frozen_count = 0
    lora_trainable_count = 0

    for name, param in adapted_model.language_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"LoRA parameter {name} should be trainable!"
            lora_trainable_count += param.numel()
        else:
            assert not param.requires_grad, f"Base LLM parameter {name} should be frozen!"
            base_llm_frozen_count += param.numel()

    assert lora_trainable_count > 0
    assert base_llm_frozen_count > 0
    assert audit["lora_trainable"] == lora_trainable_count
    assert audit["frozen_llm"] == base_llm_frozen_count


def test_modality_projections_trainable():
    """Verify that S1 and S2 projection layers are explicitly trainable."""
    config = get_lightweight_config()
    model = RSInternVL(config)

    adapted_model, audit = apply_lora(
        model=model,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    )

    for p in adapted_model.s1_projection.parameters():
        assert p.requires_grad, "S1 projection parameter must be trainable!"
    for p in adapted_model.s2_projection.parameters():
        assert p.requires_grad, "S2 projection parameter must be trainable!"

    assert audit["projections_trainable"] > 0
    assert audit["projections_trainable"] == audit["projections_total"]


def test_encoder_freezing_configuration():
    """Verify that S1 and S2 modality encoders obey freezing flags."""
    # Case 1: Trainable (default)
    model1 = RSInternVL(get_lightweight_config())
    adapted1, audit1 = apply_lora(
        model=model1,
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
    )
    for p in adapted1.s1_encoder.parameters():
        assert p.requires_grad
    for p in adapted1.s2_encoder.parameters():
        assert p.requires_grad
    assert audit1["s1_encoder_trainable"] == audit1["s1_encoder_total"]
    assert audit1["s2_encoder_trainable"] == audit1["s2_encoder_total"]

    # Case 2: Frozen
    model2 = RSInternVL(get_lightweight_config())
    adapted2, audit2 = apply_lora(
        model=model2,
        freeze_s1_encoder=True,
        freeze_s2_encoder=True,
    )
    for p in adapted2.s1_encoder.parameters():
        assert not p.requires_grad
    for p in adapted2.s2_encoder.parameters():
        assert not p.requires_grad
    assert audit2["s1_encoder_trainable"] == 0
    assert audit2["s2_encoder_trainable"] == 0


def test_lora_forward_backward_gradient_flow():
    """Verify single forward and backward pass calculates finite gradients and updates weights."""
    config = get_lightweight_config()
    model = RSInternVL(config)
    adapted_model, _ = apply_lora(
        model=model,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
    )
    adapted_model.train()

    trainable_params = [p for p in adapted_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    B = 2
    s1 = torch.randn(B, 2, 120, 120)
    s2 = torch.randn(B, 10, 120, 120)
    input_ids = torch.randint(3, 900, (B, 10), dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :4] = -100  # mask prompt tokens
    attention_mask = torch.ones((B, 10), dtype=torch.long)

    # Initial weights
    initial_s1_proj_weight = adapted_model.s1_projection.net[0].weight.clone()
    # Find a LoRA A weight tensor
    lora_weight = None
    for name, p in adapted_model.language_model.named_parameters():
        if "lora_A" in name:
            lora_weight = p.clone()
            break
    assert lora_weight is not None

    # Forward
    outputs = adapted_model(
        image_s1=s1,
        image_s2=s2,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = outputs["loss"]
    assert loss is not None
    assert math.isfinite(loss.item())

    # Backward
    loss.backward()

    # Verify gradients exist on trainable modules and are absent on frozen LLM
    for name, p in adapted_model.language_model.named_parameters():
        if "lora_" in name:
            assert p.grad is not None, f"Gradient missing on LoRA param: {name}"
            assert math.isfinite(p.grad.norm().item())
        else:
            assert p.grad is None, f"Frozen param should have no gradient: {name}"

    assert adapted_model.s1_projection.net[0].weight.grad is not None

    # Optimizer step
    optimizer.step()

    # Verify weights changed
    assert not torch.equal(adapted_model.s1_projection.net[0].weight, initial_s1_proj_weight)


def test_modular_checkpoint_saving_and_reloading(tmp_path: Path):
    """Verify that modular checkpoints save without base LLM and reload identically."""
    set_seed(42)
    config = get_lightweight_config()
    model = RSInternVL(config)
    adapted_model, _ = apply_lora(
        model=model,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
    )
    adapted_model.eval()

    optimizer = torch.optim.AdamW(
        [p for p in adapted_model.parameters() if p.requires_grad], lr=1e-4
    )

    ckpt_dir = tmp_path / "test_lora_checkpoint"
    saved_path = save_lora_checkpoint(
        model=adapted_model,
        output_dir=ckpt_dir,
        config={"training": {"epochs": 3, "lr": 1e-4}},
        optimizer=optimizer,
        epoch=2,
        global_step=20,
        metrics={"train_loss": 1.234, "val_loss": 1.567},
    )

    assert saved_path.exists()
    assert (ckpt_dir / "adapter").exists()
    assert (ckpt_dir / "adapter" / "adapter_config.json").exists()
    assert (ckpt_dir / "modality_encoders.pt").exists()
    assert (ckpt_dir / "modality_projections.pt").exists()
    assert (ckpt_dir / "training_state.pt").exists()
    assert (ckpt_dir / "config.yaml").exists()
    assert (ckpt_dir / "metrics.json").exists()

    # Reload checkpoint using deterministic base configuration
    set_seed(42)
    reloaded_model = load_lora_checkpoint(
        checkpoint_dir=ckpt_dir,
        config_override=config,
        device="cpu",
    )
    reloaded_model.eval()

    # Test forward pass equivalence between original adapted model and reloaded model
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    input_ids = torch.randint(3, 900, (1, 8), dtype=torch.long)

    with torch.no_grad():
        out_orig = adapted_model(image_s1=s1, image_s2=s2, input_ids=input_ids)
        out_reloaded = reloaded_model(image_s1=s1, image_s2=s2, input_ids=input_ids)

    assert torch.allclose(out_orig["logits"], out_reloaded["logits"], atol=1e-4)


def test_validation_pipeline_isolated():
    """Verify validation evaluation computes loss without modifying weights or computing gradients."""
    config = get_lightweight_config()
    model = RSInternVL(config)
    adapted_model, _ = apply_lora(model=model, r=8, lora_alpha=32)

    tokenizer = FallbackTokenizer(vocab_size=1000)
    collate = MultimodalCollate(tokenizer=tokenizer, max_seq_length=256)

    val_samples = [
        {
            "image_s1": torch.randn(2, 120, 120),
            "image_s2": torch.randn(10, 120, 120),
            "text": "Is water present?",
            "target_text": "Yes, water is present in this patch.",
        },
        {
            "image_s1": torch.randn(2, 120, 120),
            "image_s2": torch.randn(10, 120, 120),
            "text": "Is forest present?",
            "target_text": "No forest is present.",
        },
    ]

    val_loader = torch.utils.data.DataLoader(
        val_samples, batch_size=2, shuffle=False, collate_fn=collate
    )

    val_loss = evaluate(
        model=adapted_model,
        val_loader=val_loader,
        device=torch.device("cpu"),
        use_amp=False,
        amp_dtype=torch.float32,
    )

    assert isinstance(val_loss, float)
    assert math.isfinite(val_loss)
    assert val_loss > 0.0

    # Ensure zero gradients exist after validation
    for p in adapted_model.parameters():
        assert p.grad is None
