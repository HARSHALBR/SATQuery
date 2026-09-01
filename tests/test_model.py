"""
Unit tests for RS-InternVL model architecture, encoders, projections, fusion, and forward pass.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.fusion import MultimodalTokenFusion
from models.rs_internvl.model import RSInternVL
from models.rs_internvl.projection import S1Projection, S2Projection
from models.rs_internvl.s1_encoder import S1Encoder
from models.rs_internvl.s2_encoder import S2Encoder


def test_s1_encoder_forward_and_freezing():
    encoder = S1Encoder(in_channels=2, hidden_dim=512, freeze_backbone=True)
    x = torch.randn(2, 2, 120, 120)
    out = encoder(x)

    # 120 / 8 = 15 -> 15 * 15 = 225 tokens
    assert out.shape == (2, 225, 512)
    assert not torch.isnan(out).any()

    # Verify frozen weights
    for param in encoder.parameters():
        assert not param.requires_grad

    # Test unfreeze
    encoder.unfreeze()
    for param in encoder.parameters():
        assert param.requires_grad


def test_s2_encoder_forward_and_freezing():
    encoder = S2Encoder(in_channels=10, hidden_dim=768, img_size=120, freeze_backbone=True)
    x = torch.randn(2, 10, 120, 120)
    out = encoder(x)

    # 120 / 8 = 15 -> 15 * 15 = 225 tokens
    assert out.shape == (2, 225, 768)
    assert not torch.isnan(out).any()

    # Verify frozen weights
    for param in encoder.parameters():
        assert not param.requires_grad


def test_projection_layers():
    s1_proj = S1Projection(in_dim=512, out_dim=896, hidden_dim=1024)
    s2_proj = S2Projection(in_dim=768, out_dim=896, hidden_dim=1024)

    s1_in = torch.randn(2, 225, 512)
    s2_in = torch.randn(2, 225, 768)

    s1_out = s1_proj(s1_in)
    s2_out = s2_proj(s2_in)

    assert s1_out.shape == (2, 225, 896)
    assert s2_out.shape == (2, 225, 896)

    # Projections must be trainable
    for param in s1_proj.parameters():
        assert param.requires_grad
    for param in s2_proj.parameters():
        assert param.requires_grad


def test_multimodal_token_fusion_and_label_masking():
    fusion = MultimodalTokenFusion(hidden_dim=896)

    B = 2
    N_s1 = 225
    N_s2 = 225
    L_text = 10

    s1_tokens = torch.randn(B, N_s1, 896)
    s2_tokens = torch.randn(B, N_s2, 896)
    text_embeds = torch.randn(B, L_text, 896)
    text_labels = torch.randint(0, 1000, (B, L_text), dtype=torch.long)
    text_mask = torch.ones((B, L_text), dtype=torch.long)

    fused = fusion(
        s1_tokens=s1_tokens,
        s2_tokens=s2_tokens,
        text_embeds=text_embeds,
        text_attention_mask=text_mask,
        text_labels=text_labels,
    )

    total_len = N_s1 + N_s2 + L_text  # 460
    assert fused.inputs_embeds.shape == (B, total_len, 896)
    assert fused.attention_mask.shape == (B, total_len)
    assert fused.position_ids.shape == (B, total_len)
    assert fused.labels.shape == (B, total_len)

    # Verify visual token positions are masked with -100 in labels
    assert (fused.labels[:, :N_s1 + N_s2] == -100).all()
    # Verify text positions retain ground truth labels
    assert (fused.labels[:, N_s1 + N_s2:] == text_labels).all()


def test_full_model_forward_pass():
    # Use lightweight configuration for rapid testing
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )

    model = RSInternVL(config)
    model.eval()

    s1 = torch.randn(2, 2, 120, 120)
    s2 = torch.randn(2, 10, 120, 120)
    input_ids = torch.randint(0, 1000, (2, 8), dtype=torch.long)
    labels = torch.randint(0, 1000, (2, 8), dtype=torch.long)

    with torch.no_grad():
        outputs = model(
            image_s1=s1,
            image_s2=s2,
            input_ids=input_ids,
            labels=labels,
        )

    assert "logits" in outputs
    assert "s1_features" in outputs
    assert "s2_features" in outputs
    assert "s1_projected" in outputs
    assert "s2_projected" in outputs
    assert "fused_features" in outputs

    # Total seq len = 225 (S1) + 225 (S2) + 8 (Text) = 458
    assert outputs["logits"].shape == (2, 458, 1000)
    assert outputs["fused_features"].shape == (2, 458, 896)
    assert not torch.isnan(outputs["logits"]).any()


def test_batch_size_greater_than_one():
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )
    model = RSInternVL(config)

    s1 = torch.randn(3, 2, 120, 120)
    s2 = torch.randn(3, 10, 120, 120)
    input_ids = torch.randint(0, 1000, (3, 5), dtype=torch.long)

    with torch.no_grad():
        outputs = model(image_s1=s1, image_s2=s2, input_ids=input_ids)

    assert outputs["logits"].shape[0] == 3


def test_invalid_shape_handling():
    encoder_s1 = S1Encoder(in_channels=2)
    with pytest.raises(ValueError, match="Expected 2 input channels"):
        # 3 channels passed instead of 2
        encoder_s1(torch.randn(1, 3, 120, 120))

    encoder_s2 = S2Encoder(in_channels=10)
    with pytest.raises(ValueError, match="Expected 10 input channels"):
        # 12 channels passed instead of 10
        encoder_s2(torch.randn(1, 12, 120, 120))


def test_predict_interface():
    config = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )
    model = RSInternVL(config)

    s1 = torch.randn(2, 120, 120)
    s2 = torch.randn(10, 120, 120)

    pred = model.predict(image_s1=s1, image_s2=s2, query="Verify water presence")
    assert isinstance(pred, dict)
    assert "answer" in pred
    assert "claim" in pred
    assert "claim_type" in pred
    assert "model_score" in pred
    assert "model_version" in pred
    assert "grounding" in pred
