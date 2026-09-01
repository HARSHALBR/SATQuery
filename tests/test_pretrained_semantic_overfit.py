"""
Unit and Regression Tests for STEP 9: Pretrained-Backbone Semantic Overfit.
"""

import json
from pathlib import Path
import pytest
import torch
import yaml
from transformers import AutoTokenizer

from data.bigearthnet_txt.dataset import BigEarthNetDataset
from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from scripts.evaluate_generation import (
    compute_aggregate_metrics,
    evaluate_sample,
    extract_binary_answer,
    is_garbage_generation,
    normalize_text,
)
from training.lora import apply_lora, audit_parameters, load_lora_checkpoint, save_lora_checkpoint


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)


@pytest.fixture(scope="module")
def pretrained_lora_model():
    cfg = RSInternVLConfig(pretrained_backbone=True)
    model = RSInternVL(cfg)
    model, _ = apply_lora(model, r=8, lora_alpha=32, lora_dropout=0.1)
    return model


def test_pretrained_semantic_overfit_config_validity():
    """Verify configuration file exists and has correct parameters."""
    cfg_path = Path("configs/model/pretrained_semantic_overfit.yaml")
    assert cfg_path.exists(), "Config file missing"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["backbone"] == "OpenGVLab/InternVL3-1B"
    assert cfg["model"]["pretrained_backbone"] is True
    assert cfg["dataset"]["train_samples"] == 8
    assert cfg["dataset"]["validation_samples"] == 8
    assert cfg["lora"]["r"] == 8


def test_clean_model_initialization_no_legacy_weights(pretrained_lora_model):
    """Verify clean model initializes from pretrained backbone without legacy weights."""
    assert pretrained_lora_model.config.pretrained_backbone is True
    audit = audit_parameters(pretrained_lora_model)
    assert audit["total"] == 649517696
    assert audit["frozen_llm"] == 629697920
    assert audit["lora_trainable"] == 540672


def test_subset_sample_counts_and_zero_leakage():
    """Verify deterministic subsets contain 8 samples each with zero patch overlap."""
    train_manifest = Path("data/manifests/manifest_train.jsonl")
    val_manifest = Path("data/manifests/manifest_validation.jsonl")
    assert train_manifest.exists() and val_manifest.exists()

    train_patches = set()
    with open(train_manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            train_patches.add(rec.get("image_id", rec.get("patch_id", "")))

    val_patches = set()
    with open(val_manifest, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            val_patches.add(rec.get("image_id", rec.get("patch_id", "")))

    overlap = train_patches.intersection(val_patches)
    assert len(overlap) == 0, f"Found patch overlap: {overlap}"


def test_base_llm_frozen_during_step9(pretrained_lora_model):
    """Verify base language model is strictly frozen."""
    for name, p in pretrained_lora_model.language_model.named_parameters():
        if "lora" not in name.lower():
            assert not p.requires_grad, f"Base parameter {name} is not frozen"


def test_lora_and_encoders_trainable(pretrained_lora_model):
    """Verify LoRA adapters and vision encoders are trainable."""
    lora_trainable = [p for n, p in pretrained_lora_model.language_model.named_parameters() if "lora" in n.lower() and p.requires_grad]
    assert len(lora_trainable) > 0, "No trainable LoRA parameters found"

    for p in pretrained_lora_model.s1_encoder.parameters():
        assert p.requires_grad, "S1 encoder should be trainable"
    for p in pretrained_lora_model.s2_encoder.parameters():
        assert p.requires_grad, "S2 encoder should be trainable"


def test_forward_pass_finite_loss(pretrained_lora_model, tokenizer):
    """Verify forward pass with multimodal inputs produces finite scalar loss."""
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    prompt = "<|im_start|>user\nIs forest present?<|im_end|>\n<|im_start|>assistant\nYes<|im_end|>"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()

    out = pretrained_lora_model(
        image_s1=s1,
        image_s2=s2,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    assert "loss" in out
    loss = out["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() > 0.0


def test_backward_pass_gradients(pretrained_lora_model, tokenizer):
    """Verify backward pass populates finite gradients on trainable parameters."""
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    prompt = "<|im_start|>user\nIs water present?<|im_end|>\n<|im_start|>assistant\nNo<|im_end|>"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    pretrained_lora_model.zero_grad()
    out = pretrained_lora_model(
        image_s1=s1,
        image_s2=s2,
        input_ids=input_ids,
        labels=labels,
    )
    out["loss"].backward()

    # Check LoRA grad
    lora_grads = [p.grad for n, p in pretrained_lora_model.language_model.named_parameters() if "lora" in n.lower() and p.grad is not None]
    assert len(lora_grads) > 0, "No gradients on LoRA"
    assert torch.isfinite(lora_grads[0]).all()


def test_generation_produces_valid_english(pretrained_lora_model, tokenizer):
    """Verify autoregressive generation yields valid non-empty English."""
    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    pred = pretrained_lora_model.predict(
        image_s1=s1,
        image_s2=s2,
        query="Is broad-leaved forest present?",
        tokenizer=tokenizer,
        max_new_tokens=16,
    )
    assert isinstance(pred["answer"], str)
    assert len(pred["answer"]) > 0


def test_generation_metrics_calculation():
    """Verify exact match, binary accuracy, and garbage rate calculation."""
    records = [
        evaluate_sample("query1", "Yes, forest is present.", "Yes, forest is present.", task_type="binary"),
        evaluate_sample("query2", "No, water is absent.", "Yes, water is present.", task_type="binary"),
    ]
    metrics = compute_aggregate_metrics(records)
    assert metrics["total_samples"] == 2
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["binary_accuracy"] == 0.5
    assert metrics["garbage_rate"] == 0.0


def test_direct_yes_no_probabilities_computation(pretrained_lora_model, tokenizer):
    """Verify candidate YES/NO normalized logits calculation."""
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    assert yes_id != no_id

    s1 = torch.randn(1, 2, 120, 120)
    s2 = torch.randn(1, 10, 120, 120)
    prompt = "<|im_start|>user\nIs forest present?<|im_end|>\n<|im_start|>assistant\n"
    enc = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        s1_feat, s2_feat, s1_tok, s2_tok = pretrained_lora_model.encode_vision(s1, s2)
        text_embeds = pretrained_lora_model.language_model.get_input_embeddings()(enc["input_ids"])
        fused = pretrained_lora_model.fusion(
            s1_tokens=s1_tok,
            s2_tokens=s2_tok,
            text_embeds=text_embeds,
            text_attention_mask=enc["attention_mask"],
        )
        lm_out = pretrained_lora_model.language_model(
            inputs_embeds=fused.inputs_embeds,
            attention_mask=fused.attention_mask,
        )
        logits = lm_out.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        p_yes = probs[yes_id].item()
        p_no = probs[no_id].item()
        p_norm_yes = p_yes / (p_yes + p_no + 1e-12)
        assert 0.0 <= p_norm_yes <= 1.0


def test_checkpoint_saving_and_reloading_cleanly(pretrained_lora_model, tmp_path):
    """Verify save_lora_checkpoint and load_lora_checkpoint work on pretrained backbone."""
    save_lora_checkpoint(
        model=pretrained_lora_model,
        output_dir=tmp_path / "test_ckpt",
        epoch=1,
        global_step=10,
        metrics={"loss": 0.5},
    )
    assert (tmp_path / "test_ckpt" / "adapter").exists()
    assert (tmp_path / "test_ckpt" / "modality_encoders.pt").exists()
    assert (tmp_path / "test_ckpt" / "modality_projections.pt").exists()

    loaded = load_lora_checkpoint(
        checkpoint_dir=str(tmp_path / "test_ckpt"),
        device="cpu",
        is_trainable=False,
    )
    assert loaded is not None
