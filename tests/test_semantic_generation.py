"""
Unit tests for Step 6 Semantic Overfit & Generation Validation.

Verifies:
1. Text normalization rules
2. Binary answer extraction
3. Garbage generation detection
4. Repetition detection
5. Exact match metric computation
6. Deterministic train/validation patch isolation
7. Generation output schema conformity
8. Base LLM parameters frozen assertion
9. LoRA adapter parameters trainable assertion
10. Semantic metrics bounded in [0.0, 1.0]
"""

import json
from pathlib import Path
import pytest
import torch

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from scripts.evaluate_generation import (
    compute_aggregate_metrics,
    compute_exact_match,
    detect_repetition,
    extract_binary_answer,
    evaluate_sample,
    is_garbage_generation,
    normalize_text,
)
from training.lora import apply_lora, audit_parameters

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_text_normalization():
    """Verify deterministic text normalization."""
    assert normalize_text("  Yes, Coniferous Forest is Present!  ") == "yes coniferous forest is present"
    assert normalize_text("No,   water body... is NOT present??") == "no water body is not present"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text("Yes.\n\nYes  ") == "yes yes"


def test_binary_answer_extraction():
    """Verify robust binary answer classification."""
    assert extract_binary_answer("Yes, coniferous forest is present.") == "YES"
    assert extract_binary_answer("yes, broad-leaved forest is detected.") == "YES"
    assert extract_binary_answer("No, water body is not present.") == "NO"
    assert extract_binary_answer("no, arable land is absent.") == "NO"
    assert extract_binary_answer("The dominant land cover is forest.") == "UNKNOWN"
    assert extract_binary_answer("WINDOW/rand$db本来 DbSet本来") == "UNKNOWN"
    assert extract_binary_answer("") == "UNKNOWN"
    assert extract_binary_answer(None) == "UNKNOWN"


def test_garbage_generation_detection():
    """Verify detection of garbage, empty, and corrupted token generation."""
    empty_res = is_garbage_generation("")
    assert empty_res["is_garbage"] is True
    assert empty_res["is_empty"] is True
    assert empty_res["is_valid"] is False

    corrupted_res = is_garbage_generation("WINDOW/rand$db本来 DbSet本来<pair")
    assert corrupted_res["is_garbage"] is True
    assert corrupted_res["is_valid"] is False

    valid_res = is_garbage_generation("Yes, coniferous forest is present in this patch.")
    assert valid_res["is_valid"] is True
    assert valid_res["is_garbage"] is False


def test_repetition_detection():
    """Verify detection of token/word repetition loops."""
    rep_text = "donate donate donate donate donate donate donate donate donate"
    is_rep, score = detect_repetition(rep_text)
    assert is_rep is True
    assert score > 0.5

    subword_rep = "enalenalenalenalenalenalenalenalenal"
    is_rep_sub, score_sub = detect_repetition(subword_rep)
    assert is_rep_sub is True

    clean_text = "Yes, coniferous forest is present."
    is_rep_clean, score_clean = detect_repetition(clean_text)
    assert is_rep_clean is False


def test_exact_match_metric():
    """Verify exact match metric under normalization."""
    target = "Yes, coniferous forest is present."
    gen_exact = "yes coniferous forest is present"
    gen_punct = "Yes, coniferous forest is present!"
    gen_different = "No, coniferous forest is not present."

    assert compute_exact_match(target, gen_exact) is True
    assert compute_exact_match(target, gen_punct) is True
    assert compute_exact_match(target, gen_different) is False


def test_train_validation_patch_isolation():
    """Verify 0 patch overlap between train and validation manifests."""
    train_path = REPO_ROOT / "data/manifests/manifest_train.jsonl"
    val_path = REPO_ROOT / "data/manifests/manifest_validation.jsonl"

    if train_path.exists() and val_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_patches = {json.loads(line).get("image_id") for line in f if line.strip()}
        with open(val_path, "r", encoding="utf-8") as f:
            val_patches = {json.loads(line).get("image_id") for line in f if line.strip()}

        overlap = train_patches.intersection(val_patches)
        assert len(overlap) == 0, f"Found overlapping patch IDs: {overlap}"


def test_generation_output_schema():
    """Verify output schema of evaluate_sample()."""
    sample = evaluate_sample(
        query="Is coniferous forest present in this satellite patch?",
        target="Yes, coniferous forest is present.",
        generated="Yes, coniferous forest is present.",
        patch_id="test_patch_01",
        task_type="binary:presence",
    )

    expected_keys = {
        "patch_id",
        "query",
        "target",
        "generated",
        "normalized_target",
        "normalized_generated",
        "binary_target",
        "binary_prediction",
        "binary_match",
        "exact_match",
        "generation_valid",
        "generation_quality",
        "task_type",
    }
    assert expected_keys.issubset(set(sample.keys()))
    assert sample["binary_target"] == "YES"
    assert sample["binary_prediction"] == "YES"
    assert sample["exact_match"] is True
    assert sample["generation_valid"] is True


def test_base_llm_remains_frozen():
    """Verify base language model backbone parameters have requires_grad=False."""
    cfg = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )
    model = RSInternVL(cfg)
    model, audit = apply_lora(
        model=model,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
    )

    for name, p in model.language_model.named_parameters():
        if "lora" not in name.lower():
            assert not p.requires_grad, f"Base LLM parameter '{name}' should be frozen!"


def test_lora_parameters_remain_trainable():
    """Verify LoRA adapter parameters have requires_grad=True."""
    cfg = RSInternVLConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        llm_hidden_dim=896,
        vocab_size=1000,
    )
    model = RSInternVL(cfg)
    model, audit = apply_lora(
        model=model,
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        freeze_s1_encoder=False,
        freeze_s2_encoder=False,
    )

    lora_params = [p for n, p in model.language_model.named_parameters() if "lora" in n.lower()]
    assert len(lora_params) > 0, "Zero LoRA parameters found in model!"
    for p in lora_params:
        assert p.requires_grad, "LoRA parameter should be trainable!"


def test_semantic_metrics_are_bounded():
    """Verify computed aggregate metrics are strictly bounded in [0.0, 1.0]."""
    mock_records = [
        evaluate_sample("q1", "Yes, forest present.", "Yes, forest present.", "p1"),
        evaluate_sample("q2", "No, water present.", "WINDOW/rand$db", "p2"),
        evaluate_sample("q3", "Yes, urban present.", "yes urban present", "p3"),
        evaluate_sample("q4", "No, crops present.", "donate donate donate", "p4"),
    ]

    metrics = compute_aggregate_metrics(mock_records)
    for k, v in metrics.items():
        if k != "total_samples":
            assert 0.0 <= v <= 1.0, f"Metric '{k}' out of bounds: {v}"

    assert metrics["total_samples"] == 4
    assert metrics["exact_match_accuracy"] == 0.5
    assert metrics["binary_accuracy"] == 0.5
    assert metrics["garbage_rate"] == 0.5
