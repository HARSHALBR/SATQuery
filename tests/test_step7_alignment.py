"""
Unit tests for STEP 7: Multimodal Generation Alignment & Diagnostic Properties.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.train_lora import MultimodalCollate


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)


@pytest.fixture(scope="module")
def sample_model():
    cfg = RSInternVLConfig()
    return RSInternVL(cfg)


def test_target_tokenization_yes_no(tokenizer):
    """Verify exact token IDs for YES, NO, and EOS in InternVL3-1B tokenizer."""
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    assert len(yes_ids) == 1, f"Expected single token for 'Yes', got {yes_ids}"
    assert len(no_ids) == 1, f"Expected single token for 'No', got {no_ids}"
    assert yes_ids[0] == 9454, f"Expected token 9454 for 'Yes', got {yes_ids[0]}"
    assert no_ids[0] == 2753, f"Expected token 2753 for 'No', got {no_ids[0]}"
    assert eos_id == 151645, f"Expected token 151645 for '<|im_end|>', got {eos_id}"


def test_multimodal_collate_label_masking(tokenizer):
    """Verify prompt tokens are masked with -100 while target completion is unmasked."""
    collate_fn = MultimodalCollate(tokenizer=tokenizer, max_seq_length=128, mask_prompt=True)
    sample = {
        "image_s1": torch.zeros(2, 120, 120),
        "image_s2": torch.zeros(10, 120, 120),
        "text": "Is forest present?",
        "target_text": "Yes, forest is present.",
        "image_id": "test_001",
        "task": "binary",
    }
    batch = collate_fn([sample])

    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()

    unmasked_indices = [i for i, lbl in enumerate(labels) if lbl != -100]
    masked_indices = [i for i, lbl in enumerate(labels) if lbl == -100]

    assert len(masked_indices) > 0, "Prompt tokens must be masked with -100"
    assert len(unmasked_indices) > 0, "Target tokens must be unmasked"

    # Verify unmasked labels decode to target
    unmasked_tokens = [labels[i] for i in unmasked_indices]
    decoded_target = tokenizer.decode(unmasked_tokens, skip_special_tokens=False)
    assert "Yes, forest is present." in decoded_target
    assert "<|im_end|>" in decoded_target


def test_visual_embeddings_finite_and_shapes(sample_model):
    """Verify S1, S2, and fusion output shapes and finite values."""
    s1_img = torch.randn(1, 2, 120, 120)
    s2_img = torch.randn(1, 10, 120, 120)

    s1_feat, s2_feat, s1_tok, s2_tok = sample_model.encode_vision(s1_img, s2_img)

    assert s1_tok.shape == (1, 225, 896), f"Expected [1, 225, 896], got {s1_tok.shape}"
    assert s2_tok.shape == (1, 225, 896), f"Expected [1, 225, 896], got {s2_tok.shape}"
    assert not torch.isnan(s1_tok).any(), "S1 tokens contain NaNs"
    assert not torch.isnan(s2_tok).any(), "S2 tokens contain NaNs"
    assert not torch.isinf(s1_tok).any(), "S1 tokens contain Infs"
    assert not torch.isinf(s2_tok).any(), "S2 tokens contain Infs"


def test_binary_normalization_math():
    """Verify binary probability normalization formula handles edge cases cleanly."""
    p_yes = 0.0007
    p_no = 0.0003
    denom = max(1e-12, p_yes + p_no)
    p_yes_norm = p_yes / denom
    p_no_norm = p_no / denom

    assert pytest.approx(p_yes_norm + p_no_norm, 1e-6) == 1.0
    assert p_yes_norm > p_no_norm
    assert pytest.approx(p_yes_norm, 1e-4) == 0.7000
