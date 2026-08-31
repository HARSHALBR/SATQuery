"""
Unit & Regression Tests for STEP 8: Pretrained Language Backbone Restoration.
"""

import pytest
import torch
from transformers import AutoTokenizer

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.lora import apply_lora


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)


@pytest.fixture(scope="module")
def pretrained_model():
    cfg = RSInternVLConfig(pretrained_backbone=True)
    return RSInternVL(cfg)


def test_pretrained_checkpoint_identifier_exists():
    """Verify default model configuration specifies OpenGVLab/InternVL3-1B and pretrained_backbone=True."""
    cfg = RSInternVLConfig()
    assert cfg.model_id == "OpenGVLab/InternVL3-1B"
    assert cfg.pretrained_backbone is True
    assert cfg.llm_hidden_dim == 896
    assert cfg.vocab_size == 151674


def test_language_model_loads_pretrained_weights(pretrained_model):
    """Verify language model backbone is populated with non-trivial weights."""
    lm = pretrained_model.language_model
    embed_w = lm.get_input_embeddings().weight
    assert embed_w is not None
    assert embed_w.shape == (151674, 896)
    assert not torch.isnan(embed_w).any()
    assert not (embed_w == 0).all()


def test_language_weights_non_zero_and_variance(pretrained_model):
    """Verify weight norm and distribution match calibrated pretrained weights."""
    embed_w = pretrained_model.language_model.get_input_embeddings().weight.detach()
    w_std = torch.std(embed_w).item()
    w_mean = torch.mean(embed_w).item()
    # Pretrained embedding matrix for Qwen2 has characteristic variance
    assert 0.01 < w_std < 0.2, f"Expected pretrained std in (0.01, 0.2), got {w_std}"
    assert abs(w_mean) < 0.01, f"Expected near-zero mean, got {w_mean}"


def test_hidden_dimension_matches_projection(pretrained_model):
    """Verify LLM hidden dimension perfectly matches S1/S2 projection outputs and fusion."""
    llm_dim = pretrained_model.config.llm_hidden_dim
    assert llm_dim == 896
    assert pretrained_model.s1_projection.out_dim == llm_dim
    assert pretrained_model.s2_projection.out_dim == llm_dim
    assert pretrained_model.fusion.hidden_dim == llm_dim


def test_tokenizer_model_compatibility(tokenizer, pretrained_model):
    """Verify tokenizer encodes prompts compatible with input embeddings."""
    prompt = "Hello satellite vision"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    assert input_ids.max().item() < pretrained_model.config.vocab_size

    embeds = pretrained_model.language_model.get_input_embeddings()(input_ids)
    assert embeds.shape == (1, input_ids.shape[1], 896)


def test_base_language_parameters_frozen(pretrained_model):
    """Verify all base LLM parameters have requires_grad == False."""
    frozen_count = 0
    trainable_count = 0
    for p in pretrained_model.language_model.parameters():
        if p.requires_grad:
            trainable_count += 1
        else:
            frozen_count += 1

    assert trainable_count == 0, "Base LLM parameters must be completely frozen"
    assert frozen_count > 0, "Expected non-zero frozen parameters in language model"


def test_lora_parameters_trainable(pretrained_model):
    """Verify applying LoRA enables gradients strictly on adapters while keeping backbone frozen."""
    adapted_model, audit = apply_lora(pretrained_model, r=8, lora_alpha=32, lora_dropout=0.1)
    assert audit["lora_trainable"] == 540672
    assert audit["frozen_llm"] == 629697920
    assert audit["trainable_percentage"] > 0.0


def test_s1_s2_encoder_dimensions(pretrained_model):
    """Verify S1 and S2 encoders produce expected token sequences and dimensions."""
    s1_in = torch.randn(1, 2, 120, 120)
    s2_in = torch.randn(1, 10, 120, 120)

    s1_feat, s2_feat, s1_tok, s2_tok = pretrained_model.encode_vision(s1_in, s2_in)
    assert s1_tok.shape == (1, 225, 896)
    assert s2_tok.shape == (1, 225, 896)


def test_text_only_coherent_generation(tokenizer, pretrained_model):
    """Verify deterministic fluent English generation on a simple factual prompt."""
    prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]

    with torch.no_grad():
        outputs = pretrained_model.language_model.generate(
            input_ids=input_ids,
            max_new_tokens=16,
            do_sample=False,
        )
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert "Paris" in gen_text, f"Expected 'Paris' in generated text, got: {gen_text!r}"
