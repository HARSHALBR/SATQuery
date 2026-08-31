"""
Tests for Member 1 Structured Output Interface (RSInternVL.predict).
Verifies:
- All required fields exist (answer, claim_type, model_score, model_version)
- answer is string
- claim_type is string and non-empty
- model_score is numeric (float/int)
- model_version is string
- output is JSON serializable
"""

import json
import pytest
import torch
from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.train_lora import FallbackTokenizer


@pytest.fixture(scope="module")
def toy_predict_model():
    torch.manual_seed(0)
    cfg = RSInternVLConfig(
        model_id="OpenGVLab/InternVL3-1B",
        pretrained_backbone=False,
        img_size=16,
        s1_channels=2,
        s1_hidden_dim=64,
        s2_channels=10,
        s2_hidden_dim=64,
        projection_hidden_dim=64,
        freeze_llm=True,
    )
    model = RSInternVL(cfg)
    model.eval()
    return model


def test_structured_output_schema_and_types(toy_predict_model):
    tokenizer = FallbackTokenizer(vocab_size=1000)
    s1 = torch.randn(2, 16, 16)
    s2 = torch.randn(10, 16, 16)
    query = "Is broad-leaved forest present in this area?"

    result = toy_predict_model.predict(
        image_s1=s1,
        image_s2=s2,
        query=query,
        tokenizer=tokenizer,
        max_new_tokens=8,
    )

    # 1. Required keys presence
    assert "answer" in result, "Missing 'answer' field"
    assert "claim_type" in result, "Missing 'claim_type' field"
    assert "model_score" in result, "Missing 'model_score' field"
    assert "model_version" in result, "Missing 'model_version' field"

    # 2. Type assertions
    assert isinstance(result["answer"], str), f"answer must be str, got {type(result['answer'])}"
    assert isinstance(result["claim_type"], str), f"claim_type must be str, got {type(result['claim_type'])}"
    assert isinstance(result["model_score"], (int, float)), f"model_score must be numeric, got {type(result['model_score'])}"
    assert isinstance(result["model_version"], str), f"model_version must be str, got {type(result['model_version'])}"

    # 3. Value constraints
    assert len(result["claim_type"]) > 0, "claim_type cannot be empty"
    assert 0.0 <= result["model_score"] <= 1.0, f"model_score must be in [0, 1], got {result['model_score']}"

    # 4. JSON serializability
    json_str = json.dumps(result)
    assert len(json_str) > 0, "Failed JSON serialization"
    deserialized = json.loads(json_str)
    assert deserialized["answer"] == result["answer"]
    assert deserialized["claim_type"] == result["claim_type"]
    assert deserialized["model_score"] == result["model_score"]
    assert deserialized["model_version"] == result["model_version"]


def test_structured_output_claim_type_inference(toy_predict_model):
    tokenizer = FallbackTokenizer(vocab_size=1000)
    s1 = torch.randn(2, 16, 16)
    s2 = torch.randn(10, 16, 16)

    # Presence query
    res_bin = toy_predict_model.predict(
        image_s1=s1, image_s2=s2,
        query="Is water body present?", tokenizer=tokenizer, max_new_tokens=4,
    )
    assert res_bin["claim_type"] == "presence_verification"

    # Classification query
    res_mcq = toy_predict_model.predict(
        image_s1=s1, image_s2=s2,
        query="What is the dominant land cover class?", tokenizer=tokenizer, max_new_tokens=4,
    )
    assert res_mcq["claim_type"] == "land_cover_classification"

    # General VQA query
    res_vqa = toy_predict_model.predict(
        image_s1=s1, image_s2=s2,
        query="Describe the terrain in this tile.", tokenizer=tokenizer, max_new_tokens=4,
    )
    assert res_vqa["claim_type"] == "visual_question_answering"
