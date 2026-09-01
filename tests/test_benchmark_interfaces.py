"""
Tests for VRSBench and RSVQA evaluation interfaces in evaluation/vlm/.
Verifies:
- Import integrity
- Sample evaluation contract
- Schema matching
- Metric computation logic
"""

import pytest
import torch
from evaluation.vlm.vrsbench_interface import VRSBenchEvaluationInterface
from evaluation.vlm.rsvqa_interface import RSVQAEvaluationInterface
from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.train_lora import FallbackTokenizer


@pytest.fixture(scope="module")
def toy_eval_model():
    torch.manual_seed(0)
    cfg = RSInternVLConfig(
        model_id="OpenGVLab/InternVL3-1B",
        pretrained_backbone=False,
        img_size=16,
        s1_channels=2, s1_hidden_dim=64,
        s2_channels=10, s2_hidden_dim=64,
        projection_hidden_dim=64,
        freeze_llm=True,
    )
    model = RSInternVL(cfg)
    model.eval()
    return model


def test_vrsbench_interface_contract(toy_eval_model):
    tokenizer = FallbackTokenizer(vocab_size=1000)
    adapter = VRSBenchEvaluationInterface(toy_eval_model, tokenizer)

    sample = {
        "question_id": "vrs_test_01",
        "image_s1": torch.randn(2, 16, 16),
        "image_s2": torch.randn(10, 16, 16),
        "question": "Is water present?",
        "ground_truth_answer": "Yes",
    }
    pred = adapter.evaluate_sample(sample)

    assert pred["question_id"] == "vrs_test_01"
    assert isinstance(pred["answer"], str)
    assert isinstance(pred["is_correct"], bool)
    assert "model_score" in pred
    assert pred["model_version"] == "RS-InternVL-Step10"

    metrics = adapter.compute_metrics([pred])
    assert "accuracy" in metrics
    assert metrics["total_samples"] == 1


def test_rsvqa_interface_contract(toy_eval_model):
    tokenizer = FallbackTokenizer(vocab_size=1000)
    adapter = RSVQAEvaluationInterface(toy_eval_model, tokenizer)

    sample = {
        "question_id": 101,
        "image_s1": torch.randn(2, 16, 16),
        "image_s2": torch.randn(10, 16, 16),
        "question": "Are there buildings?",
        "answers": ["yes", "yes, buildings exist"],
    }
    pred = adapter.evaluate_sample(sample)

    assert pred["question_id"] == "101"
    assert isinstance(pred["answer"], str)
    assert isinstance(pred["is_correct"], bool)
    assert "model_score" in pred

    metrics = adapter.compute_metrics([pred])
    assert "top1_accuracy" in metrics
    assert metrics["total_samples"] == 1
