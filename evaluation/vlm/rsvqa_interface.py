"""
RSVQA Evaluation Interface Stub for RS-InternVL.

RSVQA: Remote Sensing Visual Question Answering (LR / HR / SAR).
Defines standard dataset loading schema, model call contract, and metric calculation.
"""

from typing import Any, Dict, List, Optional
import torch


class RSVQAEvaluationInterface:
    """
    Evaluation adapter for RSVQA benchmark.
    
    Contract:
    - Input sample:
      {
          "question_id": int | str,
          "image_s1": torch.Tensor [2, H, W],
          "image_s2": torch.Tensor [10, H, W],
          "question": str,
          "answers": List[str],  # possible accepted answers
          "category": "presence" | "count" | "comparison" | "rural_urban"
      }
    - Output:
      {
          "question_id": str,
          "answer": str,
          "claim_type": str,
          "model_score": float,
          "model_version": str,
          "is_correct": bool
      }
    """

    def __init__(self, model: Any, tokenizer: Any, model_version: str = "RS-InternVL-Step10"):
        self.model = model
        self.tokenizer = tokenizer
        self.model_version = model_version

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single RSVQA query through RS-InternVL."""
        image_s1 = sample.get("image_s1")
        image_s2 = sample.get("image_s2")
        question = sample.get("question", "")
        ground_truth_answers = [a.strip().lower() for a in sample.get("answers", [])]

        pred = self.model.predict(
            image_s1=image_s1,
            image_s2=image_s2,
            query=question,
            tokenizer=self.tokenizer,
            max_new_tokens=16,
        )

        pred_answer = pred.get("answer", "").strip().lower()
        is_correct = any(gt in pred_answer for gt in ground_truth_answers) if ground_truth_answers else False

        return {
            "question_id": str(sample.get("question_id", "0")),
            "question": question,
            "target_answers": sample.get("answers", []),
            "answer": pred.get("answer", ""),
            "claim_type": pred.get("claim_type", "rsvqa_query"),
            "model_score": pred.get("model_score", 0.0),
            "model_version": self.model_version,
            "is_correct": is_correct,
        }

    def compute_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute top-1 accuracy across evaluated RSVQA predictions."""
        if not predictions:
            return {"top1_accuracy": 0.0, "total_samples": 0}

        correct = sum(1 for p in predictions if p.get("is_correct", False))
        total = len(predictions)
        return {
            "top1_accuracy": round(correct / total * 100.0, 2),
            "total_samples": total,
            "correct_samples": correct,
        }
