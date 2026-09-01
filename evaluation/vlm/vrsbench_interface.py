"""
VRSBench Evaluation Interface Stub for RS-InternVL.

VRSBench: Visual Reasoning on Earth Observation imagery.
This module defines the standard benchmark evaluation contract, input formatting,
and evaluation metric computation without requiring downloading the entire external dataset.
"""

from typing import Any, Dict, List, Optional
import torch


class VRSBenchEvaluationInterface:
    """
    Evaluation adapter for VRSBench (Visual Reasoning on Remote Sensing).
    
    Contract:
    - Input sample format:
      {
          "question_id": str,
          "image_s1": torch.Tensor [2, H, W] or None,
          "image_s2": torch.Tensor [10, H, W] or [3, H, W],
          "question": str,
          "ground_truth_answer": str,
          "task_type": "presence" | "comparison" | "counting" | "reasoning"
      }
    - Output prediction format:
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
        """
        Evaluate a single VRSBench sample through RS-InternVL.
        
        Args:
            sample: Dictionary containing imagery, question, and ground truth.
            
        Returns:
            Structured prediction dictionary matching the benchmark contract.
        """
        image_s1 = sample.get("image_s1")
        image_s2 = sample.get("image_s2")
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth_answer", "").strip().lower()

        if image_s1 is None:
            # S1 placeholder if optical-only sample
            H = image_s2.shape[-2] if image_s2 is not None else 120
            W = image_s2.shape[-1] if image_s2 is not None else 120
            image_s1 = torch.zeros(2, H, W, dtype=torch.float32)

        pred = self.model.predict(
            image_s1=image_s1,
            image_s2=image_s2,
            query=question,
            tokenizer=self.tokenizer,
            max_new_tokens=32,
        )

        pred_answer = pred.get("answer", "").strip().lower()
        is_correct = (ground_truth in pred_answer) if ground_truth else False

        return {
            "question_id": sample.get("question_id", "unknown"),
            "question": question,
            "ground_truth_answer": sample.get("ground_truth_answer", ""),
            "answer": pred.get("answer", ""),
            "claim_type": pred.get("claim_type", "visual_reasoning"),
            "model_score": pred.get("model_score", 0.0),
            "model_version": self.model_version,
            "is_correct": is_correct,
        }

    def compute_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute aggregate accuracy and validity metrics across a prediction set."""
        if not predictions:
            return {"accuracy": 0.0, "total_samples": 0}

        correct = sum(1 for p in predictions if p.get("is_correct", False))
        total = len(predictions)
        return {
            "accuracy": round(correct / total * 100.0, 2),
            "total_samples": total,
            "correct_samples": correct,
        }
