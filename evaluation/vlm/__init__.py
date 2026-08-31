"""
VLM Evaluation Interfaces Module for Remote Sensing Benchmarks.

Provides standard evaluation interfaces and metric entry points for:
- BigEarthNet multimodal VQA
- VRSBench (Visual Reasoning on Remote Sensing)
- RSVQA (Remote Sensing Visual Question Answering)
"""

from evaluation.vlm.vrsbench_interface import VRSBenchEvaluationInterface
from evaluation.vlm.rsvqa_interface import RSVQAEvaluationInterface

__all__ = [
    "VRSBenchEvaluationInterface",
    "RSVQAEvaluationInterface",
]
