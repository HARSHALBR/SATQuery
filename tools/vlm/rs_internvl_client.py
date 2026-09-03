"""RS-InternVL client adapter for SATQuery AI."""

from pathlib import Path
from typing import List, Optional

import torch

from schemas.vlm import VLMClaimType, VLMContext, VLMResult
from tools.vlm.client import VLMClient


class RSInternVLClient(VLMClient):
    """
    Adapter around the trained RS-InternVL model.

    The model is loaded lazily on the first inference request.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cpu",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)

        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load RS-InternVL and tokenizer only when first needed."""

        if self._model is not None:
            return

        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"RS-InternVL checkpoint not found: {self.checkpoint_dir}"
            )

        from training.lora import load_lora_checkpoint
        from training.tokenizer import get_tokenizer

        self._model = load_lora_checkpoint(
            self.checkpoint_dir,
            device=self.device,
            is_trainable=False,
        )

        self._model.eval()

        self._tokenizer = get_tokenizer()

    def analyze(
        self,
        image_paths: List[str],
        query: str,
        context: VLMContext,
    ) -> VLMResult:
        """
        Compatibility implementation for the existing VLMClient interface.

        Actual S1/S2 tensor inference is exposed through analyze_tensors().
        """

        raise NotImplementedError(
            "RS-InternVL requires S1/S2 tensors. "
            "Use analyze_tensors() after preparing the satellite bands."
        )

    @torch.no_grad()
    def analyze_tensors(
        self,
        image_s1: torch.Tensor,
        image_s2: torch.Tensor,
        query: str,
        context: Optional[VLMContext] = None,
    ) -> VLMResult:
        """Run RS-InternVL directly on prepared S1/S2 tensors."""

        self._validate_inputs(image_s1, image_s2)

        self._load_model()

        image_s1 = image_s1.to(self.device)
        image_s2 = image_s2.to(self.device)

        prediction = self._model.predict(
            image_s1=image_s1,
            image_s2=image_s2,
            query=query,
            tokenizer=self._tokenizer,
            max_new_tokens=64,
            do_sample=False,
        )

        return self._to_vlm_result(prediction)

    @torch.no_grad()
    def analyze_observation(
        self,
        image_s1: torch.Tensor,
        image_s2: torch.Tensor,
        query: str,
        context: Optional[VLMContext] = None,
    ) -> VLMResult:
        """
        Run RS-InternVL on one satellite observation.

        This is semantic scene interpretation only. It does not perform
        temporal comparison between T1 and T2.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        return self.analyze_tensors(
            image_s1=image_s1,
            image_s2=image_s2,
            query=query,
            context=context,
        )

    @staticmethod
    def _validate_inputs(
        image_s1: torch.Tensor,
        image_s2: torch.Tensor,
    ) -> None:
        """Validate tensor shapes before invoking the model."""

        if not isinstance(image_s1, torch.Tensor):
            raise TypeError("image_s1 must be a torch.Tensor")

        if not isinstance(image_s2, torch.Tensor):
            raise TypeError("image_s2 must be a torch.Tensor")

        if image_s1.ndim != 3:
            raise ValueError(
                f"image_s1 must have shape [2,H,W], got {tuple(image_s1.shape)}"
            )

        if image_s2.ndim != 3:
            raise ValueError(
                f"image_s2 must have shape [10,H,W], got {tuple(image_s2.shape)}"
            )

        if image_s1.shape[0] != 2:
            raise ValueError(
                f"RS-InternVL requires 2 S1 channels, got {image_s1.shape[0]}"
            )

        if image_s2.shape[0] != 10:
            raise ValueError(
                f"RS-InternVL requires 10 S2 channels, got {image_s2.shape[0]}"
            )

    @staticmethod
    def _to_vlm_result(prediction: dict) -> VLMResult:
        """Convert RS-InternVL structured output into SATQuery VLMResult."""

        claim_value = prediction.get("claim_type") or prediction.get("claim")

        if not claim_value:
            raise ValueError("RS-InternVL prediction does not contain a claim")

        try:
            claim = VLMClaimType(claim_value)
        except ValueError:
            claim = VLMClaimType.GENERAL_CHANGE

        confidence = float(
            prediction.get(
                "model_score",
                prediction.get("confidence", 0.0),
            )
        )

        reasoning = (
            prediction.get("answer")
            or prediction.get("reasoning")
            or "RS-InternVL returned no textual explanation."
        )

        return VLMResult(
            claim=claim,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=str(reasoning),
        )
