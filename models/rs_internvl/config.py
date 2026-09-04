"""
Configuration class for RS-InternVL multi-modal architecture.

Dynamically loads hidden dimensions, vocabulary sizes, and architectural parameters
from the authentic InternVL3-1B configuration.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Import MODEL_S2_BANDS from the single authoritative source in the data package.
# This ensures the model config and dataset always agree on the 10-band selection.
try:
    from data.bigearthnet_txt.constants import MODEL_S2_BANDS as _MODEL_S2_BANDS
except ImportError:
    # Fallback if the data package isn't on the path (e.g. isolated model tests)
    _MODEL_S2_BANDS = [
        "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    ]

DEFAULT_S2_BANDS: List[str] = _MODEL_S2_BANDS
DEFAULT_S1_BANDS: List[str] = ["VV", "VH"]


@dataclass
class RSInternVLConfig:
    """
    Configuration for RS-InternVL multimodal remote sensing model.
    """

    # Model Identification & Checkpoint Reference
    model_id: str = "OpenGVLab/InternVL3-1B"
    config_file: Optional[str] = None
    pretrained_backbone: bool = True
    pretrained_model_path: Optional[str] = None

    # Multi-sensor Spectral & Spatial Parameters
    s1_bands: List[str] = field(default_factory=lambda: list(DEFAULT_S1_BANDS))
    s2_bands: List[str] = field(default_factory=lambda: list(DEFAULT_S2_BANDS))
    img_size: int = 120

    # Modality Encoder Dimensions
    s1_channels: int = 2
    s1_hidden_dim: int = 512
    s2_channels: int = 10
    s2_hidden_dim: int = 768

    # Dynamic LLM Dimensions (Loaded from InternVL3-1B checkpoint/config)
    llm_hidden_dim: int = 896
    vocab_size: int = 151674
    max_position_embeddings: int = 32768
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    pad_token_id: Optional[int] = None
    bos_token_id: int = 151643
    eos_token_id: int = 151643

    # Projection Layer Parameters
    projection_hidden_dim: int = 1024
    projection_dropout: float = 0.0

    # Freezing Policies (Configurable for Step 2 vs downstream fine-tuning)
    freeze_s1_encoder: bool = True
    freeze_s2_encoder: bool = True
    freeze_llm: bool = True

    # LoRA / PEFT Placeholders (for Step 4 adaptation)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # General Model Settings
    torch_dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self):
        # Sync channel counts with specified band lists
        self.s1_channels = len(self.s1_bands)
        self.s2_channels = len(self.s2_bands)

        # Attempt to dynamically load actual dimensions from InternVL config file if present
        self._load_from_internvl_config()

        # Ensure special token IDs are within vocabulary bounds
        if self.bos_token_id >= self.vocab_size:
            self.bos_token_id = max(0, min(1, self.vocab_size - 1))
        if self.eos_token_id >= self.vocab_size:
            self.eos_token_id = max(0, min(2, self.vocab_size - 1))

    def _load_from_internvl_config(self) -> None:
        """Dynamically load exact LLM hidden dimension and vocab size from InternVL config."""
        if not self.config_file:
            return

        cfg_path = Path(self.config_file)
        if not cfg_path.exists():
            # Check relative to repository root
            repo_root = Path(__file__).resolve().parent.parent.parent
            cand = repo_root / self.config_file
            if cand.exists():
                cfg_path = cand

        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw_cfg = json.load(f)

                llm_cfg = raw_cfg.get("llm_config", {})
                if llm_cfg:
                    self.llm_hidden_dim = llm_cfg.get("hidden_size", self.llm_hidden_dim)
                    self.vocab_size = llm_cfg.get("vocab_size", self.vocab_size)
                    self.max_position_embeddings = llm_cfg.get("max_position_embeddings", self.max_position_embeddings)
                    self.num_hidden_layers = llm_cfg.get("num_hidden_layers", self.num_hidden_layers)
                    self.num_attention_heads = llm_cfg.get("num_attention_heads", self.num_attention_heads)
                    self.num_key_value_heads = llm_cfg.get("num_key_value_heads", self.num_key_value_heads)
                    self.intermediate_size = llm_cfg.get("intermediate_size", self.intermediate_size)
                    self.bos_token_id = llm_cfg.get("bos_token_id", self.bos_token_id)
                    self.eos_token_id = llm_cfg.get("eos_token_id", self.eos_token_id)
                    logger.info(
                        f"Loaded dynamic parameters from {cfg_path.name}: "
                        f"llm_hidden_dim={self.llm_hidden_dim}, vocab_size={self.vocab_size}"
                    )
            except Exception as e:
                logger.warning(f"Could not parse local config {cfg_path}: {e}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "RSInternVLConfig":
        """
        Create RSInternVLConfig from a pretrained model identifier or path.
        """
        config = cls(model_id=pretrained_model_name_or_path, **kwargs)
        return config

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
