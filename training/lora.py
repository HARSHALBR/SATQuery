"""
RS-InternVL: PEFT / LoRA Fine-Tuning Module (Step 4).

Implements parameter-efficient fine-tuning for RS-InternVL using Hugging Face PEFT.
Freezes the pretrained language model backbone while training LoRA adapters,
modality projections, and task-specific S1/S2 modality encoders.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL

logger = logging.getLogger("rs_internvl.lora")

# Standard candidate attention and projection module names for Qwen2 / InternVL
DEFAULT_LORA_TARGET_CANDIDATES: List[str] = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def find_target_modules(
    model: nn.Module,
    candidate_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Inspect the actual module hierarchy of the language model backbone to find
    matched target module names without blind hardcoding.

    Args:
        model: Language model or root module (e.g. Qwen2ForCausalLM).
        candidate_names: List of module name substrings to match against.

    Returns:
        Sorted list of matched unique target module name suffixes.

    Raises:
        ValueError: If zero target modules are matched.
    """
    candidates = candidate_names or ["q_proj", "v_proj"]

    # Traverse named modules to find existing Linear projection layer names
    matched_names: Set[str] = set()
    all_module_names: List[str] = []

    for name, module in model.named_modules():
        all_module_names.append(name)
        for cand in candidates:
            # Check if candidate matches the terminal submodule name (e.g. 'self_attn.q_proj' -> 'q_proj')
            if name.endswith(f".{cand}") or name == cand or cand in name.split("."):
                matched_names.add(cand)

    matched_list = sorted(list(matched_names))

    if not matched_list:
        raise ValueError(
            f"Zero target modules matched candidate list {candidates}! "
            f"Available submodules in model: {all_module_names[:20]}... (total {len(all_module_names)})"
        )

    logger.info(
        f"LoRA Target Module Match: Found {len(matched_list)} module types matching {candidates}: {matched_list}"
    )
    return matched_list


def build_lora_config(
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """
    Construct a validated Hugging Face LoraConfig.

    Args:
        r: LoRA rank dimension.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout probability.
        bias: Bias training mode ("none", "all", "lora_only").
        target_modules: List of target module names (e.g. ["q_proj", "v_proj"]).
        task_type: Task type for PEFT (default TaskType.CAUSAL_LM).

    Returns:
        LoraConfig instance.
    """
    target = target_modules or ["q_proj", "v_proj"]
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target,
        task_type=task_type,
    )


def audit_parameters(model: RSInternVL) -> Dict[str, Any]:
    """
    Perform a complete parameter audit by component across the adapted model.

    Returns:
        Dictionary containing parameter counts and trainability percentages.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Component parameter breakdowns
    s1_total = sum(p.numel() for p in model.s1_encoder.parameters())
    s1_trainable = sum(p.numel() for p in model.s1_encoder.parameters() if p.requires_grad)

    s2_total = sum(p.numel() for p in model.s2_encoder.parameters())
    s2_trainable = sum(p.numel() for p in model.s2_encoder.parameters() if p.requires_grad)

    proj_total = sum(p.numel() for p in model.s1_projection.parameters()) + sum(
        p.numel() for p in model.s2_projection.parameters()
    )
    proj_trainable = sum(
        p.numel() for p in model.s1_projection.parameters() if p.requires_grad
    ) + sum(p.numel() for p in model.s2_projection.parameters() if p.requires_grad)

    llm_total = sum(p.numel() for p in model.language_model.parameters())
    llm_trainable = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
    frozen_llm = llm_total - llm_trainable

    pct_trainable = (trainable_params / max(1, total_params)) * 100.0

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_percentage": round(pct_trainable, 4),
        "s1_encoder_total": s1_total,
        "s1_encoder_trainable": s1_trainable,
        "s2_encoder_total": s2_total,
        "s2_encoder_trainable": s2_trainable,
        "projections_total": proj_total,
        "projections_trainable": proj_trainable,
        "lora_trainable": llm_trainable,
        "frozen_llm": frozen_llm,
    }


def print_parameter_audit(audit: Dict[str, Any], title: str = "RS-INTERNVL PARAMETER AUDIT") -> None:
    """Print formatted parameter breakdown to logger and stdout."""
    sep = "=" * 65
    logger.info(sep)
    logger.info(f"            {title}            ")
    logger.info(sep)
    logger.info(f"Total Parameters:             {audit['total']:,}")
    logger.info(f"Trainable Parameters:         {audit['trainable']:,} ({audit['trainable_percentage']}%)")
    logger.info(f"Frozen Parameters:            {audit['frozen']:,}")
    logger.info(f"  - S1 SAR Encoder:           {audit['s1_encoder_trainable']:,} / {audit['s1_encoder_total']:,} trainable")
    logger.info(f"  - S2 Optical Encoder:       {audit['s2_encoder_trainable']:,} / {audit['s2_encoder_total']:,} trainable")
    logger.info(f"  - Modality Projections:     {audit['projections_trainable']:,} / {audit['projections_total']:,} trainable")
    logger.info(f"  - LoRA Adapters (in LLM):   {audit['lora_trainable']:,} trainable")
    logger.info(f"  - Base LLM Backbone:        {audit['frozen_llm']:,} frozen")
    logger.info(sep)


def apply_lora(
    model: RSInternVL,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: Optional[List[str]] = None,
    freeze_s1_encoder: bool = False,
    freeze_s2_encoder: bool = False,
    lora_config: Optional[LoraConfig] = None,
) -> Tuple[RSInternVL, Dict[str, Any]]:
    """
    Apply Hugging Face PEFT / LoRA to the RS-InternVL architecture.

    Freezing Policy:
    1. Base InternVL / Qwen2 Language Model: FROZEN.
    2. LoRA Adapters inside Language Model: TRAINABLE.
    3. S1 Projection & S2 Projection: TRAINABLE.
    4. S1 Encoder & S2 Encoder: Configurable (default TRAINABLE because our
       task-specific encoders are randomly initialized, not pretrained).

    Args:
        model: Base RSInternVL model.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout probability.
        bias: Bias configuration ("none").
        target_modules: Optional specific target module names. If None,
                        dynamically inspects language model for 'q_proj' and 'v_proj'.
        freeze_s1_encoder: If True, freezes S1 encoder weights.
        freeze_s2_encoder: If True, freezes S2 encoder weights.
        lora_config: Optional pre-constructed LoraConfig.

    Returns:
        Tuple of (adapted_model, parameter_audit_dict).
    """
    # 1. Inspect language model for actual target modules
    target_candidates = target_modules or ["q_proj", "v_proj"]
    matched_targets = find_target_modules(model.language_model, target_candidates)

    # 2. Build LoraConfig if not supplied
    if lora_config is None:
        lora_config = build_lora_config(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=matched_targets,
        )

    # 3. Apply PEFT to the underlying language model
    logger.info(
        f"Applying LoRA (r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}) to language model..."
    )
    model.language_model = get_peft_model(model.language_model, lora_config)

    # 4. Configure S1/S2 Modality Encoders
    # Note: Encoders are randomly initialized task-specific modules, so they are trainable by default.
    if freeze_s1_encoder:
        model.s1_encoder.freeze()
        logger.info("S1 SAR Encoder: FROZEN")
    else:
        model.s1_encoder.unfreeze()
        logger.info("S1 SAR Encoder: TRAINABLE (randomly initialized task encoder)")

    if freeze_s2_encoder:
        model.s2_encoder.freeze()
        logger.info("S2 Optical Encoder: FROZEN")
    else:
        model.s2_encoder.unfreeze()
        logger.info("S2 Optical Encoder: TRAINABLE (randomly initialized task encoder)")

    # 5. Ensure Modality Projections are TRAINABLE
    for p in model.s1_projection.parameters():
        p.requires_grad = True
    for p in model.s2_projection.parameters():
        p.requires_grad = True
    for p in model.fusion.parameters():
        p.requires_grad = True

    # 6. Audit parameter trainability
    audit = audit_parameters(model)
    print_parameter_audit(audit, title="RS-INTERNVL LoRA ADAPTATION AUDIT")

    return model, audit


def save_lora_checkpoint(
    model: RSInternVL,
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save modular LoRA checkpoint without saving redundant copies of the frozen base LLM.

    Saved structure:
        output_dir/
            adapter/                  # PEFT LoRA adapter weights & adapter_config.json
            modality_encoders.pt      # S1 and S2 encoder weights
            modality_projections.pt   # S1 and S2 projection + fusion weights
            training_state.pt         # Optimizer, scheduler, step, and epoch state
            config.yaml               # Training configuration
            metrics.json              # Current metrics snapshot

    Args:
        model: Adapted RSInternVL model with PeftModel language_model.
        output_dir: Destination directory.
        config: Training configuration dictionary.
        optimizer: Optional optimizer for saving training state.
        scheduler: Optional learning rate scheduler.
        epoch: Current epoch number.
        global_step: Current global optimization step.
        metrics: Optional dictionary of evaluation/training metrics.

    Returns:
        Path to the saved checkpoint directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    adapter_dir = out_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save LoRA Adapter weights & config
    if isinstance(model.language_model, PeftModel) or hasattr(model.language_model, "save_pretrained"):
        model.language_model.save_pretrained(adapter_dir)
        logger.info(f"Saved LoRA adapter -> {adapter_dir}")
    else:
        # Fallback if language_model has adapter state dict
        adapter_state = {
            k: v for k, v in model.language_model.state_dict().items() if "lora" in k.lower()
        }
        torch.save(adapter_state, adapter_dir / "adapter_model.bin")
        logger.info(f"Saved LoRA weights state dict -> {adapter_dir / 'adapter_model.bin'}")

    def _safe_torch_save(obj: Any, path: Path):
        """Robust atomic PyTorch serialization preventing Windows zip container stream errors."""
        tmp_path = path.with_suffix(path.suffix + f".tmp_{os.getpid()}_{int(time.time()*1000)%10000}")
        try:
            try:
                torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
            except Exception:
                torch.save(obj, tmp_path)
            if tmp_path.exists():
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
                os.replace(str(tmp_path), str(path))
        except Exception as e:
            logger.warning(f"Atomic save fallback for {path}: {e}")
            try:
                torch.save(obj, path)
            except Exception as final_e:
                logger.error(f"Could not save {path}: {final_e}")
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    # 2. Save Modality Encoders
    encoders_path = out_path / "modality_encoders.pt"
    _safe_torch_save(
        {
            "s1_encoder": model.s1_encoder.state_dict(),
            "s2_encoder": model.s2_encoder.state_dict(),
        },
        encoders_path,
    )
    logger.info(f"Saved modality encoders -> {encoders_path}")

    # 3. Save Modality Projections
    projections_path = out_path / "modality_projections.pt"
    _safe_torch_save(
        {
            "s1_projection": model.s1_projection.state_dict(),
            "s2_projection": model.s2_projection.state_dict(),
            "fusion": model.fusion.state_dict(),
        },
        projections_path,
    )
    logger.info(f"Saved modality projections -> {projections_path}")

    # 4. Save Training State (Optimizer & Scheduler)
    training_state_path = out_path / "training_state.pt"
    state_dict_payload: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        state_dict_payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state_dict_payload["scheduler_state_dict"] = scheduler.state_dict()

    _safe_torch_save(state_dict_payload, training_state_path)
    logger.info(f"Saved training state -> {training_state_path}")

    # 5. Save Configuration
    if config is not None:
        config_path = out_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved active config -> {config_path}")

    # 6. Save Metrics
    if metrics is not None:
        metrics_path = out_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics snapshot -> {metrics_path}")

    return out_path


def load_lora_checkpoint(
    checkpoint_dir: Union[str, Path],
    config_override: Optional[RSInternVLConfig] = None,
    device: Union[str, torch.device] = "cpu",
    is_trainable: bool = False,
    freeze_s1_encoder: bool = False,
    freeze_s2_encoder: bool = False,
) -> RSInternVL:
    """
    Reconstruct the complete RS-InternVL model from a modular LoRA checkpoint:
    Base InternVL3-1B + S1/S2 Encoders + S1/S2 Projections + LoRA Adapter.

    Args:
        checkpoint_dir: Path to the modular checkpoint directory.
        config_override: Optional RSInternVLConfig instance.
        device: Device to place the reconstructed model on.
        is_trainable: Whether the reloaded LoRA adapter should remain trainable.

    Returns:
        Fully reconstructed RSInternVL model ready for inference or continued training.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_path}")

    # 1. Resolve configuration
    config = config_override
    config_yaml_path = ckpt_path / "config.yaml"
    if config is None and config_yaml_path.exists():
        with open(config_yaml_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        model_cfg = cfg_dict.get("model", {})
        config = RSInternVLConfig(
            model_id=model_cfg.get("backbone", "OpenGVLab/InternVL3-1B"),
            img_size=model_cfg.get("img_size", 120),
            num_hidden_layers=model_cfg.get("num_hidden_layers", 24),
            num_attention_heads=model_cfg.get("num_attention_heads", 14),
            num_key_value_heads=model_cfg.get("num_key_value_heads", 2),
            intermediate_size=model_cfg.get("intermediate_size", 4864),
            llm_hidden_dim=model_cfg.get("llm_hidden_dim", 896),
            vocab_size=model_cfg.get("vocab_size", 151674),
        )

    if config is None:
        config = RSInternVLConfig()

    if "semantic_overfit" in str(ckpt_path) or "checkpoints/lora" in str(ckpt_path).replace("\\", "/"):
        logger.warning(
            f"Checkpoint at {ckpt_path} originated from earlier Step 4-6 experiments. "
            f"Note: Adapters trained on random language backbones should not be confused with pretrained-backbone checkpoints."
        )

    logger.info(f"Instantiating RSInternVL model (pretrained_backbone={config.pretrained_backbone}) on device={device}...")
    model = RSInternVL(config)

    # 2. Load Modality Encoders
    encoders_path = ckpt_path / "modality_encoders.pt"
    if encoders_path.exists():
        enc_payload = torch.load(encoders_path, map_location=device, weights_only=False)
        if "s1_encoder" in enc_payload:
            model.s1_encoder.load_state_dict(enc_payload["s1_encoder"])
        if "s2_encoder" in enc_payload:
            model.s2_encoder.load_state_dict(enc_payload["s2_encoder"])
        logger.info(f"Loaded modality encoders from {encoders_path}")
    else:
        logger.warning(f"modality_encoders.pt not found in {ckpt_path}; using initialized weights.")

    # 3. Load Modality Projections
    projections_path = ckpt_path / "modality_projections.pt"
    if projections_path.exists():
        proj_payload = torch.load(projections_path, map_location=device, weights_only=False)
        if "s1_projection" in proj_payload:
            model.s1_projection.load_state_dict(proj_payload["s1_projection"])
        if "s2_projection" in proj_payload:
            model.s2_projection.load_state_dict(proj_payload["s2_projection"])
        if "fusion" in proj_payload:
            model.fusion.load_state_dict(proj_payload["fusion"])
        logger.info(f"Loaded modality projections from {projections_path}")
    else:
        logger.warning(f"modality_projections.pt not found in {ckpt_path}; using initialized weights.")

    # 4. Load LoRA Adapter onto Language Model
    adapter_dir = ckpt_path / "adapter"
    if adapter_dir.exists():
        logger.info(f"Loading LoRA adapter from {adapter_dir}...")
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            str(adapter_dir),
            is_trainable=is_trainable,
        )
        logger.info("Successfully reloaded LoRA adapter into language model backbone.")
    else:
        logger.warning(f"adapter/ folder not found in {ckpt_path}.")

    if is_trainable:
        if freeze_s1_encoder:
            model.s1_encoder.freeze()
        else:
            model.s1_encoder.unfreeze()

        if freeze_s2_encoder:
            model.s2_encoder.freeze()
        else:
            model.s2_encoder.unfreeze()

        for p in model.s1_projection.parameters():
            p.requires_grad = True
        for p in model.s2_projection.parameters():
            p.requires_grad = True
        for p in model.fusion.parameters():
            p.requires_grad = True

    model.to(device)
    return model
