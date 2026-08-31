#!/usr/bin/env python3
"""
RS-InternVL: Production PEFT / LoRA Fine-Tuning Pipeline (Step 4).

Fine-tunes the RS-InternVL architecture on BigEarthNet.txt with LoRA adapters
on the InternVL3-1B language model backbone while keeping base LLM weights frozen
and training modality projections and S1/S2 modality encoders.

Usage:
    python training/train_lora.py --config configs/model/lora.yaml
    python training/train_lora.py --config configs/model/lora.yaml --max-train-samples 500 --max-val-samples 100 --epochs 3
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from data.bigearthnet_txt.dataset import BigEarthNetDataset
from models.rs_internvl.config import (
    DEFAULT_S1_BANDS,
    DEFAULT_S2_BANDS,
    RSInternVLConfig,
)
from models.rs_internvl.model import RSInternVL
from training.lora import (
    apply_lora,
    audit_parameters,
    load_lora_checkpoint,
    print_parameter_audit,
    save_lora_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_lora")


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FallbackTokenizer:
    """
    Self-contained character/byte fallback tokenizer for offline environments.
    """

    def __init__(self, vocab_size: int = 151674):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token = "<|pad|>"
        self.eos_token = "<|im_end|>"
        self.bos_token = "<|im_start|>"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [(ord(c) % (self.vocab_size - 10)) + 3 for c in text]
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        chars = []
        for tid in token_ids:
            if tid in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            if tid >= 3:
                chars.append(chr((tid - 3) % 128))
        return "".join(chars)

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        batch_ids = [self.encode(t, add_special_tokens=False) for t in texts]
        if max_length:
            batch_ids = [ids[:max_length] for ids in batch_ids]

        max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0
        padded_ids = []
        attn_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            attn_masks.append([1] * len(ids) + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            }
        return {"input_ids": padded_ids, "attention_mask": attn_masks}


def get_tokenizer(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", vocab_size: int = 151674):
    """Load Hugging Face tokenizer with FallbackTokenizer if offline."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    except Exception as e:
        logger.info(f"Using FallbackTokenizer (model_id={model_id}): {e}")
        return FallbackTokenizer(vocab_size=vocab_size)


class MultimodalCollate:
    """
    Collate function preparing multimodal batch tensors with visual-token
    and instruction prompt label masking (-100) for causal language modeling loss.
    """

    def __init__(self, tokenizer: Any, max_seq_length: int = 512, mask_prompt: bool = True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self.eos_token_id = getattr(tokenizer, "eos_token_id", 2) or 2

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        s1_tensors = [item["image_s1"] for item in batch]
        s2_tensors = [item["image_s2"] for item in batch]

        batch_s1 = torch.stack(s1_tensors, dim=0)  # [B, 2, H, W]
        batch_s2 = torch.stack(s2_tensors, dim=0)  # [B, 10, H, W]

        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for item in batch:
            prompt = item.get("text", "")
            target = item.get("target_text", "")

            # Formulate standard chat prompt & target completion
            prompt_str = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            target_str = f"{target}<|im_end|>\n"

            if hasattr(self.tokenizer, "encode"):
                prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
                target_ids = self.tokenizer.encode(target_str, add_special_tokens=False)
            else:
                prompt_ids = self.tokenizer(prompt_str)["input_ids"]
                target_ids = self.tokenizer(target_str)["input_ids"]

            seq_ids = prompt_ids + target_ids
            if len(seq_ids) > self.max_seq_length:
                seq_ids = seq_ids[:self.max_seq_length]

            # Mask prompt tokens with -100 so loss is computed strictly on target answers
            if self.mask_prompt:
                prompt_len = min(len(prompt_ids), len(seq_ids))
                seq_labels = [-100] * prompt_len + seq_ids[prompt_len:]
            else:
                seq_labels = list(seq_ids)

            seq_mask = [1] * len(seq_ids)

            input_ids_list.append(seq_ids)
            labels_list.append(seq_labels)
            attention_mask_list.append(seq_mask)

        # Pad to longest sequence in batch
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for ids, lbls, mask in zip(input_ids_list, labels_list, attention_mask_list):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)
            padded_attention_masks.append(mask + [0] * pad_len)

        return {
            "image_s1": batch_s1,
            "image_s2": batch_s2,
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "raw_samples": batch,
        }


def load_dataset_split(
    manifest_path: str,
    data_root: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    seed: int = 42,
    img_size: int = 120,
    s1_bands: Optional[List[str]] = None,
    s2_bands: Optional[List[str]] = None,
) -> Dataset:
    """
    Load BigEarthNetDataset for a specified split with optional sample subsetting.
    """
    s1_bands = s1_bands or list(DEFAULT_S1_BANDS)
    s2_bands = s2_bands or list(DEFAULT_S2_BANDS)

    # Search for manifest path if relative
    p = Path(manifest_path)
    if not p.exists():
        candidates = [
            REPO_ROOT / manifest_path,
            REPO_ROOT / "data" / "manifests" / f"manifest_{split}.jsonl",
            REPO_ROOT / "data" / "manifests" / "manifest_full.jsonl",
            REPO_ROOT / "data" / "manifests" / "manifest_train.jsonl",
        ]
        for cand in candidates:
            if cand.exists():
                p = cand
                break

    if not p.exists():
        raise FileNotFoundError(f"Manifest not found for split '{split}' at path: {manifest_path}")

    logger.info(f"Loading '{split}' dataset from: {p}")
    dataset = BigEarthNetDataset(
        manifest_path=p,
        data_root=data_root,
        s1_bands=s1_bands,
        s2_bands=s2_bands,
        img_size=img_size,
        split=split if p.name == "manifest_full.jsonl" else None,
        is_training=(split == "train"),
        strict=False,
    )

    total_len = len(dataset)
    if total_len == 0:
        # If split filter resulted in 0 samples (e.g. manifest contains general samples), try without filter
        dataset = BigEarthNetDataset(
            manifest_path=p,
            data_root=data_root,
            s1_bands=s1_bands,
            s2_bands=s2_bands,
            img_size=img_size,
            is_training=(split == "train"),
            strict=False,
        )
        total_len = len(dataset)

    if total_len == 0:
        raise ValueError(f"No samples loaded from manifest {p} for split='{split}'")

    if max_samples is not None and max_samples < total_len:
        rng = np.random.default_rng(seed)
        indices = rng.choice(total_len, size=max_samples, replace=False)
        indices.sort()
        logger.info(f"Subsetting '{split}' split: {max_samples} / {total_len} samples (seed={seed})")
        return Subset(dataset, indices.tolist())

    logger.info(f"Loaded '{split}' split with {total_len} total samples.")
    return dataset


def compute_grad_norm(model: nn.Module) -> float:
    """Calculate total L2 norm of gradients across all trainable parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return math.sqrt(total_norm_sq)


@torch.no_grad()
def evaluate(
    model: RSInternVL,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """
    Evaluate model on validation split and return average validation loss.
    """
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0

    for batch in val_loader:
        img_s1 = batch["image_s1"].to(device)
        img_s2 = batch["image_s2"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(
                    image_s1=img_s1,
                    image_s2=img_s2,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.get("loss")
        else:
            outputs = model(
                image_s1=img_s1,
                image_s2=img_s2,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.get("loss")

        if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
            val_loss_sum += loss.item()
            val_batches += 1

    model.train()
    return val_loss_sum / max(1, val_batches)


def train_lora(
    config_path: Union[str, Path] = "configs/model/lora.yaml",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    device_override: Optional[str] = None,
    freeze_s1_override: Optional[bool] = None,
    freeze_s2_override: Optional[bool] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute production LoRA fine-tuning experiment and return complete metrics.
    """
    # 1. Load Configuration
    cfg_file = Path(config_path)
    if not cfg_file.exists() and (REPO_ROOT / config_path).exists():
        cfg_file = REPO_ROOT / config_path

    logger.info(f"Loading configuration from: {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})
    ckpt_cfg = cfg.get("checkpointing", {})
    out_cfg = cfg.get("output", {})

    # Apply command-line overrides
    if max_train_samples is not None:
        ds_cfg["max_train_samples"] = max_train_samples
    if max_val_samples is not None:
        ds_cfg["max_validation_samples"] = max_val_samples
    if epochs_override is not None:
        train_cfg["epochs"] = epochs_override
    if batch_size_override is not None:
        train_cfg["batch_size"] = batch_size_override
    if lr_override is not None:
        train_cfg["learning_rate"] = lr_override
    if freeze_s1_override is not None:
        model_cfg["freeze_s1_encoder"] = freeze_s1_override
    if freeze_s2_override is not None:
        model_cfg["freeze_s2_encoder"] = freeze_s2_override

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # 2. Select Device & Precision
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = device.type == "cuda"
    mixed_prec = train_cfg.get("mixed_precision", "auto").lower()

    if not use_cuda and hasattr(os, "cpu_count"):
        # Optimize CPU multi-threading
        cpu_cores = os.cpu_count() or 4
        torch.set_num_threads(cpu_cores)

    if mixed_prec == "auto":
        use_amp = use_cuda
        amp_dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    elif mixed_prec in ("fp16", "bf16"):
        use_amp = use_cuda
        amp_dtype = torch.bfloat16 if mixed_prec == "bf16" else torch.float16
    else:
        use_amp = False
        amp_dtype = torch.float32

    # Hardware & Package Environment Reporting
    import peft
    import transformers

    cuda_available = torch.cuda.is_available()
    cuda_dev_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    gpu_mem_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if cuda_available
        else "N/A"
    )

    logger.info("=" * 65)
    logger.info("       RS-INTERNVL TRAINING ENVIRONMENT & HARDWARE       ")
    logger.info("=" * 65)
    logger.info(f"CUDA Available:           {cuda_available}")
    logger.info(f"CUDA Device Name:         {cuda_dev_name}")
    logger.info(f"GPU Total Memory:         {gpu_mem_gb} GB" if cuda_available else "GPU Total Memory:         N/A")
    logger.info(f"Execution Target Device:  {device}")
    logger.info(f"PyTorch Version:          {torch.__version__}")
    logger.info(f"Transformers Version:     {transformers.__version__}")
    logger.info(f"PEFT Version:             {peft.__version__}")
    logger.info(f"Mixed Precision:          {mixed_prec} (AMP Active: {use_amp})")
    logger.info("=" * 65)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # 3. Setup Output & Checkpoint Directories
    checkpoint_dir = Path(ckpt_cfg.get("checkpoint_dir", "checkpoints/lora"))
    log_dir = Path(out_cfg.get("log_dir", "outputs/lora"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 4. Load Datasets & DataLoaders (Train & Validation)
    tokenizer = get_tokenizer(
        model_id=model_cfg.get("backbone", "OpenGVLab/InternVL3-1B"),
        vocab_size=model_cfg.get("vocab_size", 151674),
    )

    collate_fn = MultimodalCollate(
        tokenizer=tokenizer,
        max_seq_length=train_cfg.get("max_seq_length", 512),
        mask_prompt=train_cfg.get("mask_prompt", True),
    )

    train_dataset = load_dataset_split(
        manifest_path=ds_cfg.get("train_manifest", "data/manifests/manifest_train.jsonl"),
        data_root=ds_cfg.get("data_root", "data/bigearthnet_txt"),
        split="train",
        max_samples=ds_cfg.get("max_train_samples"),
        seed=seed,
        img_size=ds_cfg.get("img_size", 120),
        s1_bands=ds_cfg.get("s1_bands"),
        s2_bands=ds_cfg.get("s2_bands"),
    )

    val_dataset = load_dataset_split(
        manifest_path=ds_cfg.get("validation_manifest", "data/manifests/manifest_validation.jsonl"),
        data_root=ds_cfg.get("data_root", "data/bigearthnet_txt"),
        split="validation",
        max_samples=ds_cfg.get("max_validation_samples"),
        seed=seed,
        img_size=ds_cfg.get("img_size", 120),
        s1_bands=ds_cfg.get("s1_bands"),
        s2_bands=ds_cfg.get("s2_bands"),
    )

    batch_size = train_cfg.get("batch_size", 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 5. Build Model & Apply LoRA
    base_config = RSInternVLConfig(
        model_id=model_cfg.get("backbone", "OpenGVLab/InternVL3-1B"),
        config_file=model_cfg.get("config_file"),
        img_size=model_cfg.get("img_size", 120),
        s1_channels=model_cfg.get("s1_channels", 2),
        s1_hidden_dim=model_cfg.get("s1_hidden_dim", 512),
        s2_channels=model_cfg.get("s2_channels", 10),
        s2_hidden_dim=model_cfg.get("s2_hidden_dim", 768),
        projection_hidden_dim=model_cfg.get("projection_hidden_dim", 1024),
        projection_dropout=model_cfg.get("projection_dropout", 0.0),
        freeze_s1_encoder=model_cfg.get("freeze_s1_encoder", False),
        freeze_s2_encoder=model_cfg.get("freeze_s2_encoder", False),
        freeze_llm=True,  # Base LLM is always frozen when applying LoRA
    )

    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        model = load_lora_checkpoint(
            checkpoint_dir=resume_from_checkpoint,
            config_override=base_config,
            device=device,
            is_trainable=True,
        )
        audit = audit_parameters(model)
        target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    else:
        logger.info(f"Instantiating RSInternVL architecture...")
        model = RSInternVL(base_config)
        target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])

        model, audit = apply_lora(
            model=model,
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            bias=lora_cfg.get("bias", "none"),
            target_modules=target_modules,
            freeze_s1_encoder=model_cfg.get("freeze_s1_encoder", False),
            freeze_s2_encoder=model_cfg.get("freeze_s2_encoder", False),
        )

    model.to(device)

    # 6. Setup Optimizer, Scaler, and Schedulers
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Zero trainable parameters found! Ensure LoRA and projections are trainable.")

    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    grad_accum_steps = max(1, train_cfg.get("gradient_accumulation_steps", 4))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    epochs = train_cfg.get("epochs", 3)

    # 7. Initial Validation Baseline
    logger.info("Computing initial baseline validation loss before training...")
    initial_val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)
    logger.info(f"Initial Baseline Validation Loss: {initial_val_loss:.4f}")

    # 8. Training Loop
    epoch_records: List[Dict[str, Any]] = []
    step_history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_ckpt_dir = checkpoint_dir / "best"
    latest_ckpt_dir = checkpoint_dir / "latest"

    global_step = 0
    total_samples_processed = 0
    total_start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        epoch_train_loss_sum = 0.0
        batch_count = 0
        epoch_samples = 0
        latest_grad_norm = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img_s1 = batch["image_s1"].to(device)
            img_s2 = batch["image_s2"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            b_size = img_s1.shape[0]
            epoch_samples += b_size
            total_samples_processed += b_size

            # Forward pass
            if use_amp:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                    outputs = model(
                        image_s1=img_s1,
                        image_s2=img_s2,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"]
            else:
                outputs = model(
                    image_s1=img_s1,
                    image_s2=img_s2,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"]

            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Invalid loss encountered at epoch {epoch}, batch {batch_idx}: {loss}")

            loss_val = loss.item()
            epoch_train_loss_sum += loss_val
            batch_count += 1

            # Normalize loss for gradient accumulation
            loss_accum = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss_accum).backward()
            else:
                loss_accum.backward()

            # Optimizer Step on accumulation boundary
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if use_amp:
                    scaler.unscale_(optimizer)

                grad_norm = compute_grad_norm(model)
                latest_grad_norm = grad_norm

                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                step_record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch": batch_idx + 1,
                    "loss": round(loss_val, 6),
                    "grad_norm": round(grad_norm, 6),
                    "lr": lr,
                }
                step_history.append(step_record)

                if global_step % train_cfg.get("log_every_n_steps", 1) == 0:
                    logger.info(
                        f"Epoch [{epoch:02d}/{epochs:02d}] Step {global_step:03d} | "
                        f"Train Loss: {loss_val:.4f} | Grad Norm: {grad_norm:.4f}"
                    )

        epoch_time = time.time() - epoch_start_time
        avg_train_loss = epoch_train_loss_sum / max(1, batch_count)
        samples_per_sec = epoch_samples / max(0.001, epoch_time)

        # Validation Step
        val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)

        # GPU Memory Logging
        peak_vram_mb = 0.0
        if use_cuda:
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "validation_loss": round(val_loss, 6),
            "learning_rate": lr,
            "gradient_norm": round(latest_grad_norm, 6),
            "epoch_time_seconds": round(epoch_time, 2),
            "samples_per_second": round(samples_per_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2) if use_cuda else None,
        }
        epoch_records.append(epoch_summary)

        logger.info("-" * 65)
        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} Complete | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Throughput: {samples_per_sec:.1f} samples/s | Time: {epoch_time:.2f}s"
            + (f" | Peak VRAM: {peak_vram_mb:.1f}MB" if use_cuda else "")
        )
        logger.info("-" * 65)

        # Save Checkpoints
        if ckpt_cfg.get("save_every_epoch", False):
            try:
                epoch_ckpt_dir = checkpoint_dir / f"epoch_{epoch:02d}"
                save_lora_checkpoint(
                    model=model,
                    output_dir=epoch_ckpt_dir,
                    config=cfg,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_summary,
                )
            except Exception as e:
                logger.warning(f"Could not save epoch {epoch} checkpoint: {e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if ckpt_cfg.get("save_best", True):
                try:
                    save_lora_checkpoint(
                        model=model,
                        output_dir=best_ckpt_dir,
                        config=cfg,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        metrics=epoch_summary,
                    )
                    logger.info(f"New best validation loss: {val_loss:.4f} -> Saved to {best_ckpt_dir}")
                except Exception as e:
                    logger.warning(f"Could not save best checkpoint at epoch {epoch}: {e}")

        if ckpt_cfg.get("save_latest", True):
            try:
                save_lora_checkpoint(
                    model=model,
                    output_dir=latest_ckpt_dir,
                    config=cfg,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=epoch_summary,
                )
            except Exception as e:
                logger.warning(f"Could not save latest checkpoint at epoch {epoch}: {e}")

    total_time = time.time() - total_start_time
    initial_train_loss = epoch_records[0]["train_loss"] if epoch_records else 0.0
    final_train_loss = epoch_records[-1]["train_loss"] if epoch_records else 0.0
    final_val_loss = epoch_records[-1]["validation_loss"] if epoch_records else 0.0

    # 9. Final Summary Metrics
    metrics = {
        "model_backbone": model_cfg.get("backbone", "OpenGVLab/InternVL3-1B"),
        "device": str(device),
        "target_lora_modules": target_modules,
        "total_parameters": audit["total"],
        "trainable_parameters": audit["trainable"],
        "trainable_percentage": audit["trainable_percentage"],
        "s1_encoder_trainable": audit["s1_encoder_trainable"],
        "s2_encoder_trainable": audit["s2_encoder_trainable"],
        "projections_trainable": audit["projections_trainable"],
        "lora_trainable": audit["lora_trainable"],
        "frozen_llm": audit["frozen_llm"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "epochs": epochs,
        "total_steps": global_step,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_validation_loss": round(initial_val_loss, 6),
        "final_validation_loss": final_val_loss,
        "best_validation_loss": round(best_val_loss, 6),
        "overall_throughput_samples_per_sec": round(
            total_samples_processed / max(0.001, total_time), 2
        ),
        "total_training_time_seconds": round(total_time, 2),
        "peak_vram_mb": round(peak_vram_mb, 2) if use_cuda else None,
        "checkpoint_dir": str(latest_ckpt_dir),
        "best_checkpoint_dir": str(best_ckpt_dir),
        "epoch_records": epoch_records,
    }

    # Save metrics and step history
    metrics_path = log_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    step_log_path = log_dir / "step_log.jsonl"
    with open(step_log_path, "w", encoding="utf-8") as f:
        for entry in step_history:
            f.write(json.dumps(entry) + "\n")

    logger.info("=" * 70)
    logger.info("           RS-INTERNVL LoRA FINE-TUNING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Target Modules:           {metrics['target_lora_modules']}")
    logger.info(f"Total Parameters:         {metrics['total_parameters']:,}")
    logger.info(f"Trainable Parameters:     {metrics['trainable_parameters']:,} ({metrics['trainable_percentage']}%)")
    logger.info(f"Frozen Base LLM:          {metrics['frozen_llm']:,}")
    logger.info(f"Device:                   {metrics['device']}")
    if use_cuda:
        logger.info(f"Peak GPU VRAM:            {metrics['peak_vram_mb']:.1f} MB")
    logger.info(f"Initial Train Loss:       {metrics['initial_train_loss']:.4f}")
    logger.info(f"Final Train Loss:         {metrics['final_train_loss']:.4f}")
    best_train_loss = min(r["train_loss"] for r in epoch_records) if epoch_records else initial_train_loss
    logger.info(f"Best Train Loss:          {best_train_loss:.4f}")
    logger.info(f"Initial Validation Loss:  {metrics['initial_validation_loss']:.4f}")
    logger.info(f"Final Validation Loss:    {metrics['final_validation_loss']:.4f}")
    logger.info(f"Best Validation Loss:     {metrics['best_validation_loss']:.4f}")
    logger.info(f"Overall Throughput:       {metrics['overall_throughput_samples_per_sec']} samples/sec")
    logger.info(f"Total Training Time:      {metrics['total_training_time_seconds']:.2f}s")
    logger.info(f"Checkpoint Best Path:     {metrics['best_checkpoint_dir']}")
    logger.info(f"Checkpoint Latest Path:   {metrics['checkpoint_dir']}")
    logger.info("-" * 70)
    logger.info("EPOCH LOSS CURVES & METRICS TABLE:")
    logger.info(f"{'Epoch':<7} | {'Train Loss':<12} | {'Val Loss':<10} | {'Grad Norm':<10} | {'LR':<8} | {'Time (s)':<8} | {'Throughput':<12}")
    logger.info("-" * 70)
    for r in epoch_records:
        logger.info(
            f"{r['epoch']:<7} | {r['train_loss']:<12.4f} | {r['validation_loss']:<10.4f} | "
            f"{r['gradient_norm']:<10.4f} | {r['learning_rate']:<8.1e} | {r['epoch_time_seconds']:<8.1f} | "
            f"{r['samples_per_second']:<12.2f}"
        )
    logger.info("=" * 70)

    # 10. Verification: Reload Best Checkpoint and Run Real Validation Inference
    logger.info("\nPerforming Checkpoint Reload & Real Validation Sample Inference Verification...")
    try:
        reloaded_model = load_lora_checkpoint(
            checkpoint_dir=best_ckpt_dir,
            device=device,
            is_trainable=False,
        )
        reloaded_model.eval()

        # Get first sample from validation dataset
        val_sample = val_dataset[0] if len(val_dataset) > 0 else train_dataset[0]
        val_s1 = val_sample["image_s1"].to(device)
        val_s2 = val_sample["image_s2"].to(device)
        query = val_sample.get("text") or "Is coniferous forest present in this satellite patch?"
        target_answer = val_sample.get("target_text") or "Yes, coniferous forest is present."

        pred = reloaded_model.predict(
            image_s1=val_s1,
            image_s2=val_s2,
            query=query,
            tokenizer=tokenizer,
            max_new_tokens=64,
            do_sample=False,
        )

        print("\n" + "=" * 70)
        print("    POST-TRAINING CHECKPOINT INFERENCE ON REAL VALIDATION SAMPLE")
        print("=" * 70)
        print(f"Query:            {query}")
        print(f"Target:           {target_answer}")
        print(f"Generated answer: {pred['answer']}")
        print(f"Model score:      {pred['model_score']}")
        print(f"Claim:            {pred['claim']}")
        print(f"Claim type:       {pred['claim_type']}")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Inference verification on reloaded checkpoint failed: {e}", exc_info=True)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="RS-InternVL Production LoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/model/lora.yaml", help="Path to lora.yaml")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max train samples override")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples override")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", "--learning-rate", type=float, default=None, help="Learning rate override")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--freeze-s1-encoder", action="store_true", default=None, help="Freeze S1 SAR encoder")
    parser.add_argument("--freeze-s2-encoder", action="store_true", default=None, help="Freeze S2 Optical encoder")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint directory to resume from")

    args = parser.parse_args()

    train_lora(
        config_path=args.config,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        device_override=args.device,
        freeze_s1_override=args.freeze_s1_encoder,
        freeze_s2_override=args.freeze_s2_encoder,
        resume_from_checkpoint=args.resume_from,
    )


if __name__ == "__main__":
    main()
