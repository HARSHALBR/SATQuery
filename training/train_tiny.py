#!/usr/bin/env python3
"""
RS-InternVL: Tiny-Subset Overfitting Feasibility Pipeline (Step 3).

Deliberately overfits a deterministic tiny BigEarthNet.txt training subset to verify
gradient flow, multimodal token fusion, attention masking, and language model loss alignment
through the complete RS-InternVL architecture.

Usage:
    python training/train_tiny.py --config training/config.yaml
    python training/train_tiny.py --config training/config.yaml --num-samples 16 --epochs 20
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

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from models.rs_internvl.config import RSInternVLConfig, DEFAULT_S1_BANDS, DEFAULT_S2_BANDS
from models.rs_internvl.model import RSInternVL
from data.bigearthnet_txt.dataset import BigEarthNetDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_tiny")


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
    Lightweight character/byte-level fallback tokenizer if HF tokenizer is offline.
    Ensures tests and offline training runs remain 100% self-contained.
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
        # Simple ASCII / UTF-8 mapping shifted by special tokens
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
        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        batch_ids = [self.encode(t, add_special_tokens=False) for t in texts]
        if max_length:
            batch_ids = [ids[:max_length] for ids in batch_ids]

        max_len = max(len(ids) for ids in batch_ids) if batch_ids else 0
        padded_ids = []
        attn_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            padded = ids + [self.pad_token_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            padded_ids.append(padded)
            attn_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
            }
        return {"input_ids": padded_ids, "attention_mask": attn_masks}


def get_tokenizer(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", vocab_size: int = 151674):
    """Attempt to load HuggingFace tokenizer with FallbackTokenizer if offline."""
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
        logger.warning(f"Could not load HuggingFace tokenizer '{model_id}': {e}. Using FallbackTokenizer.")
        return FallbackTokenizer(vocab_size=vocab_size)


class MultimodalCollate:
    """
    Collate function that prepares multi-sensor satellite images, tokenizes instructions,
    and constructs aligned label tensors where visual tokens and instruction prompts
    are masked with -100 so language-model loss is computed strictly on target answers.
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

            # Formulate instruction prompt and target completion
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
                # Truncate keeping target if possible
                seq_ids = seq_ids[:self.max_seq_length]

            # Mask prompt tokens with -100 if mask_prompt is True
            if self.mask_prompt:
                prompt_len = min(len(prompt_ids), len(seq_ids))
                seq_labels = [-100] * prompt_len + seq_ids[prompt_len:]
            else:
                seq_labels = list(seq_ids)

            seq_mask = [1] * len(seq_ids)

            input_ids_list.append(seq_ids)
            labels_list.append(seq_labels)
            attention_mask_list.append(seq_mask)

        # Pad batch to longest sequence in batch
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


def load_tiny_dataset(
    manifest_path: str,
    data_root: str,
    num_samples: int = 16,
    seed: int = 42,
    img_size: int = 120,
    s1_bands: Optional[List[str]] = None,
    s2_bands: Optional[List[str]] = None,
) -> Subset:
    """
    Load the existing BigEarthNetDataset and extract a deterministic tiny subset
    from the training split ONLY.
    """
    s1_bands = s1_bands or list(DEFAULT_S1_BANDS)
    s2_bands = s2_bands or list(DEFAULT_S2_BANDS)

    # Resolve manifest path
    p = Path(manifest_path)
    if not p.exists():
        # Search repository paths
        candidates = [
            REPO_ROOT / manifest_path,
            REPO_ROOT / "data" / "manifests" / "manifest_train.jsonl",
            REPO_ROOT / "data" / "manifests" / "manifest_full.jsonl",
            REPO_ROOT / "data" / "bigearthnet_txt" / "BigEarthNet.txt.parquet",
        ]
        for cand in candidates:
            if cand.exists():
                p = cand
                break

    if not p.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found at '{manifest_path}' (or searched alternatives)."
        )

    logger.info(f"Loading BigEarthNet training dataset from: {p}")
    full_train_dataset = BigEarthNetDataset(
        manifest_path=p,
        data_root=data_root,
        s1_bands=s1_bands,
        s2_bands=s2_bands,
        img_size=img_size,
        split="train",
        is_training=True,
        strict=False,
    )

    total_train = len(full_train_dataset)
    if total_train == 0:
        raise ValueError(f"No samples found for split='train' in manifest {p}")

    # Deterministic subset selection using numpy RNG with seed
    rng = np.random.default_rng(seed)
    actual_k = min(num_samples, total_train)
    indices = rng.choice(total_train, size=actual_k, replace=False)
    indices.sort()
    indices_list = indices.tolist()

    logger.info(
        f"Selected deterministic tiny training subset: {actual_k} samples from {total_train} total "
        f"(seed={seed}, indices={indices_list[:5]}...)"
    )

    return Subset(full_train_dataset, indices_list)


def build_model(
    model_cfg: Dict[str, Any],
    device: torch.device,
) -> RSInternVL:
    """
    Instantiate RSInternVL model and configure trainable/frozen parameters.
    """
    config = RSInternVLConfig(
        s1_channels=2,
        s2_channels=10,
        img_size=model_cfg.get("img_size", 120),
        num_hidden_layers=model_cfg.get("num_hidden_layers", 4),
        num_attention_heads=model_cfg.get("num_attention_heads", 14),
        num_key_value_heads=model_cfg.get("num_key_value_heads", 2),
        intermediate_size=model_cfg.get("intermediate_size", 4864),
        llm_hidden_dim=model_cfg.get("llm_hidden_dim", 896),
        vocab_size=model_cfg.get("vocab_size", 151674),
        freeze_s1_encoder=model_cfg.get("freeze_s1_encoder", False),
        freeze_s2_encoder=model_cfg.get("freeze_s2_encoder", False),
        freeze_llm=model_cfg.get("freeze_llm", False),
    )

    model = RSInternVL(config)

    # Explicitly configure trainability based on config
    freeze_s1 = model_cfg.get("freeze_s1_encoder", False)
    freeze_s2 = model_cfg.get("freeze_s2_encoder", False)
    freeze_llm = model_cfg.get("freeze_llm", False)

    if freeze_s1:
        model.s1_encoder.freeze()
    else:
        model.s1_encoder.unfreeze()

    if freeze_s2:
        model.s2_encoder.freeze()
    else:
        model.s2_encoder.unfreeze()

    if freeze_llm:
        model._freeze_llm()
    else:
        model.unfreeze_llm()

    model.to(device)
    return model


def compute_grad_norm(model: nn.Module) -> float:
    """Calculate total L2 norm of gradients across all trainable parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return math.sqrt(total_norm_sq)


def train_tiny(
    config_path: Union[str, Path] = "training/config.yaml",
    num_samples_override: Optional[int] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    learning_rate_override: Optional[float] = None,
    device_override: Optional[str] = None,
    freeze_s1_override: Optional[bool] = None,
    freeze_s2_override: Optional[bool] = None,
    freeze_llm_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Execute tiny subset overfitting experiment and return experiment summary.
    """
    # 1. Load Configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    out_cfg = cfg.get("output", {})

    # Apply command-line overrides
    if num_samples_override is not None:
        ds_cfg["num_samples"] = num_samples_override
    if epochs_override is not None:
        train_cfg["epochs"] = epochs_override
    if batch_size_override is not None:
        train_cfg["batch_size"] = batch_size_override
    if learning_rate_override is not None:
        train_cfg["learning_rate"] = learning_rate_override
    if freeze_s1_override is not None:
        model_cfg["freeze_s1_encoder"] = freeze_s1_override
    if freeze_s2_override is not None:
        model_cfg["freeze_s2_encoder"] = freeze_s2_override
    if freeze_llm_override is not None:
        model_cfg["freeze_llm"] = freeze_llm_override

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # 2. Select Device & Precision
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = device.type == "cuda"
    mixed_prec = train_cfg.get("mixed_precision", "no").lower()
    use_amp = use_cuda and mixed_prec in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if mixed_prec == "bf16" else torch.float16

    logger.info(f"Execution Device: {device} | CUDA Available: {use_cuda} | Mixed Precision: {mixed_prec} (Active: {use_amp})")

    # 3. Setup Output Directories
    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "checkpoints/tiny_overfit"))
    log_dir = Path(out_cfg.get("log_dir", "outputs/tiny_overfit"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save active config snapshot
    active_config_path = checkpoint_dir / "training_config.yaml"
    with open(active_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # 4. Load Dataset & DataLoader
    num_samples = ds_cfg.get("num_samples", 16)
    subset_dataset = load_tiny_dataset(
        manifest_path=ds_cfg.get("train_manifest", "data/manifests/manifest_train.jsonl"),
        data_root=ds_cfg.get("data_root", "data/bigearthnet_txt"),
        num_samples=num_samples,
        seed=ds_cfg.get("seed", 42),
        img_size=ds_cfg.get("img_size", 120),
        s1_bands=ds_cfg.get("s1_bands"),
        s2_bands=ds_cfg.get("s2_bands"),
    )

    tokenizer = get_tokenizer(
        model_id=model_cfg.get("checkpoint") or "Qwen/Qwen2.5-0.5B-Instruct",
        vocab_size=model_cfg.get("vocab_size", 151674),
    )

    collate_fn = MultimodalCollate(
        tokenizer=tokenizer,
        max_seq_length=train_cfg.get("max_seq_length", 512),
        mask_prompt=True,
    )

    batch_size = train_cfg.get("batch_size", 2)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 5. Build Model & Audit Parameters
    model = build_model(model_cfg, device)
    param_counts = model.get_num_parameters()

    logger.info("=" * 65)
    logger.info("           RS-INTERNVL TINY TRAINING AUDIT          ")
    logger.info("=" * 65)
    logger.info(f"Total Parameters:        {param_counts['total']:,}")
    logger.info(f"Trainable Parameters:    {param_counts['trainable']:,}")
    logger.info(f"Frozen Parameters:       {param_counts['frozen']:,}")
    logger.info(f"  - S1 Encoder:          {param_counts['s1_encoder']:,} (Frozen: {model_cfg.get('freeze_s1_encoder', False)})")
    logger.info(f"  - S2 Encoder:          {param_counts['s2_encoder']:,} (Frozen: {model_cfg.get('freeze_s2_encoder', False)})")
    logger.info(f"  - Projections:         {param_counts['projections']:,} (Trainable)")
    logger.info(f"  - Language Model:      {param_counts['language_model']:,} (Frozen: {model_cfg.get('freeze_llm', False)})")
    logger.info("=" * 65)

    # 6. Setup Optimizer & Scaler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found in model! Check freezing configurations.")

    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_accum_steps = max(1, train_cfg.get("gradient_accumulation_steps", 1))
    clip_norm = float(train_cfg.get("clip_grad_norm", 1.0))

    # 7. Training Loop
    epochs = train_cfg.get("epochs", 20)
    step_history: List[Dict[str, Any]] = []
    epoch_losses: List[float] = []
    best_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    final_checkpoint_path = checkpoint_dir / "final_checkpoint.pt"

    global_step = 0
    start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss_sum = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(dataloader):
            img_s1 = batch["image_s1"].to(device)
            img_s2 = batch["image_s2"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with optional autocast
            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
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
                raise RuntimeError(f"Encountered invalid loss value at epoch {epoch}, step {batch_idx}: {loss}")

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            batch_count += 1

            # Normalize for gradient accumulation
            loss_accum = loss / grad_accum_steps

            if use_amp:
                scaler.scale(loss_accum).backward()
            else:
                loss_accum.backward()

            # Optimizer Step & Gradient Logging on accumulation boundary
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                if use_amp:
                    scaler.unscale_(optimizer)

                # Gradient statistics
                grad_norm = compute_grad_norm(model)
                if not math.isfinite(grad_norm):
                    logger.warning(f"Non-finite gradient norm detected at step {global_step}: {grad_norm}")

                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)

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
                        f"Loss: {loss_val:.4f} | Grad Norm: {grad_norm:.4f}"
                    )

        avg_epoch_loss = epoch_loss_sum / max(1, batch_count)
        epoch_losses.append(avg_epoch_loss)

        logger.info(
            f"--> Epoch {epoch:02d} Complete | Avg Loss: {avg_epoch_loss:.4f} "
            f"({'New Best' if avg_epoch_loss < best_loss else 'Best: ' + f'{best_loss:.4f}'})"
        )

        # Save Best Checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            if train_cfg.get("save_best", True):
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss,
                        "config": cfg,
                    },
                    best_checkpoint_path,
                )
                logger.info(f"Saved best checkpoint to: {best_checkpoint_path}")

    # 8. Save Final Checkpoint
    torch.save(
        {
            "epoch": epochs,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_losses[-1] if epoch_losses else None,
            "config": cfg,
        },
        final_checkpoint_path,
    )
    logger.info(f"Saved final checkpoint to: {final_checkpoint_path}")

    total_time = time.time() - start_time
    initial_loss = epoch_losses[0] if epoch_losses else 0.0
    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    loss_decreased = final_loss < initial_loss

    # 9. Save Metrics & Step Log
    metrics = {
        "device": str(device),
        "num_samples": num_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "total_steps": global_step,
        "trainable_parameters": param_counts["trainable"],
        "frozen_parameters": param_counts["frozen"],
        "total_parameters": param_counts["total"],
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "best_loss": round(best_loss, 6),
        "loss_decreased": loss_decreased,
        "loss_reduction_pct": round(((initial_loss - final_loss) / max(1e-6, initial_loss)) * 100, 2),
        "training_time_seconds": round(total_time, 2),
        "checkpoint_path": str(final_checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
    }

    metrics_path = log_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log_jsonl_path = log_dir / "step_log.jsonl"
    with open(log_jsonl_path, "w", encoding="utf-8") as f:
        for entry in step_history:
            f.write(json.dumps(entry) + "\n")

    logger.info("=" * 65)
    logger.info("        RS-INTERNVL OVERFITTING EXPERIMENT SUMMARY        ")
    logger.info("=" * 65)
    logger.info(f"Device:                  {metrics['device']}")
    logger.info(f"Number of Samples:       {metrics['num_samples']}")
    logger.info(f"Trainable Parameters:    {metrics['trainable_parameters']:,}")
    logger.info(f"Initial Loss:            {metrics['initial_loss']:.4f}")
    logger.info(f"Final Loss:              {metrics['final_loss']:.4f}")
    logger.info(f"Loss Decreased:          {metrics['loss_decreased']} ({metrics['loss_reduction_pct']}%)")
    logger.info(f"Final Checkpoint:        {metrics['checkpoint_path']}")
    logger.info("=" * 65)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="RS-InternVL Tiny Subset Overfit Training")
    parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of training samples (8, 16, 32)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--learning-rate", "--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda')")
    parser.add_argument("--freeze-s1-encoder", action="store_true", default=None, help="Freeze S1 SAR encoder")
    parser.add_argument("--freeze-s2-encoder", action="store_true", default=None, help="Freeze S2 Optical encoder")
    parser.add_argument("--freeze-llm", action="store_true", default=None, help="Freeze Language Model backbone")

    args = parser.parse_args()

    train_tiny(
        config_path=args.config,
        num_samples_override=args.num_samples,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        learning_rate_override=args.learning_rate,
        device_override=args.device,
        freeze_s1_override=args.freeze_s1_encoder,
        freeze_s2_override=args.freeze_s2_encoder,
        freeze_llm_override=args.freeze_llm,
    )


if __name__ == "__main__":
    main()
