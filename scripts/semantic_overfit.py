#!/usr/bin/env python3
"""
RS-InternVL: Step 6 Semantic Overfit & Generation Validation Script.

Executes a controlled semantic-overfit experiment on a deterministic 8-sample
BigEarthNet training subset, comparing training vs validation semantic generation,
exact-match accuracy, binary classification accuracy, and garbage/repetition rates.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

from data.bigearthnet_txt.dataset import BigEarthNetDataset
from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from scripts.evaluate_generation import (
    compute_aggregate_metrics,
    evaluate_sample,
    is_garbage_generation,
    normalize_text,
)
from training.lora import (
    apply_lora,
    audit_parameters,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from training.train_lora import (
    MultimodalCollate,
    get_tokenizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("semantic_overfit")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for determinism."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_deterministic_subsets(
    train_dataset: BigEarthNetDataset,
    val_dataset: BigEarthNetDataset,
    num_train: int = 8,
    num_val: int = 8,
    seed: int = 42,
    output_dir: Path = Path("outputs/semantic_overfit"),
) -> Tuple[Subset, Subset, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Select deterministic train and validation subsets and verify 0 patch overlap.
    """
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_train = len(train_dataset)
    total_val = len(val_dataset)

    train_indices = list(range(min(num_train, total_train)))
    val_indices = list(range(min(num_val, total_val)))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    # Collect metadata records
    train_meta = []
    train_patch_ids = set()
    for idx in train_indices:
        item = train_dataset.records[idx] if hasattr(train_dataset, "records") else {}
        patch_id = item.get("image_id", item.get("patch_id", f"train_{idx:02d}"))
        train_patch_ids.add(patch_id)
        train_meta.append({
            "index": idx,
            "patch_id": patch_id,
            "query": item.get("text_input", item.get("input", item.get("text", ""))),
            "target": item.get("text_output", item.get("output", item.get("target_text", ""))),
            "claim_type": item.get("task_type", item.get("type", "binary:presence")),
        })

    val_meta = []
    val_patch_ids = set()
    for idx in val_indices:
        item = val_dataset.records[idx] if hasattr(val_dataset, "records") else {}
        patch_id = item.get("image_id", item.get("patch_id", f"val_{idx:02d}"))
        val_patch_ids.add(patch_id)
        val_meta.append({
            "index": idx,
            "patch_id": patch_id,
            "query": item.get("text_input", item.get("input", item.get("text", ""))),
            "target": item.get("text_output", item.get("output", item.get("target_text", ""))),
            "claim_type": item.get("task_type", item.get("type", "binary:presence")),
        })

    # Zero patch overlap assertion
    overlap = train_patch_ids.intersection(val_patch_ids)
    if len(overlap) > 0:
        raise ValueError(f"FATAL: Train and Validation subsets share overlapping patches: {overlap}")

    # Save train_subset.json
    train_subset_path = output_dir / "train_subset.json"
    with open(train_subset_path, "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2, ensure_ascii=False)
    logger.info("Saved deterministic 8-sample train subset metadata -> %s", train_subset_path)

    return train_subset, val_subset, train_meta, val_meta


def evaluate_subset_generation(
    model: RSInternVL,
    dataset_subset: Subset,
    meta_list: List[Dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 32,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run greedy autoregressive generation for all samples in a subset and evaluate metrics.
    """
    model.eval()
    eval_records = []

    with torch.no_grad():
        for i, meta in enumerate(meta_list):
            sample = dataset_subset[i]
            s1_img = sample["image_s1"].unsqueeze(0).to(device)  # [1, 2, H, W]
            s2_img = sample["image_s2"].unsqueeze(0).to(device)  # [1, 10, H, W]
            query = meta["query"]
            target = meta["target"]
            patch_id = meta["patch_id"]
            task_type = meta["claim_type"]

            pred = model.predict(
                image_s1=s1_img,
                image_s2=s2_img,
                query=query,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            rec = evaluate_sample(
                query=query,
                target=target,
                generated=pred["answer"],
                patch_id=patch_id,
                task_type=task_type,
            )
            rec["model_score"] = pred.get("model_score", 0.0)
            eval_records.append(rec)

    metrics = compute_aggregate_metrics(eval_records)
    return eval_records, metrics


def run_semantic_overfit(
    config_path: str = "configs/model/semantic_overfit.yaml",
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute Step 6 semantic overfit training, progressive generation evaluation,
    checkpoint saving, and final best checkpoint verification.
    """
    # 1. Load config
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if device_str == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))

    checkpoint_dir = Path(cfg["output"].get("checkpoint_dir", "checkpoints/semantic_overfit"))
    output_dir = Path(cfg["output"].get("output_dir", "outputs/semantic_overfit"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_interval_epochs = int(cfg["training"].get("eval_interval_epochs", 5))
    epochs = epochs_override if epochs_override is not None else cfg["training"].get("epochs", 50)
    lr = lr_override if lr_override is not None else float(cfg["training"].get("learning_rate", 1e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 0.01))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    max_new_tokens = int(cfg["generation"].get("max_new_tokens", 32))
    seed = int(cfg["training"].get("seed", 42))

    set_seed(seed)

    # 2. Hardware Environment Logging
    logger.info("======================================================================")
    logger.info("           STEP 6: SEMANTIC OVERFIT & GENERATION VALIDATION           ")
    logger.info("======================================================================")
    logger.info("Target Device:              %s", device_str)
    logger.info("PyTorch Version:            %s", torch.__version__)
    logger.info("Epochs:                     %d", epochs)
    logger.info("Learning Rate:              %e", lr)
    logger.info("Eval Interval (Epochs):     %d", eval_interval_epochs)
    logger.info("Output Directory:           %s", output_dir)
    logger.info("Checkpoint Directory:       %s", checkpoint_dir)

    # 3. Load Datasets & Build Deterministic Subsets
    ds_cfg = cfg.get("dataset", {})
    train_manifest = ds_cfg.get("train_manifest", "data/manifests/manifest_train.jsonl")
    val_manifest = ds_cfg.get("validation_manifest", "data/manifests/manifest_validation.jsonl")
    data_root = ds_cfg.get("data_root", "data/bigearthnet_txt")

    train_ds = BigEarthNetDataset(
        data_root=data_root,
        manifest_path=train_manifest,
        s1_bands=ds_cfg.get("s1_bands", ["VV", "VH"]),
        s2_bands=ds_cfg.get("s2_bands", None),
        img_size=ds_cfg.get("img_size", 120),
        split="train",
        strict=False,
    )
    val_ds = BigEarthNetDataset(
        data_root=data_root,
        manifest_path=val_manifest,
        s1_bands=ds_cfg.get("s1_bands", ["VV", "VH"]),
        s2_bands=ds_cfg.get("s2_bands", None),
        img_size=ds_cfg.get("img_size", 120),
        split="validation",
        strict=False,
    )

    train_subset, val_subset, train_meta, val_meta = create_deterministic_subsets(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_train=ds_cfg.get("train_samples", 8),
        num_val=ds_cfg.get("validation_samples", 8),
        seed=seed,
        output_dir=output_dir,
    )

    logger.info("Training samples:           %d", len(train_subset))
    logger.info("Validation samples:         %d", len(val_subset))
    logger.info("Patch overlap:              0")

    # 4. Tokenizer & Collate
    tokenizer = get_tokenizer(cfg["model"].get("backbone", "OpenGVLab/InternVL3-1B"))
    collate_fn = MultimodalCollate(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 5. Load Model Checkpoint & Parameter Audit
    init_ckpt = cfg["model"].get("checkpoint", "checkpoints/lora/best")
    logger.info("Loading initial checkpoint from: %s", init_ckpt)

    model = load_lora_checkpoint(
        checkpoint_dir=init_ckpt,
        device=device_str,
        is_trainable=True,
    )
    model.to(device)

    # Parameter Audit Assertions
    audit = audit_parameters(model)
    logger.info("----------------------------------------------------------------------")
    logger.info("                       EXPLICIT PARAMETER AUDIT                       ")
    logger.info("----------------------------------------------------------------------")
    logger.info("Total parameters:           %d", audit["total"])
    logger.info("Trainable parameters:       %d (%.4f%%)", audit["trainable"], audit["trainable_percentage"])
    logger.info("Frozen parameters:          %d", audit["frozen"])
    logger.info("LoRA parameters:            %d", audit["lora_trainable"])
    logger.info("S1 parameters:              %d", audit["s1_encoder_total"])
    logger.info("S2 parameters:              %d", audit["s2_encoder_total"])
    logger.info("Projection parameters:      %d", audit["projections_total"])
    logger.info("Frozen LLM parameters:      %d", audit["frozen_llm"])

    # Strict Assertions
    for name, p in model.language_model.named_parameters():
        if "lora" not in name.lower():
            assert not p.requires_grad, f"ASSERTION FAILED: Base LLM parameter '{name}' is not frozen!"
    
    lora_params = [p for n, p in model.language_model.named_parameters() if "lora" in n.lower()]
    assert len(lora_params) > 0, "ASSERTION FAILED: No LoRA parameters found!"
    for p in lora_params:
        assert p.requires_grad, "ASSERTION FAILED: LoRA parameter is not trainable!"

    # 6. Optimizer Setup
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # 7. Initial Baseline Evaluation (Epoch 0)
    logger.info("----------------------------------------------------------------------")
    logger.info("Computing Epoch 0 Baseline Losses and Generation Metrics...")
    
    # Train Loss Eval
    model.eval()
    init_train_loss = 0.0
    with torch.no_grad():
        for b in train_loader:
            b_dev = {k: v.to(device) for k, v in b.items() if k != "raw_samples" and isinstance(v, torch.Tensor)}
            out = model(**b_dev)
            init_train_loss += out["loss"].item()
    init_train_loss /= max(1, len(train_loader))

    # Val Loss Eval
    init_val_loss = 0.0
    with torch.no_grad():
        for b in val_loader:
            b_dev = {k: v.to(device) for k, v in b.items() if k != "raw_samples" and isinstance(v, torch.Tensor)}
            out = model(**b_dev)
            init_val_loss += out["loss"].item()
    init_val_loss /= max(1, len(val_loader))

    train_recs_0, train_met_0 = evaluate_subset_generation(model, train_subset, train_meta, tokenizer, device, max_new_tokens)
    val_recs_0, val_met_0 = evaluate_subset_generation(model, val_subset, val_meta, tokenizer, device, max_new_tokens)

    logger.info("Baseline Epoch 0: Train Loss: %.4f | Val Loss: %.4f | Train ExactMatch: %.2f%% | Val ExactMatch: %.2f%%",
                init_train_loss, init_val_loss, train_met_0["exact_match_accuracy"] * 100, val_met_0["exact_match_accuracy"] * 100)

    # 8. Training Loop with Progressive Epoch Generation
    history: List[Dict[str, Any]] = []
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    total_start_time = time.time()

    last_train_metrics = train_met_0
    last_val_metrics = val_met_0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_dev = {k: v.to(device) for k, v in batch.items() if k != "raw_samples" and isinstance(v, torch.Tensor)}
            outputs = model(**batch_dev)
            loss = outputs["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        # Validation Loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_dev = {k: v.to(device) for k, v in batch.items() if k != "raw_samples" and isinstance(v, torch.Tensor)}
                out = model(**batch_dev)
                val_loss += out["loss"].item()
        avg_val_loss = val_loss / max(1, len(val_loader))

        # Generation Evaluation for all 8 train + 8 val samples
        should_eval_gen = (epoch % eval_interval_epochs == 0) or (epoch == epochs) or (epoch == 1)
        if should_eval_gen:
            train_records, train_metrics = evaluate_subset_generation(
                model=model,
                dataset_subset=train_subset,
                meta_list=train_meta,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            val_records, val_metrics = evaluate_subset_generation(
                model=model,
                dataset_subset=val_subset,
                meta_list=val_meta,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            last_train_metrics = train_metrics
            last_val_metrics = val_metrics

            # Save Epoch Generation Artifact
            epoch_artifact = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(avg_val_loss, 4),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "train_predictions": train_records,
                "val_predictions": val_records,
            }
            epoch_art_path = output_dir / f"generation_epoch_{epoch:02d}.json"
            with open(epoch_art_path, "w", encoding="utf-8") as f:
                json.dump(epoch_artifact, f, indent=2, ensure_ascii=False)
        else:
            train_metrics = last_train_metrics
            val_metrics = last_val_metrics

        epoch_time = time.time() - epoch_start

        # Checkpoint Saving: Best & Latest
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            best_epoch = epoch
            save_lora_checkpoint(
                model=model,
                output_dir=checkpoint_dir / "best",
                config=cfg,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_exact_match": train_metrics["exact_match_accuracy"],
                    "train_binary_accuracy": train_metrics["binary_accuracy"],
                },
            )

        # Save latest checkpoint
        save_lora_checkpoint(
            model=model,
            output_dir=checkpoint_dir / "latest",
            config=cfg,
            optimizer=optimizer,
            epoch=epoch,
            metrics={
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_exact_match": train_metrics["exact_match_accuracy"],
                "train_binary_accuracy": train_metrics["binary_accuracy"],
            },
        )

        # Record progressive history
        history_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "train_exact_match": train_metrics["exact_match_accuracy"],
            "train_binary_accuracy": train_metrics["binary_accuracy"],
            "train_validity": train_metrics["generation_validity_rate"],
            "train_garbage_rate": train_metrics["garbage_rate"],
            "train_repetition_rate": train_metrics["repetition_rate"],
            "val_exact_match": val_metrics["exact_match_accuracy"],
            "val_binary_accuracy": val_metrics["binary_accuracy"],
            "val_validity": val_metrics["generation_validity_rate"],
            "val_garbage_rate": val_metrics["garbage_rate"],
            "val_repetition_rate": val_metrics["repetition_rate"],
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(history_entry)

        # Progressive metrics save
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": round(best_val_loss, 4),
            }, f, indent=2)

        logger.info(
            "Epoch [%02d/%02d] | Train Loss: %.4f | Val Loss: %.4f | Train EM: %.1f%% | Train BinAcc: %.1f%% | Train Valid: %.1f%% | Val EM: %.1f%% | Time: %.1fs%s",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
            train_metrics["exact_match_accuracy"] * 100,
            train_metrics["binary_accuracy"] * 100,
            train_metrics["generation_validity_rate"] * 100,
            val_metrics["exact_match_accuracy"] * 100,
            epoch_time,
            " (BEST)" if is_best else "",
        )

    total_training_time = time.time() - total_start_time

    # 9. Final Evaluation on Reloaded BEST Checkpoint
    logger.info("======================================================================")
    logger.info("Reloading BEST checkpoint in fresh model instance for final evaluation...")
    reloaded_model = load_lora_checkpoint(
        checkpoint_dir=str(checkpoint_dir / "best"),
        device=device_str,
    )
    reloaded_model.to(device)

    final_train_recs, final_train_metrics = evaluate_subset_generation(
        model=reloaded_model,
        dataset_subset=train_subset,
        meta_list=train_meta,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
    )
    final_val_recs, final_val_metrics = evaluate_subset_generation(
        model=reloaded_model,
        dataset_subset=val_subset,
        meta_list=val_meta,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
    )

    # Save final prediction artifacts
    with open(output_dir / "final_train_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_train_recs, f, indent=2, ensure_ascii=False)
    with open(output_dir / "final_validation_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_val_recs, f, indent=2, ensure_ascii=False)

    logger.info("Saved final predictions -> %s and %s",
                output_dir / "final_train_predictions.json",
                output_dir / "final_validation_predictions.json")

    # 10. Compute Complete Results Summary
    summary = {
        "device": device_str,
        "epochs": epochs,
        "total_training_time_s": round(total_training_time, 2),
        "initial_train_loss": round(init_train_loss, 4),
        "final_train_loss": round(history[-1]["train_loss"], 4),
        "best_train_loss": round(best_train_loss, 4),
        "initial_val_loss": round(init_val_loss, 4),
        "final_val_loss": round(history[-1]["val_loss"], 4),
        "best_val_loss": round(best_val_loss, 4),
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "final_train_predictions": final_train_recs,
        "final_val_predictions": final_val_recs,
        "best_checkpoint_path": str(checkpoint_dir / "best"),
        "latest_checkpoint_path": str(checkpoint_dir / "latest"),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="RS-InternVL Step 6 Semantic Overfit & Generation Validation")
    parser.add_argument("--config", type=str, default="configs/model/semantic_overfit.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    summary = run_semantic_overfit(
        config_path=args.config,
        epochs_override=args.epochs,
        lr_override=args.lr,
    )

    print("\n" + "=" * 70)
    print("      STEP 6: SEMANTIC OVERFIT & GENERATION VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Final Train Loss:            {summary['final_train_loss']:.4f}")
    print(f"Final Validation Loss:       {summary['final_val_loss']:.4f}")
    print(f"Train Exact Match Accuracy:  {summary['final_train_metrics']['exact_match_accuracy'] * 100:.2f}%")
    print(f"Train Binary Accuracy:       {summary['final_train_metrics']['binary_accuracy'] * 100:.2f}%")
    print(f"Train Validity Rate:         {summary['final_train_metrics']['generation_validity_rate'] * 100:.2f}%")
    print(f"Train Garbage Rate:          {summary['final_train_metrics']['garbage_rate'] * 100:.2f}%")
    print("-" * 70)
    print(f"Val Exact Match Accuracy:    {summary['final_val_metrics']['exact_match_accuracy'] * 100:.2f}%")
    print(f"Val Binary Accuracy:         {summary['final_val_metrics']['binary_accuracy'] * 100:.2f}%")
    print(f"Val Validity Rate:           {summary['final_val_metrics']['generation_validity_rate'] * 100:.2f}%")
    print(f"Val Garbage Rate:            {summary['final_val_metrics']['garbage_rate'] * 100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
