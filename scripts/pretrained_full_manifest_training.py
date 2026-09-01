"""
STEP 10: Pretrained Full-Manifest Training.

Official baseline training on ALL currently prepared real BigEarthNet samples:
  - Train:      32 samples (from data/manifests/manifest_train.jsonl)
  - Validation:  8 samples (from data/manifests/manifest_validation.jsonl)

Key improvements over Step 9:
  1. Full 32-sample training set (no subset)
  2. Cosine LR scheduler with linear warmup
  3. Gradient accumulation (steps=4, effective BS=4)
  4. Extended metrics: val loss, binary P/R/F1, MCQ accuracy, token length stats
  5. Disk-safe: saves only best/ checkpoint
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import yaml

from data.bigearthnet_txt.dataset import BigEarthNetDataset
from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from scripts.evaluate_generation import (
    compute_aggregate_metrics,
    evaluate_sample,
    extract_binary_answer,
    is_garbage_generation,
    normalize_text,
)
from training.lora import apply_lora, audit_parameters, save_lora_checkpoint
from training.train_lora import MultimodalCollate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("step10_full_manifest")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(model_id: str = "OpenGVLab/InternVL3-1B") -> Any:
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def compute_binary_metrics(records: List[Dict]) -> Dict[str, float]:
    """Compute precision, recall, F1, and accuracy for binary YES/NO tasks."""
    tp = fp = tn = fn = 0
    for rec in records:
        target = rec.get("target", "")
        generated = rec.get("generated_text", rec.get("generated", ""))
        t_upper = target.strip().upper()
        g_upper = generated.strip().upper()

        if t_upper.startswith("YES") or t_upper.startswith("NO"):
            gt = "YES" if t_upper.startswith("YES") else "NO"
            pred = extract_binary_answer(generated)
            if pred is None:
                pred = "NO"  # treat unanswered as NO
            if gt == "YES":
                if pred == "YES":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == "NO":
                    tn += 1
                else:
                    fp += 1

    total_binary = tp + fp + tn + fn
    accuracy = (tp + tn) / max(1, total_binary) * 100.0
    precision = tp / max(1, tp + fp) * 100.0
    recall = tp / max(1, tp + fn) * 100.0
    f1_denom = precision + recall
    f1 = 2 * precision * recall / max(1e-8, f1_denom)

    return {
        "binary_accuracy_pct": round(accuracy, 2),
        "binary_precision_pct": round(precision, 2),
        "binary_recall_pct": round(recall, 2),
        "binary_f1_pct": round(f1, 2),
        "binary_tp": tp,
        "binary_fp": fp,
        "binary_tn": tn,
        "binary_fn": fn,
        "binary_total": total_binary,
    }


def compute_mcq_metrics(records: List[Dict]) -> Dict[str, float]:
    """Compute accuracy for MCQ/land-cover tasks."""
    correct = total = 0
    for rec in records:
        target = rec.get("target", "")
        generated = rec.get("generated_text", rec.get("generated", ""))
        t_upper = target.strip().upper()
        if t_upper.startswith("YES") or t_upper.startswith("NO"):
            continue
        total += 1
        norm_t = normalize_text(target)
        norm_g = normalize_text(generated)
        if norm_t and norm_g and norm_t in norm_g:
            correct += 1

    accuracy = (correct / max(1, total)) * 100.0 if total > 0 else None
    return {
        "mcq_accuracy_pct": round(accuracy, 2) if accuracy is not None else None,
        "mcq_total": total,
        "mcq_correct": correct,
    }


def compute_length_stats(records: List[Dict]) -> Dict[str, Any]:
    """Compute generation length statistics."""
    lengths = [rec.get("generation_length", len(rec.get("generated_text", "").split())) for rec in records]
    if not lengths:
        return {"avg_length": None, "min_length": None, "max_length": None}
    return {
        "avg_length": round(sum(lengths) / len(lengths), 2),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def evaluate_yes_no_logits(
    model: RSInternVL,
    tokenizer: Any,
    dataset: Any,
    meta_list: List[Dict],
    device: torch.device,
) -> Tuple[List[Dict], float]:
    """Compute direct P(YES) vs P(NO) candidate probabilities."""
    model.eval()
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    records = []
    correct = total_binary = 0

    with torch.no_grad():
        for i, meta in enumerate(meta_list):
            sample = dataset[i]
            s1_img = sample["image_s1"].unsqueeze(0).to(device)
            s2_img = sample["image_s2"].unsqueeze(0).to(device)
            query = meta["query"]
            target = meta["target"]
            patch_id = meta["patch_id"]

            gt_upper = target.strip().upper()
            gt_class = "YES" if gt_upper.startswith("YES") else ("NO" if gt_upper.startswith("NO") else "OTHER")

            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            s1_feat, s2_feat, s1_tok, s2_tok = model.encode_vision(s1_img, s2_img)
            text_embeds = model.language_model.get_input_embeddings()(input_ids)
            fused = model.fusion(
                s1_tokens=s1_tok, s2_tokens=s2_tok,
                text_embeds=text_embeds, text_attention_mask=attention_mask,
            )
            lm_outputs = model.language_model(
                inputs_embeds=fused.inputs_embeds,
                attention_mask=fused.attention_mask,
                return_dict=True,
            )
            next_logits = lm_outputs.logits[0, -1, :]
            all_probs = F.softmax(next_logits, dim=-1)
            p_yes_raw = float(all_probs[yes_token_id])
            p_no_raw = float(all_probs[no_token_id])
            denom = max(1e-12, p_yes_raw + p_no_raw)
            p_yes_norm = p_yes_raw / denom
            p_no_norm = p_no_raw / denom
            pred_class = "YES" if p_yes_norm >= 0.5 else "NO"
            is_correct = pred_class == gt_class

            if gt_class in ("YES", "NO"):
                total_binary += 1
                if is_correct:
                    correct += 1

            records.append({
                "patch_id": patch_id,
                "query": query,
                "target": target,
                "ground_truth_class": gt_class,
                "p_yes_normalized": round(p_yes_norm, 4),
                "p_no_normalized": round(p_no_norm, 4),
                "predicted_class": pred_class,
                "is_correct": is_correct,
            })

    acc = (correct / max(1, total_binary)) * 100.0
    return records, acc


def evaluate_generation(
    model: RSInternVL,
    tokenizer: Any,
    dataset: Any,
    meta_list: List[Dict],
    device: torch.device,
    max_new_tokens: int = 32,
) -> Tuple[List[Dict], Dict]:
    """Run greedy generation and compute all semantic metrics."""
    model.eval()
    eval_records = []

    with torch.no_grad():
        for i, meta in enumerate(meta_list):
            sample = dataset[i]
            s1_img = sample["image_s1"].unsqueeze(0).to(device)
            s2_img = sample["image_s2"].unsqueeze(0).to(device)
            query = meta["query"]
            target = meta["target"]
            patch_id = meta["patch_id"]
            task_type = meta.get("claim_type", "binary")

            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            tok_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([tok_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            gen_tokens, _ = model.generate(
                image_s1=s1_img, image_s2=s2_img,
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
            )
            gen_ids = gen_tokens[0].tolist() if gen_tokens.numel() > 0 else []
            raw_gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            rec = evaluate_sample(
                query=query, target=target, generated=raw_gen,
                patch_id=patch_id, task_type=task_type,
            )
            rec["generated_text"] = raw_gen
            rec["token_ids"] = gen_ids
            rec["generation_length"] = len(gen_ids)
            eval_records.append(rec)

    agg = compute_aggregate_metrics(eval_records)
    agg["exact_match_pct"] = round(agg["exact_match_accuracy"] * 100.0, 2)
    agg["binary_accuracy_pct"] = round(agg["binary_accuracy"] * 100.0, 2)
    agg["validity_rate_pct"] = round(agg["generation_validity_rate"] * 100.0, 2)
    agg["garbage_rate_pct"] = round(agg["garbage_rate"] * 100.0, 2)

    binary_metrics = compute_binary_metrics(eval_records)
    mcq_metrics = compute_mcq_metrics(eval_records)
    length_stats = compute_length_stats(eval_records)

    return eval_records, {**agg, **binary_metrics, **mcq_metrics, **length_stats}


def compute_val_loss(
    model: RSInternVL,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute average validation loss without gradient accumulation."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            s1 = batch["image_s1"].to(device)
            s2 = batch["image_s2"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                image_s1=s1, image_s2=s2,
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs["loss"].item()
            count += 1
    return total_loss / max(1, count)


def build_meta_list(dataset: Any) -> List[Dict]:
    """Extract query/target/patch_id metadata from dataset samples."""
    meta = []
    for i in range(len(dataset)):
        sample = dataset[i]
        meta.append({
            "query": sample["text"],
            "target": sample["target_text"],
            "patch_id": sample["image_id"],
            "claim_type": sample.get("claim_type", "binary"),
        })
    return meta


def run_experiment(config_path: str = "configs/model/pretrained_full_manifest.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    checkpoint_dir = Path(cfg["output"].get("checkpoint_dir", "checkpoints/pretrained_lora"))
    output_dir = Path(cfg["output"].get("output_dir", "outputs/pretrained_lora"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["training"].get("epochs", 25))
    eval_epochs = set(cfg["training"].get("eval_epochs", [0, 1, 2, 5, 10, 15, 20, 25]))
    lr = float(cfg["training"].get("learning_rate", 1e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 0.01))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    grad_accum_steps = int(cfg["training"].get("gradient_accumulation_steps", 4))
    warmup_steps = int(cfg["training"].get("warmup_steps", 8))
    max_new_tokens = int(cfg["generation"].get("max_new_tokens", 32))
    seed = int(cfg["training"].get("seed", 42))

    set_seed(seed)

    logger.info("=" * 70)
    logger.info("    STEP 10: PRETRAINED FULL-MANIFEST TRAINING")
    logger.info("=" * 70)
    logger.info("Device:                       %s", device_str)
    logger.info("PyTorch:                      %s", torch.__version__)
    if torch.cuda.is_available():
        logger.info("GPU:                          %s", torch.cuda.get_device_name(0))
        logger.info("GPU Memory:                   %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    logger.info("Epochs:                       %d", epochs)
    logger.info("Gradient Accumulation Steps:  %d", grad_accum_steps)
    logger.info("Effective Batch Size:         %d", grad_accum_steps)
    logger.info("Learning Rate:                %e", lr)
    logger.info("LR Scheduler:                 Cosine with Warmup (%d warmup steps)", warmup_steps)
    logger.info("Eval Epochs:                  %s", sorted(eval_epochs))

    # --- Dataset ---
    ds_cfg = cfg.get("dataset", {})
    train_ds = BigEarthNetDataset(
        data_root=ds_cfg.get("data_root", "data/bigearthnet_txt"),
        manifest_path=ds_cfg.get("train_manifest", "data/manifests/manifest_train.jsonl"),
        s1_bands=ds_cfg.get("s1_bands", ["VV", "VH"]),
        s2_bands=None,
        img_size=ds_cfg.get("img_size", 120),
        split="train",
        strict=False,
    )
    val_ds = BigEarthNetDataset(
        data_root=ds_cfg.get("data_root", "data/bigearthnet_txt"),
        manifest_path=ds_cfg.get("validation_manifest", "data/manifests/manifest_validation.jsonl"),
        s1_bands=ds_cfg.get("s1_bands", ["VV", "VH"]),
        s2_bands=None,
        img_size=ds_cfg.get("img_size", 120),
        split="validation",
        strict=False,
    )

    assert len(train_ds) == 32, f"Expected 32 train samples, got {len(train_ds)}"
    assert len(val_ds) == 8, f"Expected 8 validation samples, got {len(val_ds)}"

    # Verify zero patch overlap
    train_ids = {train_ds[i]["image_id"] for i in range(len(train_ds))}
    val_ids = {val_ds[i]["image_id"] for i in range(len(val_ds))}
    overlap = train_ids & val_ids
    assert len(overlap) == 0, f"PATCH LEAKAGE DETECTED: {overlap}"

    logger.info("Train samples:                %d", len(train_ds))
    logger.info("Validation samples:           %d", len(val_ds))
    logger.info("Patch overlap:                0 (verified)")

    train_meta = build_meta_list(train_ds)
    val_meta = build_meta_list(val_ds)

    # --- Tokenizer & DataLoaders ---
    tokenizer = get_tokenizer(cfg["model"].get("backbone", "OpenGVLab/InternVL3-1B"))
    collate_fn = MultimodalCollate(tokenizer=tokenizer, max_seq_length=512)

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
    )

    # --- Model ---
    logger.info("Loading pretrained RSInternVL model...")
    model_cfg = RSInternVLConfig(
        model_id=cfg["model"].get("backbone", "OpenGVLab/InternVL3-1B"),
        pretrained_backbone=True,
        img_size=cfg["model"].get("img_size", 120),
        s1_channels=cfg["model"].get("s1_channels", 2),
        s1_hidden_dim=cfg["model"].get("s1_hidden_dim", 512),
        s2_channels=cfg["model"].get("s2_channels", 10),
        s2_hidden_dim=cfg["model"].get("s2_hidden_dim", 768),
        projection_hidden_dim=cfg["model"].get("projection_hidden_dim", 1024),
        freeze_s1_encoder=cfg["model"].get("freeze_s1_encoder", False),
        freeze_s2_encoder=cfg["model"].get("freeze_s2_encoder", False),
        freeze_llm=cfg["model"].get("freeze_llm", True),
    )
    base_model = RSInternVL(model_cfg)

    lora_cfg = cfg.get("lora", {})
    model, audit = apply_lora(
        base_model,
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        freeze_s1_encoder=cfg["model"].get("freeze_s1_encoder", False),
        freeze_s2_encoder=cfg["model"].get("freeze_s2_encoder", False),
    )
    model.to(device)

    # Parameter assertions
    assert audit["total"] == 649517696, f"Param count mismatch: {audit['total']}"
    assert audit["frozen_llm"] == 629697920, f"Frozen LLM mismatch: {audit['frozen_llm']}"
    assert audit["trainable"] == 19819776, f"Trainable mismatch: {audit['trainable']}"

    # Verify base LLM frozen
    for name, p in model.language_model.named_parameters():
        if "lora" not in name.lower():
            assert not p.requires_grad, f"Base LLM param '{name}' is not frozen!"

    logger.info("Parameter audit passed.")
    logger.info("Total params:                 %d", audit["total"])
    logger.info("Frozen LLM:                   %d", audit["frozen_llm"])
    logger.info("Trainable:                    %d", audit["trainable"])

    # --- Optimizer & Scheduler ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps)
    )

    epoch_history = []
    best_val_loss = float("inf")
    best_val_binary_acc = -1.0

    # --- Epoch 0 Baseline ---
    if 0 in eval_epochs:
        logger.info("=== EVALUATION AT EPOCH 0 (Pretrained Baseline) ===")
        val_loss_0 = compute_val_loss(model, val_loader, device)
        t_recs0, t_met0 = evaluate_generation(model, tokenizer, train_ds, train_meta, device, max_new_tokens)
        v_recs0, v_met0 = evaluate_generation(model, tokenizer, val_ds, val_meta, device, max_new_tokens)
        _, t_yn0 = evaluate_yes_no_logits(model, tokenizer, train_ds, train_meta, device)
        _, v_yn0 = evaluate_yes_no_logits(model, tokenizer, val_ds, val_meta, device)
        logger.info("[Epoch 0] Val Loss: %.4f | Train BinAcc: %.1f%% | Val BinAcc: %.1f%% | Train Garbage: %.1f%% | Val Garbage: %.1f%%",
                    val_loss_0, t_met0["binary_accuracy_pct"], v_met0["binary_accuracy_pct"],
                    t_met0["garbage_rate_pct"], v_met0["garbage_rate_pct"])
        epoch_history.append({
            "epoch": 0, "train_loss": None, "val_loss": round(val_loss_0, 4), "learning_rate": lr,
            "train_binary_acc": t_met0["binary_accuracy_pct"], "train_binary_f1": t_met0.get("binary_f1_pct"),
            "train_garbage_rate": t_met0["garbage_rate_pct"], "train_validity_rate": t_met0["validity_rate_pct"],
            "train_yn_logits_acc": round(t_yn0, 2),
            "val_binary_acc": v_met0["binary_accuracy_pct"], "val_binary_f1": v_met0.get("binary_f1_pct"),
            "val_garbage_rate": v_met0["garbage_rate_pct"], "val_validity_rate": v_met0["validity_rate_pct"],
            "val_yn_logits_acc": round(v_yn0, 2),
            "val_mcq_accuracy": v_met0.get("mcq_accuracy_pct"),
        })

    # --- Training Loop ---
    logger.info("Starting %d-epoch training (gradient accumulation=%d)...", epochs, grad_accum_steps)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step_idx, batch in enumerate(train_loader):
            s1 = batch["image_s1"].to(device)
            s2 = batch["image_s2"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                image_s1=s1, image_s2=s2,
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"] / grad_accum_steps
            loss.backward()

            total_loss += outputs["loss"].item()
            num_batches += 1

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(1, num_batches)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr

        logger.info("Epoch %2d/%d | Train Loss: %.4f | LR: %.2e", epoch, epochs, avg_loss, current_lr)

        if epoch in eval_epochs:
            val_loss = compute_val_loss(model, val_loader, device)
            t_recs, t_met = evaluate_generation(model, tokenizer, train_ds, train_meta, device, max_new_tokens)
            v_recs, v_met = evaluate_generation(model, tokenizer, val_ds, val_meta, device, max_new_tokens)
            _, t_yn = evaluate_yes_no_logits(model, tokenizer, train_ds, train_meta, device)
            _, v_yn = evaluate_yes_no_logits(model, tokenizer, val_ds, val_meta, device)

            logger.info(
                "[Epoch %d] Val Loss: %.4f | Train BinAcc: %.1f%% | Val BinAcc: %.1f%% | "
                "Train BinF1: %.1f%% | Val BinF1: %.1f%% | Train Garbage: %.1f%% | Val Garbage: %.1f%%",
                epoch, val_loss,
                t_met["binary_accuracy_pct"], v_met["binary_accuracy_pct"],
                t_met.get("binary_f1_pct", 0.0), v_met.get("binary_f1_pct", 0.0),
                t_met["garbage_rate_pct"], v_met["garbage_rate_pct"],
            )
            logger.info("[Epoch %d] Sample Train Gen: %r", epoch, t_recs[0]["generated_text"])
            logger.info("[Epoch %d] Sample Val Gen:   %r", epoch, v_recs[0]["generated_text"])

            record = {
                "epoch": epoch, "train_loss": round(avg_loss, 4), "val_loss": round(val_loss, 4),
                "learning_rate": round(current_lr, 8),
                "train_binary_acc": t_met["binary_accuracy_pct"],
                "train_binary_precision": t_met.get("binary_precision_pct"),
                "train_binary_recall": t_met.get("binary_recall_pct"),
                "train_binary_f1": t_met.get("binary_f1_pct"),
                "train_garbage_rate": t_met["garbage_rate_pct"],
                "train_validity_rate": t_met["validity_rate_pct"],
                "train_yn_logits_acc": round(t_yn, 2),
                "train_avg_length": t_met.get("avg_length"),
                "train_mcq_accuracy": t_met.get("mcq_accuracy_pct"),
                "val_binary_acc": v_met["binary_accuracy_pct"],
                "val_binary_precision": v_met.get("binary_precision_pct"),
                "val_binary_recall": v_met.get("binary_recall_pct"),
                "val_binary_f1": v_met.get("binary_f1_pct"),
                "val_garbage_rate": v_met["garbage_rate_pct"],
                "val_validity_rate": v_met["validity_rate_pct"],
                "val_yn_logits_acc": round(v_yn, 2),
                "val_avg_length": v_met.get("avg_length"),
                "val_mcq_accuracy": v_met.get("mcq_accuracy_pct"),
            }
            epoch_history.append(record)

            # Checkpoint: save best on val loss improvement OR val binary acc improvement
            improved = (val_loss < best_val_loss) or (v_met["binary_accuracy_pct"] > best_val_binary_acc)
            if improved:
                best_val_loss = min(best_val_loss, val_loss)
                best_val_binary_acc = max(best_val_binary_acc, v_met["binary_accuracy_pct"])
                save_lora_checkpoint(
                    model=model,
                    output_dir=checkpoint_dir / "best",
                    epoch=epoch,
                    global_step=epoch * num_batches,
                    metrics={"val_loss": best_val_loss, "val_binary_acc": best_val_binary_acc},
                    config=cfg,
                )
                logger.info("*** New best checkpoint saved at epoch %d (val_loss=%.4f, val_bin_acc=%.1f%%) ***",
                            epoch, best_val_loss, best_val_binary_acc)

    total_time = time.time() - start_time
    logger.info("Training completed in %.1fs (%.1f min)", total_time, total_time / 60.0)

    # --- Final Comprehensive Evaluation ---
    logger.info("=== FINAL EVALUATION AT EPOCH %d ===", epochs)
    final_val_loss = compute_val_loss(model, val_loader, device)
    final_t_recs, final_t_met = evaluate_generation(model, tokenizer, train_ds, train_meta, device, max_new_tokens)
    final_v_recs, final_v_met = evaluate_generation(model, tokenizer, val_ds, val_meta, device, max_new_tokens)
    _, final_t_yn = evaluate_yes_no_logits(model, tokenizer, train_ds, train_meta, device)
    _, final_v_yn = evaluate_yes_no_logits(model, tokenizer, val_ds, val_meta, device)

    # Save predictions
    with open(output_dir / "final_train_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_t_recs, f, indent=2, ensure_ascii=False)
    with open(output_dir / "final_validation_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_v_recs, f, indent=2, ensure_ascii=False)

    results = {
        "step": 10,
        "experiment": "pretrained_full_manifest_training",
        "training_time_seconds": round(total_time, 1),
        "total_epochs": epochs,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_binary_acc_pct": round(best_val_binary_acc, 2),
        "final_val_loss": round(final_val_loss, 4),
        "epoch_history": epoch_history,
        "final_train_metrics": final_t_met,
        "final_val_metrics": final_v_met,
        "final_train_yn_logits_acc": round(final_t_yn, 2),
        "final_val_yn_logits_acc": round(final_v_yn, 2),
        "parameter_audit": audit,
        "config_path": config_path,
        "checkpoint_dir": str(checkpoint_dir / "best"),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("STEP 10 SUMMARY:")
    logger.info("  Train samples:         32")
    logger.info("  Val samples:            8")
    logger.info("  Epochs:               %d", epochs)
    logger.info("  Best Val Loss:        %.4f", best_val_loss)
    logger.info("  Best Val Binary Acc:  %.1f%%", best_val_binary_acc)
    logger.info("  Final Train BinAcc:   %.1f%%", final_t_met["binary_accuracy_pct"])
    logger.info("  Final Val BinAcc:     %.1f%%", final_v_met["binary_accuracy_pct"])
    logger.info("  Final Train Garbage:  %.1f%%", final_t_met["garbage_rate_pct"])
    logger.info("  Final Val Garbage:    %.1f%%", final_v_met["garbage_rate_pct"])
    logger.info("  Checkpoint:           %s", checkpoint_dir / "best")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="RS-InternVL Step 10 Full Manifest Training")
    parser.add_argument("--config", type=str, default="configs/model/pretrained_full_manifest.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 epoch for pre-flight verification")
    args = parser.parse_args()

    if args.dry_run:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["training"]["epochs"] = 1
        cfg["training"]["eval_epochs"] = [0, 1]
        tmp_cfg = "configs/model/pretrained_full_manifest_dryrun.yaml"
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        run_experiment(config_path=tmp_cfg)
    else:
        run_experiment(config_path=args.config)


if __name__ == "__main__":
    main()
