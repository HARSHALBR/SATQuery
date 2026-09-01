"""
STEP 9: Pretrained-Backbone Semantic Overfit & Generation Validation Experiment.

Trains a clean RS-InternVL model (with authentic pretrained language backbone and LoRA)
on the exact same 8-sample training subset from Step 6 to verify semantic memorization,
task accuracy, and natural language fluency.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 stdout for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
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
logger = logging.getLogger("pretrained_semantic_overfit")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_tokenizer(model_id: str = "OpenGVLab/InternVL3-1B") -> Any:
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def evaluate_yes_no_logits(
    model: RSInternVL,
    tokenizer: Any,
    dataset_subset: Any,
    meta_list: List[Dict[str, Any]],
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], float]:
    """Compute candidate YES/NO probabilities at the assistant token boundary."""
    model.eval()
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]   # 9454
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]     # 2753

    records = []
    correct = 0
    total_binary = 0

    with torch.no_grad():
        for i, meta in enumerate(meta_list):
            sample = dataset_subset[i]
            s1_img = sample["image_s1"].unsqueeze(0).to(device)
            s2_img = sample["image_s2"].unsqueeze(0).to(device)
            query = meta["query"]
            target = meta["target"]
            patch_id = meta["patch_id"]

            gt_upper = target.strip().upper()
            if gt_upper.startswith("YES"):
                gt_class = "YES"
            elif gt_upper.startswith("NO"):
                gt_class = "NO"
            else:
                gt_class = "OTHER"

            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            s1_feat, s2_feat, s1_tok, s2_tok = model.encode_vision(s1_img, s2_img)
            text_embeds = model.language_model.get_input_embeddings()(input_ids)

            fused = model.fusion(
                s1_tokens=s1_tok,
                s2_tokens=s2_tok,
                text_embeds=text_embeds,
                text_attention_mask=attention_mask,
            )

            lm_outputs = model.language_model(
                inputs_embeds=fused.inputs_embeds,
                attention_mask=fused.attention_mask,
                return_dict=True,
            )
            next_logits = lm_outputs.logits[0, -1, :]
            all_probs = F.softmax(next_logits, dim=-1)

            p_yes_raw = float(all_probs[yes_token_id].item())
            p_no_raw = float(all_probs[no_token_id].item())
            binary_denom = max(1e-12, p_yes_raw + p_no_raw)
            p_yes_norm = p_yes_raw / binary_denom
            p_no_norm = p_no_raw / binary_denom

            pred_class = "YES" if p_yes_norm >= 0.5 else "NO"
            is_correct = (pred_class == gt_class)

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


def evaluate_generation_on_subset(
    model: RSInternVL,
    tokenizer: Any,
    dataset_subset: Any,
    meta_list: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 32,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run greedy generation and evaluate semantic metrics."""
    model.eval()
    eval_records = []

    with torch.no_grad():
        for i, meta in enumerate(meta_list):
            sample = dataset_subset[i]
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

            gen_tokens, token_probs = model.generate(
                image_s1=s1_img,
                image_s2=s2_img,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            gen_ids = gen_tokens[0].tolist() if gen_tokens.numel() > 0 else []
            raw_gen = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            first_tok = tokenizer.decode([gen_ids[0]]) if gen_ids else ""

            rec = evaluate_sample(
                query=query,
                target=target,
                generated=raw_gen,
                patch_id=patch_id,
                task_type=task_type,
            )
            rec["generated_text"] = raw_gen
            rec["token_ids"] = gen_ids
            rec["first_token"] = first_tok
            rec["generation_length"] = len(gen_ids)

            eval_records.append(rec)

    metrics = compute_aggregate_metrics(eval_records)
    # Convert fractions to percentages for reporting consistency
    metrics["exact_match_pct"] = round(metrics["exact_match_accuracy"] * 100.0, 2)
    metrics["binary_accuracy_pct"] = round(metrics["binary_accuracy"] * 100.0, 2)
    metrics["validity_rate_pct"] = round(metrics["generation_validity_rate"] * 100.0, 2)
    metrics["garbage_rate_pct"] = round(metrics["garbage_rate"] * 100.0, 2)
    return eval_records, metrics


def run_experiment(
    config_path: str = "configs/model/pretrained_semantic_overfit.yaml",
    epochs_override: Optional[int] = None,
    lr_override: Optional[float] = None,
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    checkpoint_dir = Path(cfg["output"].get("checkpoint_dir", "checkpoints/pretrained_semantic_overfit"))
    output_dir = Path(cfg["output"].get("output_dir", "outputs/pretrained_semantic_overfit"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_epochs = set(cfg["training"].get("eval_epochs", [0, 1, 2, 5, 10]))
    epochs = epochs_override if epochs_override is not None else cfg["training"].get("epochs", 10)
    lr = lr_override if lr_override is not None else float(cfg["training"].get("learning_rate", 1e-4))
    weight_decay = float(cfg["training"].get("weight_decay", 0.01))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    max_new_tokens = int(cfg["generation"].get("max_new_tokens", 32))
    seed = int(cfg["training"].get("seed", 42))

    set_seed(seed)

    logger.info("======================================================================")
    logger.info("      STEP 9: PRETRAINED-BACKBONE SEMANTIC OVERFIT EXPERIMENT         ")
    logger.info("======================================================================")
    logger.info("Target Device:              %s", device_str)
    logger.info("PyTorch Version:            %s", torch.__version__)
    logger.info("Epochs:                     %d", epochs)
    logger.info("Learning Rate:              %e", lr)
    logger.info("Eval Epochs:                %s", sorted(list(eval_epochs)))
    logger.info("Output Directory:           %s", output_dir)
    logger.info("Checkpoint Directory:       %s", checkpoint_dir)

    # 1. Load Datasets & Exact Deterministic Subsets from Step 6
    ds_cfg = cfg.get("dataset", {})
    train_manifest = ds_cfg.get("train_manifest", "data/manifests/manifest_train.jsonl")
    val_manifest = ds_cfg.get("validation_manifest", "data/manifests/manifest_validation.jsonl")
    data_root = ds_cfg.get("data_root", "data/bigearthnet_txt")

    train_ds = BigEarthNetDataset(
        data_root=data_root,
        manifest_path=train_manifest,
        s1_bands=ds_cfg.get("s1_bands", ["VV", "VH"]),
        s2_bands=None,
        img_size=ds_cfg.get("img_size", 120),
        split="train",
        strict=False,
    )
    val_ds = BigEarthNetDataset(
        data_root=data_root,
        manifest_path=val_manifest,
        s1_bands=ds_cfg.get("s1_bands", ["VV", "VH"]),
        s2_bands=None,
        img_size=ds_cfg.get("img_size", 120),
        split="validation",
        strict=False,
    )

    # Load 8-sample training subset metadata from Step 6
    subset_file = Path(ds_cfg.get("train_subset_file", "outputs/semantic_overfit/train_subset.json"))
    if subset_file.exists():
        with open(subset_file, "r", encoding="utf-8") as f:
            train_meta = json.load(f)
    else:
        train_meta = [{"query": train_ds[i]["text"], "target": train_ds[i]["target_text"], "patch_id": train_ds[i]["image_id"], "claim_type": "binary"} for i in range(8)]

    val_meta = [{"query": val_ds[i]["text"], "target": val_ds[i]["target_text"], "patch_id": val_ds[i]["image_id"], "claim_type": "binary"} for i in range(8)]

    train_subset = Subset(train_ds, list(range(len(train_meta))))
    val_subset = Subset(val_ds, list(range(len(val_meta))))

    logger.info("Training samples:           %d", len(train_subset))
    logger.info("Validation samples:         %d", len(val_subset))

    # 2. Tokenizer & Collate
    tokenizer = get_tokenizer(cfg["model"].get("backbone", "OpenGVLab/InternVL3-1B"))
    collate_fn = MultimodalCollate(tokenizer=tokenizer, max_seq_length=512)

    train_loader = DataLoader(
        train_subset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    # 3. Clean Model Initialization from Authentic Pretrained Backbone
    logger.info("Instantiating clean RSInternVL model with authentic pretrained language backbone...")
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

    # Apply LoRA
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

    # Strict Parameter Audit Assertions
    assert audit["total"] == 649517696, f"Expected 649,517,696 params, got {audit['total']}"
    assert audit["frozen_llm"] == 629697920, f"Expected 629,697,920 frozen params, got {audit['frozen_llm']}"
    assert audit["trainable"] == 19819776, f"Expected 19,819,776 trainable params, got {audit['trainable']}"

    for name, p in model.language_model.named_parameters():
        if "lora" not in name.lower():
            assert not p.requires_grad, f"ASSERTION FAILED: Base LLM parameter '{name}' is not frozen!"

    # 4. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    epoch_history = []
    best_exact_match = -1.0
    best_loss = float("inf")

    # Evaluate Epoch 0 Baseline
    if 0 in eval_epochs:
        logger.info("=== EVALUATION AT EPOCH 0 (Pretrained Baseline) ===")
        t_records, t_metrics = evaluate_generation_on_subset(model, tokenizer, train_subset, train_meta, device, max_new_tokens)
        v_records, v_metrics = evaluate_generation_on_subset(model, tokenizer, val_subset, val_meta, device, max_new_tokens)
        _, t_yn_acc = evaluate_yes_no_logits(model, tokenizer, train_subset, train_meta, device)
        _, v_yn_acc = evaluate_yes_no_logits(model, tokenizer, val_subset, val_meta, device)

        logger.info(
            "[Epoch 0] Train EM: %.1f%% | Train BinAcc: %.1f%% | Train Garbage: %.1f%% | Train Y/N Logits Acc: %.1f%%",
            t_metrics["exact_match_pct"],
            t_metrics["binary_accuracy_pct"],
            t_metrics["garbage_rate_pct"],
            t_yn_acc,
        )
        logger.info(
            "[Epoch 0] Val EM:   %.1f%% | Val BinAcc:   %.1f%% | Val Garbage:   %.1f%% | Val Y/N Logits Acc:   %.1f%%",
            v_metrics["exact_match_pct"],
            v_metrics["binary_accuracy_pct"],
            v_metrics["garbage_rate_pct"],
            v_yn_acc,
        )
        epoch_history.append({
            "epoch": 0,
            "train_loss": None,
            "train_exact_match": t_metrics["exact_match_pct"],
            "train_binary_acc": t_metrics["binary_accuracy_pct"],
            "train_garbage_rate": t_metrics["garbage_rate_pct"],
            "train_yn_logits_acc": t_yn_acc,
            "val_exact_match": v_metrics["exact_match_pct"],
            "val_binary_acc": v_metrics["binary_accuracy_pct"],
            "val_garbage_rate": v_metrics["garbage_rate_pct"],
            "val_yn_logits_acc": v_yn_acc,
        })

    # 5. Training Loop
    logger.info("Starting 10-Epoch Optimization...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            image_s1 = batch["image_s1"].to(device)
            image_s2 = batch["image_s2"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                image_s1=image_s1,
                image_s2=image_s2,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f}")

        # Evaluation at scheduled epochs
        if epoch in eval_epochs:
            t_records, t_metrics = evaluate_generation_on_subset(model, tokenizer, train_subset, train_meta, device, max_new_tokens)
            v_records, v_metrics = evaluate_generation_on_subset(model, tokenizer, val_subset, val_meta, device, max_new_tokens)
            _, t_yn_acc = evaluate_yes_no_logits(model, tokenizer, train_subset, train_meta, device)
            _, v_yn_acc = evaluate_yes_no_logits(model, tokenizer, val_subset, val_meta, device)

            logger.info(
                "[Epoch %d] Train EM: %.1f%% | Train BinAcc: %.1f%% | Train Garbage: %.1f%% | Train Y/N Logits Acc: %.1f%%",
                epoch,
                t_metrics["exact_match_pct"],
                t_metrics["binary_accuracy_pct"],
                t_metrics["garbage_rate_pct"],
                t_yn_acc,
            )
            logger.info(
                "[Epoch %d] Val EM:   %.1f%% | Val BinAcc:   %.1f%% | Val Garbage:   %.1f%% | Val Y/N Logits Acc:   %.1f%%",
                epoch,
                v_metrics["exact_match_pct"],
                v_metrics["binary_accuracy_pct"],
                v_metrics["garbage_rate_pct"],
                v_yn_acc,
            )

            # Sample prediction preview
            logger.info("Sample Train Gen #0: %r", t_records[0]["generated_text"])
            logger.info("Sample Val Gen #0:   %r", v_records[0]["generated_text"])

            epoch_history.append({
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "train_exact_match": t_metrics["exact_match_pct"],
                "train_binary_acc": t_metrics["binary_accuracy_pct"],
                "train_garbage_rate": t_metrics["garbage_rate_pct"],
                "train_yn_logits_acc": t_yn_acc,
                "val_exact_match": v_metrics["exact_match_pct"],
                "val_binary_acc": v_metrics["binary_accuracy_pct"],
                "val_garbage_rate": v_metrics["garbage_rate_pct"],
                "val_yn_logits_acc": v_yn_acc,
            })

            # Checkpoint saving
            if avg_loss < best_loss or t_metrics["exact_match_pct"] >= best_exact_match:
                best_loss = avg_loss
                best_exact_match = t_metrics["exact_match_pct"]
                save_lora_checkpoint(
                    model=model,
                    output_dir=checkpoint_dir / "best",
                    epoch=epoch,
                    global_step=epoch * num_batches,
                    metrics={"train_exact_match": best_exact_match, "train_loss": best_loss},
                    config=cfg,
                )

    total_time = time.time() - start_time
    logger.info(f"Step 9 Training Completed in {total_time:.2f}s.")

    # 6. Final Comprehensive Evaluation & Outputs
    final_train_records, final_train_metrics = evaluate_generation_on_subset(
        model, tokenizer, train_subset, train_meta, device, max_new_tokens
    )
    final_val_records, final_val_metrics = evaluate_generation_on_subset(
        model, tokenizer, val_subset, val_meta, device, max_new_tokens
    )
    train_yn_records, final_t_yn = evaluate_yes_no_logits(model, tokenizer, train_subset, train_meta, device)
    val_yn_records, final_v_yn = evaluate_yes_no_logits(model, tokenizer, val_subset, val_meta, device)

    # Save per-sample predictions
    with open(output_dir / "final_train_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_train_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "final_validation_predictions.json", "w", encoding="utf-8") as f:
        json.dump(final_val_records, f, indent=2, ensure_ascii=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "epoch_history": epoch_history,
            "final_train_metrics": final_train_metrics,
            "final_validation_metrics": final_val_metrics,
            "final_train_yn_logits_acc": final_t_yn,
            "final_val_yn_logits_acc": final_v_yn,
        }, f, indent=2, ensure_ascii=False)

    return {
        "epoch_history": epoch_history,
        "final_train_metrics": final_train_metrics,
        "final_val_metrics": final_val_metrics,
        "final_train_records": final_train_records,
        "final_val_records": final_val_records,
    }


def main():
    parser = argparse.ArgumentParser(description="RS-InternVL Step 9 Pretrained Semantic Overfit")
    parser.add_argument("--config", type=str, default="configs/model/pretrained_semantic_overfit.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    run_experiment(config_path=args.config, epochs_override=args.epochs, lr_override=args.lr)


if __name__ == "__main__":
    main()
