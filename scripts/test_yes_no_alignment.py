"""
STEP 7F: Minimal YES/NO Classification Test

Evaluates direct logits and probabilities for YES vs NO candidate tokens
at the assistant prompt boundary for the 8 training samples and 8 validation samples.

Determines whether the model has learned the binary classification boundary
even if autoregressive free-form generation is degenerate.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

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
import torch.nn.functional as F
from transformers import AutoTokenizer

from data.bigearthnet_txt.dataset import BigEarthNetDataset
from models.rs_internvl.model import RSInternVL
from training.lora import load_lora_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_yes_no")


def evaluate_yes_no_probabilities(
    model: RSInternVL,
    tokenizer: Any,
    dataset: BigEarthNetDataset,
    subset_meta: List[Dict[str, Any]],
    split_name: str,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Evaluate P(YES) vs P(NO) for all samples in a subset."""
    model.eval()
    model.to(device)

    # Token IDs for YES and NO in InternVL / Qwen2 tokenizer
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]   # 9454
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]     # 2753

    logger.info(f"Target Token IDs: YES -> {yes_token_id} ({tokenizer.decode([yes_token_id])!r}), NO -> {no_token_id} ({tokenizer.decode([no_token_id])!r})")

    results = []
    correct_count = 0
    total_binary_count = 0

    with torch.no_grad():
        for i, meta in enumerate(subset_meta):
            sample = dataset[i]
            s1_img = sample["image_s1"].unsqueeze(0).to(device)  # [1, 2, H, W]
            s2_img = sample["image_s2"].unsqueeze(0).to(device)  # [1, 10, H, W]
            query = meta["query"]
            target = meta["target"]
            patch_id = meta["patch_id"]
            task_type = meta.get("claim_type", "binary")

            # Ground truth binary class
            gt_upper = target.strip().upper()
            if gt_upper.startswith("YES"):
                gt_class = "YES"
            elif gt_upper.startswith("NO"):
                gt_class = "NO"
            else:
                gt_class = "OTHER"

            # Prompt representation
            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            encoded = tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Vision encoding
            s1_feat, s2_feat, s1_tok, s2_tok = model.encode_vision(s1_img, s2_img)
            text_embeds = model.language_model.get_input_embeddings()(input_ids)

            # Multimodal fusion
            fused = model.fusion(
                s1_tokens=s1_tok,
                s2_tokens=s2_tok,
                text_embeds=text_embeds,
                text_attention_mask=attention_mask,
            )

            # Forward to get next-token logits
            lm_outputs = model.language_model(
                inputs_embeds=fused.inputs_embeds,
                attention_mask=fused.attention_mask,
                return_dict=True,
            )
            next_logits = lm_outputs.logits[0, -1, :]  # [vocab_size]
            all_probs = F.softmax(next_logits, dim=-1)

            # Absolute probabilities
            p_yes_raw = float(all_probs[yes_token_id].item())
            p_no_raw = float(all_probs[no_token_id].item())

            # Binary normalized probability
            binary_denom = max(1e-12, p_yes_raw + p_no_raw)
            p_yes_norm = p_yes_raw / binary_denom
            p_no_norm = p_no_raw / binary_denom

            # Top token in entire vocab
            top_val, top_id = torch.max(all_probs, dim=-1)
            top_token_str = tokenizer.decode([top_id.item()])
            top_token_prob = float(top_val.item())

            pred_class = "YES" if p_yes_norm >= 0.5 else "NO"
            is_correct = (pred_class == gt_class)

            if gt_class in ("YES", "NO"):
                total_binary_count += 1
                if is_correct:
                    correct_count += 1

            entry = {
                "split": split_name,
                "index": i,
                "patch_id": patch_id,
                "query": query,
                "target": target,
                "ground_truth_class": gt_class,
                "p_yes_raw": round(p_yes_raw, 8),
                "p_no_raw": round(p_no_raw, 8),
                "p_yes_normalized": round(p_yes_norm, 4),
                "p_no_normalized": round(p_no_norm, 4),
                "predicted_class": pred_class,
                "is_correct": is_correct,
                "top_vocab_token": top_token_str,
                "top_vocab_prob": round(top_token_prob, 6),
                "top_vocab_id": int(top_id.item()),
            }
            results.append(entry)

            logger.info(
                "[%s #%d] GT: %-5s | P(YES): %.4f, P(NO): %.4f -> Pred: %-3s (%s) | Top: %r (p=%.4f)",
                split_name,
                i,
                gt_class,
                p_yes_norm,
                p_no_norm,
                pred_class,
                "OK" if is_correct else "ERR",
                top_token_str,
                top_token_prob,
            )

    acc = (correct_count / max(1, total_binary_count)) * 100.0
    logger.info(f"[{split_name}] Binary Accuracy: {correct_count}/{total_binary_count} ({acc:.2f}%)")

    return results


def main():
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)
    ckpt_path = Path("checkpoints/semantic_overfit/best")
    model = load_lora_checkpoint(ckpt_path, device=device)

    # 1. Load subsets
    train_manifest = "data/manifests/manifest_train.jsonl"
    val_manifest = "data/manifests/manifest_validation.jsonl"
    data_root = "data/bigearthnet_txt"

    train_ds = BigEarthNetDataset(
        data_root=data_root,
        manifest_path=train_manifest,
        s1_bands=["VV", "VH"],
        s2_bands=None,
        img_size=120,
        split="train",
        strict=False,
    )
    val_ds = BigEarthNetDataset(
        data_root=data_root,
        manifest_path=val_manifest,
        s1_bands=["VV", "VH"],
        s2_bands=None,
        img_size=120,
        split="validation",
        strict=False,
    )

    # Load 8-sample metadata
    train_meta_file = Path("outputs/semantic_overfit/train_subset.json")
    if train_meta_file.exists():
        with open(train_meta_file, "r", encoding="utf-8") as f:
            train_meta = json.load(f)
    else:
        train_meta = [{"query": train_ds[i]["text"], "target": train_ds[i]["target_text"], "patch_id": train_ds[i]["image_id"], "claim_type": "binary"} for i in range(min(8, len(train_ds)))]

    val_meta = [{"query": val_ds[i]["text"], "target": val_ds[i]["target_text"], "patch_id": val_ds[i]["image_id"], "claim_type": "binary"} for i in range(min(8, len(val_ds)))]

    logger.info("======================================================================")
    logger.info("          STEP 7F: MINIMAL YES/NO DIRECT PROBABILITY AUDIT            ")
    logger.info("======================================================================")

    train_results = evaluate_yes_no_probabilities(model, tokenizer, train_ds, train_meta, "Train", device)
    val_results = evaluate_yes_no_probabilities(model, tokenizer, val_ds, val_meta, "Validation", device)

    all_results = {
        "train": train_results,
        "validation": val_results,
    }

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step7f_yes_no_alignment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Step 7F results to {out_file}")


if __name__ == "__main__":
    main()
