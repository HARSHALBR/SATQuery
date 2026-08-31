"""
STEP 8E & 8G: Clean Pretrained Baseline & Multimodal Interface Audit

Evaluates the un-finetuned authentic pretrained RS-InternVL model across:
1. Text only
2. Text + Zero visual embeddings
3. Text + Random visual embeddings
4. Text + Real S1 features
5. Text + Real S2 features
6. Text + Real S1 + S2 multimodal fusion
7. Candidate YES/NO token probabilities
8. Parameter freezing & trainability audit
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.lora import apply_lora, audit_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pretrained_baseline")


def compute_tensor_stats(t: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Compute tensor statistics."""
    if t is None:
        return {
            "shape": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "nan_count": 0,
            "inf_count": 0,
        }

    t_float = t.detach().float()
    return {
        "shape": list(t.shape),
        "mean": round(float(torch.mean(t_float).item()), 6),
        "std": round(float(torch.std(t_float).item()), 6),
        "min": round(float(torch.min(t_float).item()), 6),
        "max": round(float(torch.max(t_float).item()), 6),
        "nan_count": int(torch.isnan(t_float).sum().item()),
        "inf_count": int(torch.isinf(t_float).sum().item()),
    }


def generate_with_visual_tokens(
    model: RSInternVL,
    tokenizer: Any,
    prompt: str,
    vis_tokens: Optional[torch.Tensor],  # [1, N_vis, hidden_dim]
    condition_name: str,
    device: str = "cpu",
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Generate tokens from prompt with given visual tokens."""
    model.eval()
    model.to(device)

    eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    text_mask = encoded["attention_mask"].to(device)

    # 1. Text embedding
    text_embeds = model.language_model.get_input_embeddings()(input_ids)

    # 2. Multimodal concatenation
    if vis_tokens is not None:
        vis_embeds = vis_tokens.to(device)
        vis_mask = torch.ones((1, vis_embeds.shape[1]), dtype=text_mask.dtype, device=device)
        inputs_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
        attention_mask = torch.cat([vis_mask, text_mask], dim=1)
    else:
        inputs_embeds = text_embeds
        attention_mask = text_mask

    generated_ids: List[int] = []
    token_strs: List[str] = []
    past_key_values = None
    cur_embeds = inputs_embeds
    cur_mask = attention_mask

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model.language_model(
                inputs_embeds=cur_embeds,
                attention_mask=cur_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            tid = next_token_id.item()

            generated_ids.append(tid)
            token_strs.append(tokenizer.decode([tid], skip_special_tokens=False))

            if tid == eos_token_id:
                break

            cur_embeds = model.language_model.get_input_embeddings()(next_token_id)
            if cur_mask is not None:
                next_mask = torch.ones((1, 1), dtype=cur_mask.dtype, device=device)
                cur_mask = torch.cat([cur_mask, next_mask], dim=1)

    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    first_token_str = token_strs[0] if token_strs else ""
    first_token_id = generated_ids[0] if generated_ids else None
    eos_reached = (generated_ids[-1] == eos_token_id) if generated_ids else False

    stats = compute_tensor_stats(vis_tokens)

    result = {
        "condition": condition_name,
        "embedding_stats": stats,
        "prompt": prompt,
        "generated_text": decoded_text,
        "token_ids": generated_ids,
        "token_strings": token_strs,
        "first_token_id": first_token_id,
        "first_token_str": first_token_str,
        "eos_behavior": "EOS_REACHED" if eos_reached else "MAX_TOKENS_REACHED",
        "generation_length": len(generated_ids),
    }

    logger.info(
        "Condition: %-25s | Stats: Mean=%+.4f, Std=%.4f, NaNs=%d | Generated: %r (Len: %d, First: %r, EOS: %s)",
        condition_name,
        stats["mean"] if stats["mean"] is not None else 0.0,
        stats["std"] if stats["std"] is not None else 0.0,
        stats["nan_count"],
        decoded_text,
        len(generated_ids),
        first_token_str,
        result["eos_behavior"],
    )
    return result


def main():
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)

    # 1. Instantiate clean model with authentic pretrained language backbone
    cfg = RSInternVLConfig(pretrained_backbone=True)
    base_model = RSInternVL(cfg)

    # 2. Apply LoRA adaptation
    model, param_audit = apply_lora(base_model, r=8, lora_alpha=32, lora_dropout=0.1)

    logger.info("======================================================================")
    logger.info("       STEP 8: PRETRAINED RS-INTERNVL MULTIMODAL BASELINE AUDIT       ")
    logger.info("======================================================================")

    # 3. Load 1 real sample from validation dataset
    val_manifest = "data/manifests/manifest_validation.jsonl"
    ds = BigEarthNetDataset(
        manifest_path=val_manifest,
        data_root="data/bigearthnet_txt",
        img_size=120,
        strict=False,
    )
    sample = ds[0]
    s1_tensor = sample["image_s1"].unsqueeze(0).to(device)  # [1, 2, 120, 120]
    s2_tensor = sample["image_s2"].unsqueeze(0).to(device)  # [1, 10, 120, 120]
    query = sample["text"]
    target = sample["target_text"]

    hidden_dim = cfg.llm_hidden_dim
    num_vis_tokens = 450

    with torch.no_grad():
        s1_feat = model.s1_encoder(s1_tensor)
        s1_proj = model.s1_projection(s1_feat)
        s2_feat = model.s2_encoder(s2_tensor)
        s2_proj = model.s2_projection(s2_feat)
        fused = torch.cat([s1_proj, s2_proj], dim=1)

    conditions = [
        ("1_Text_Only", None),
        ("2_Zero_Visual_Tokens", torch.zeros(1, num_vis_tokens, hidden_dim)),
        ("3_Random_Visual_Tokens", torch.randn(1, num_vis_tokens, hidden_dim) * 0.02),
        ("4_Real_S1_Only", s1_proj),
        ("5_Real_S2_Only", s2_proj),
        ("6_Real_S1_S2_Fusion", fused),
    ]

    multimodal_results = []
    for cond_name, vis_tensor in conditions:
        res = generate_with_visual_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt=query,
            vis_tokens=vis_tensor,
            condition_name=cond_name,
            device=device,
        )
        multimodal_results.append(res)

    # 4. Candidate YES/NO token probabilities on the pretrained baseline
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]   # 9454
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]     # 2753

    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        fused_seq = model.fusion(
            s1_tokens=s1_proj,
            s2_tokens=s2_proj,
            text_embeds=text_embeds,
            text_attention_mask=attention_mask,
        )
        lm_outputs = model.language_model(
            inputs_embeds=fused_seq.inputs_embeds,
            attention_mask=fused_seq.attention_mask,
            return_dict=True,
        )
        next_logits = lm_outputs.logits[0, -1, :]
        probs = F.softmax(next_logits, dim=-1)
        p_yes_raw = float(probs[yes_token_id].item())
        p_no_raw = float(probs[no_token_id].item())
        binary_denom = max(1e-12, p_yes_raw + p_no_raw)
        p_yes_norm = p_yes_raw / binary_denom
        p_no_norm = p_no_raw / binary_denom

        top_prob, top_idx = torch.max(probs, dim=-1)
        top_token_str = tokenizer.decode([top_idx.item()])

    yes_no_audit = {
        "query": query,
        "target": target,
        "p_yes_raw": p_yes_raw,
        "p_no_raw": p_no_raw,
        "p_yes_normalized": round(p_yes_norm, 4),
        "p_no_normalized": round(p_no_norm, 4),
        "top_token": top_token_str,
        "top_token_prob": round(float(top_prob.item()), 6),
    }

    logger.info("----------------------------------------------------------------------")
    logger.info("YES/NO Baseline: P(YES)=%.4f, P(NO)=%.4f | Top Token: %r (p=%.4f)", p_yes_norm, p_no_norm, top_token_str, float(top_prob.item()))
    logger.info("----------------------------------------------------------------------")

    full_report = {
        "model_id": cfg.model_id,
        "parameter_audit": param_audit,
        "multimodal_conditions": multimodal_results,
        "yes_no_audit": yes_no_audit,
    }

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step8_pretrained_baseline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Step 8 baseline results to {out_file}")


if __name__ == "__main__":
    main()
