"""
STEP 7C: Real S1/S2 Multimodal Generation Feature Test

Uses one real validation sample to evaluate:
1. Text only
2. S1 only (SAR features only)
3. S2 only (Optical features only)
4. S1 + S2 multimodal fusion

Reports:
- visual embedding shape, mean, std, min, max, NaN/Inf count
- generated output string, first generated token, generation length
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
logger = logging.getLogger("debug_real_multimodal")


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
        "Condition: %-20s | Shape: %-16s | Mean: %+.4f, Std: %.4f | Generated: %r (Len: %d, First: %r)",
        condition_name,
        str(stats["shape"]),
        stats["mean"] if stats["mean"] is not None else 0.0,
        stats["std"] if stats["std"] is not None else 0.0,
        decoded_text,
        len(generated_ids),
        first_token_str,
    )
    return result


def main():
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)
    ckpt_path = Path("checkpoints/semantic_overfit/best")
    model = load_lora_checkpoint(ckpt_path, device=device)

    # Load 1 real sample from validation dataset
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

    logger.info(f"Sample Patch ID: {sample['image_id']}")
    logger.info(f"Query:           {query}")
    logger.info(f"Target:          {target}")

    with torch.no_grad():
        # Encode vision
        s1_feat = model.s1_encoder(s1_tensor)             # [1, 225, 512]
        s1_proj = model.s1_projection(s1_feat)           # [1, 225, 896]

        s2_feat = model.s2_encoder(s2_tensor)             # [1, 225, 768]
        s2_proj = model.s2_projection(s2_feat)           # [1, 225, 896]

        fused = torch.cat([s1_proj, s2_proj], dim=1)      # [1, 450, 896]

    conditions = [
        ("1_Text_Only", None),
        ("2_S1_Only", s1_proj),
        ("3_S2_Only", s2_proj),
        ("4_S1_S2_Fusion", fused),
    ]

    results = []
    for cond_name, vis_tensor in conditions:
        res = generate_with_visual_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt=query,
            vis_tokens=vis_tensor,
            condition_name=cond_name,
            device=device,
        )
        results.append(res)

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step7c_real_multimodal_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Step 7C results to {out_file}")


if __name__ == "__main__":
    main()
