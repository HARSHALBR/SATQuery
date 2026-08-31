"""
STEP 7B: Dummy Visual Token Test

Tests generation with controlled dummy visual prefixes:
A. No visual tokens
B. Zero visual embeddings ([1, 450, 896])
C. Small random visual embeddings ([1, 450, 896], std=0.02)
D. Constant visual embeddings ([1, 450, 896], val=0.1)

Compares generated outputs under identical prompt and decoding settings.
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
from transformers import AutoTokenizer

from models.rs_internvl.config import RSInternVLConfig
from models.rs_internvl.model import RSInternVL
from training.lora import load_lora_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("debug_dummy_visual")

PROMPT = "Is coniferous forest present in this satellite patch?"


def generate_with_prefix(
    model: RSInternVL,
    tokenizer: Any,
    prompt: str,
    visual_prefix: torch.Tensor = None,  # [1, N_vis, hidden_dim] or None
    condition_name: str = "no_visual",
    device: str = "cpu",
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    """Generate tokens from prompt with optional visual token prefix."""
    model.eval()
    model.to(device)

    eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    encoded = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    text_mask = encoded["attention_mask"].to(device)

    # 1. Text embedding
    text_embeds = model.language_model.get_input_embeddings()(input_ids)

    # 2. Multimodal fusion
    if visual_prefix is not None:
        vis_embeds = visual_prefix.to(device)
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

    result = {
        "condition": condition_name,
        "prefix_shape": list(visual_prefix.shape) if visual_prefix is not None else None,
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
        "Condition: %-25s -> Generated: %r (Len: %d, First: %r, EOS: %s)",
        condition_name,
        decoded_text,
        len(generated_ids),
        first_token_str,
        result["eos_behavior"],
    )
    return result


def main():
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)
    ckpt_path = Path("checkpoints/semantic_overfit/best")
    model = load_lora_checkpoint(ckpt_path, device=device)

    hidden_dim = 896
    num_vis_tokens = 450  # 225 (S1) + 225 (S2)

    conditions = [
        ("A_No_Visual_Tokens", None),
        ("B_Zero_Embeddings", torch.zeros(1, num_vis_tokens, hidden_dim)),
        ("C_Small_Random_Gaussian", torch.randn(1, num_vis_tokens, hidden_dim) * 0.02),
        ("D_Constant_Embeddings", torch.ones(1, num_vis_tokens, hidden_dim) * 0.1),
    ]

    results = []
    for cond_name, prefix_tensor in conditions:
        res = generate_with_prefix(
            model=model,
            tokenizer=tokenizer,
            prompt=PROMPT,
            visual_prefix=prefix_tensor,
            condition_name=cond_name,
            device=device,
        )
        results.append(res)

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step7b_dummy_visual_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Step 7B results to {out_file}")


if __name__ == "__main__":
    main()
