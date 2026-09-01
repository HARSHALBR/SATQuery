"""
STEP 8D: Text-Only Baseline Generation with Pretrained Language Backbone

Tests text-only generation on simple and conversational prompts using the authentic
pretrained InternVL3-1B language model backbone.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("debug_text_only")

PROMPTS = [
    "What is the capital of France?",
    "Answer yes or no: Is water present?",
    "Question: Is forest present? Answer:",
    "Answer: Yes, forest is present.",
    "Is coniferous forest present in this satellite patch?",
    "What is the dominant land cover class?",
]


def test_text_only_generation(
    model: RSInternVL,
    tokenizer: Any,
    prompts: List[str],
    model_name: str,
    device: str = "cpu",
    max_new_tokens: int = 32,
) -> List[Dict[str, Any]]:
    """Test text-only generation without any visual inputs."""
    model.eval()
    model.to(device)
    results = []

    eos_token_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")

    for prompt in prompts:
        # 1. Format prompt in standard InternVL chat format
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        encoded = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # 2. Text-only embeddings
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)

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

        res = {
            "model_name": model_name,
            "prompt": prompt,
            "formatted_prompt": formatted_prompt,
            "generated_text": decoded_text,
            "token_ids": generated_ids,
            "token_strings": token_strs,
            "first_token_id": first_token_id,
            "first_token_str": first_token_str,
            "eos_behavior": "EOS_REACHED" if eos_reached else "MAX_TOKENS_REACHED",
            "generation_length": len(generated_ids),
        }
        results.append(res)
        logger.info(
            "[%s] Prompt: %r -> Generated: %r (Len: %d, First: %r, EOS: %s)",
            model_name,
            prompt,
            decoded_text,
            len(generated_ids),
            first_token_str,
            res["eos_behavior"],
        )

    return results


def main():
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)

    # 1. Base RSInternVL with Authentic Pretrained Language Backbone
    cfg = RSInternVLConfig(pretrained_backbone=True)
    logger.info("--- Testing RSInternVL with Pretrained Language Backbone (Text Only) ---")
    model = RSInternVL(cfg)
    results = test_text_only_generation(model, tokenizer, PROMPTS, "Pretrained_RSInternVL_Backbone", device)

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step8d_text_only_pretrained_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Step 8D results to {out_file}")


if __name__ == "__main__":
    main()
