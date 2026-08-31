"""
STEP 7D: Tokenization and Label Masking Audit

Audits the exact tokenization, prompt formatting, BOS/EOS handling,
and label masking (-100 on visual prefix + prompt, unmasked target completion).
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

from training.train_lora import MultimodalCollate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("audit_tokenization")

TARGET_EXAMPLES = [
    ("Is coniferous forest present in this satellite patch?", "Yes, coniferous forest is present."),
    ("Is water body present in this area?", "No, water body is not present."),
    ("Is pastures land cover present in this scene?", "Yes, pastures are present in the patch."),
]


def audit_sample_tokenization(
    query: str,
    target: str,
    tokenizer: Any,
    collate_fn: MultimodalCollate,
) -> Dict[str, Any]:
    """Audit tokenization and collation masking for a single query-target pair."""
    # 1. Direct target tokenization
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    target_tokens = [tokenizer.decode([tid]) for tid in target_ids]

    # 2. Query tokenization
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    query_tokens = [tokenizer.decode([qid]) for qid in query_ids]

    # 3. Simulate collation as done during training in MultimodalCollate
    raw_sample = {
        "image_s1": torch.zeros(2, 120, 120),
        "image_s2": torch.zeros(10, 120, 120),
        "text": query,
        "target_text": target,
        "image_id": "test_patch_001",
        "task": "binary",
    }
    batch = collate_fn([raw_sample])

    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    attention_mask = batch["attention_mask"][0].tolist()

    full_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]

    # Unmasked label positions
    unmasked_indices = [i for i, lbl in enumerate(labels) if lbl != -100]
    unmasked_label_ids = [labels[i] for i in unmasked_indices]
    unmasked_label_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in unmasked_label_ids]
    unmasked_text = tokenizer.decode(unmasked_label_ids, skip_special_tokens=False)

    masked_indices = [i for i, lbl in enumerate(labels) if lbl == -100]
    masked_input_ids = [input_ids[i] for i in masked_indices]
    masked_prompt_text = tokenizer.decode(masked_input_ids, skip_special_tokens=False)

    # Check EOS presence
    eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|im_end|>")
    has_eos_in_unmasked = (eos_id in unmasked_label_ids)

    audit_entry = {
        "query": query,
        "raw_target": target,
        "target_token_ids": target_ids,
        "target_tokens": target_tokens,
        "target_token_count": len(target_ids),
        "first_target_token_id": target_ids[0] if target_ids else None,
        "first_target_token_str": target_tokens[0] if target_tokens else None,
        "total_seq_length": len(input_ids),
        "num_masked_tokens": len(masked_indices),
        "num_unmasked_tokens": len(unmasked_indices),
        "masked_prompt_text": masked_prompt_text,
        "unmasked_label_tokens": unmasked_label_tokens,
        "unmasked_decoded_text": unmasked_text,
        "has_eos_in_labels": has_eos_in_unmasked,
        "eos_token_id": eos_id,
        "labels_match_input_ids_at_unmasked": all(input_ids[i] == labels[i] for i in unmasked_indices),
    }

    logger.info("----------------------------------------------------------------------")
    logger.info("Query:              %s", query)
    logger.info("Target:             %s", target)
    logger.info("Target Token IDs:   %s", target_ids)
    logger.info("Target Tokens:      %s", target_tokens)
    logger.info("Unmasked Count:     %d (Masked: %d, Total: %d)", len(unmasked_indices), len(masked_indices), len(input_ids))
    logger.info("Unmasked Decoded:   %r", unmasked_text)
    logger.info("Has EOS in Labels:  %s (EOS ID: %d)", has_eos_in_unmasked, eos_id)
    logger.info("Labels Match Inputs:%s", audit_entry["labels_match_input_ids_at_unmasked"])

    return audit_entry


def main():
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)
    collate_fn = MultimodalCollate(tokenizer=tokenizer, max_seq_length=512)

    audits = []
    for query, target in TARGET_EXAMPLES:
        entry = audit_sample_tokenization(query, target, tokenizer, collate_fn)
        audits.append(entry)

    out_dir = Path("outputs/debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "step7d_tokenization_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Step 7D tokenization audit to {out_file}")


if __name__ == "__main__":
    main()
