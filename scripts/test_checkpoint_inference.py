#!/usr/bin/env python3
"""
RS-InternVL: Step 5A Checkpoint Inference Verification Script.

Verifies that the saved LoRA checkpoint can be loaded in a fresh, clean Python process
and used for actual inference on real BigEarthNet Sentinel-1 / Sentinel-2 satellite imagery.

Usage:
    python scripts/test_checkpoint_inference.py
    python scripts/test_checkpoint_inference.py --checkpoint checkpoints/lora/best --sample-idx 0
"""

import argparse
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows — prevents cp1252 UnicodeEncodeError when
# the model generates tokens that decode to non-ASCII characters.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from data.bigearthnet_txt.dataset import BigEarthNetDataset
from training.lora import audit_parameters, load_lora_checkpoint
from training.train_lora import get_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_checkpoint_inference")


def verify_checkpoint_inference(
    checkpoint_dir: str = "checkpoints/lora/best",
    manifest_path: str = "data/manifests/manifest_train.jsonl",
    data_root: str = "data/bigearthnet_txt",
    sample_idx: int = 0,
    device_str: str = "cpu",
    query_override: str = None,
) -> dict:
    """
    Verify complete checkpoint loading, parameter audit, and multimodal inference.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists() and (REPO_ROOT / checkpoint_dir).exists():
        ckpt_path = REPO_ROOT / checkpoint_dir

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    device = torch.device(device_str)
    logger.info(f"Target Device: {device}")
    logger.info(f"Loading checkpoint from: {ckpt_path}")

    # 1. Load full model from modular LoRA checkpoint
    model = load_lora_checkpoint(
        checkpoint_dir=ckpt_path,
        device=device,
        is_trainable=True,
    )
    model.eval()

    # 2. Verify Parameter Configuration
    audit = audit_parameters(model)
    logger.info("Verifying parameter configuration...")

    base_llm_frozen = all(
        not p.requires_grad
        for name, p in model.language_model.named_parameters()
        if "lora_" not in name
    )
    lora_trainable = any(
        p.requires_grad
        for name, p in model.language_model.named_parameters()
        if "lora_" in name
    )
    s1_encoder_trainable = all(p.requires_grad for p in model.s1_encoder.parameters())
    s2_encoder_trainable = all(p.requires_grad for p in model.s2_encoder.parameters())
    projections_trainable = (
        all(p.requires_grad for p in model.s1_projection.parameters())
        and all(p.requires_grad for p in model.s2_projection.parameters())
    )

    logger.info(f"  - Base LLM Frozen:          {base_llm_frozen} ({audit['frozen_llm']:,} params)")
    logger.info(f"  - LoRA Trainable:           {lora_trainable} ({audit['lora_trainable']:,} params)")
    logger.info(f"  - S1 Encoder Trainable:     {s1_encoder_trainable} ({audit['s1_encoder_trainable']:,} params)")
    logger.info(f"  - S2 Encoder Trainable:     {s2_encoder_trainable} ({audit['s2_encoder_trainable']:,} params)")
    logger.info(f"  - Projections Trainable:    {projections_trainable} ({audit['projections_trainable']:,} params)")

    assert base_llm_frozen, "Verification Failed: Base LLM is not completely frozen!"
    assert lora_trainable, "Verification Failed: LoRA adapters are not trainable!"
    assert projections_trainable, "Verification Failed: Projections are not trainable!"

    # 3. Load one REAL BigEarthNet sample
    manifest = Path(manifest_path)
    if not manifest.exists() and (REPO_ROOT / manifest_path).exists():
        manifest = REPO_ROOT / manifest_path

    logger.info(f"Loading real sample from manifest: {manifest}")
    # s2_bands=None → MODEL_S2_BANDS (10 bands: B02–B12 excluding B01/B09).
    # This matches the S2Encoder's expected input dimension of 10 channels.
    dataset = BigEarthNetDataset(
        manifest_path=manifest,
        data_root=data_root,
        s2_bands=None,   # → MODEL_S2_BANDS = [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
        is_training=False,
        strict=False,
    )

    if len(dataset) == 0:
        raise ValueError(f"No samples loaded from manifest {manifest}")

    sample = dataset[sample_idx % len(dataset)]
    img_s1 = sample["image_s1"].to(device)
    img_s2 = sample["image_s2"].to(device)
    query = query_override or sample.get("text") or sample.get("text_input") or "Is coniferous forest present in this satellite patch?"
    target_answer = sample.get("target_text") or sample.get("text_output") or ""

    # 4. Tokenizer
    tokenizer = get_tokenizer(
        model_id=model.config.model_id,
        vocab_size=model.config.vocab_size,
    )

    # 5. Execute model.predict() with real generation
    logger.info(f"Running model.predict() with query: '{query}'...")
    pred = model.predict(
        image_s1=img_s1,
        image_s2=img_s2,
        query=query,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=False,
    )

    # 6. Format and print required output
    print("\n" + "=" * 65)
    print("       RS-INTERNVL CHECKPOINT INFERENCE VERIFICATION")
    print("=" * 65)
    print(f"Checkpoint:       {ckpt_path}")
    print(f"Model:            RS-InternVL (Backbone: {model.config.model_id})")
    print(f"S1 shape:         {img_s1.shape}")
    print(f"S2 shape:         {img_s2.shape}")
    print(f"Query:            {query}")
    if target_answer:
        print(f"Target answer:    {target_answer}")
    print(f"Generated answer: {pred['answer']}")
    print(f"Claim:            {pred['claim']}")
    print(f"Claim type:       {pred['claim_type']}")
    print(f"Model score:      {pred['model_score']}")
    print(f"Model version:    {pred['model_version']}")
    print("=" * 65 + "\n")

    return {
        "checkpoint": str(ckpt_path),
        "model": f"RS-InternVL (Backbone: {model.config.model_id})",
        "s1_shape": str(img_s1.shape),
        "s2_shape": str(img_s2.shape),
        "query": query,
        "generated_answer": pred["answer"],
        "claim": pred["claim"],
        "claim_type": pred["claim_type"],
        "model_score": pred["model_score"],
        "model_version": pred["model_version"],
    }


def main():
    parser = argparse.ArgumentParser(description="Test RS-InternVL checkpoint inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/lora/best", help="Path to checkpoint")
    parser.add_argument("--manifest", type=str, default="data/manifests/manifest_train.jsonl", help="Path to manifest")
    parser.add_argument("--data-root", type=str, default="data/bigearthnet_txt", help="Data root directory")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to test")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu', 'cuda')")
    parser.add_argument("--query", type=str, default=None, help="Custom query string")

    args = parser.parse_args()

    verify_checkpoint_inference(
        checkpoint_dir=args.checkpoint,
        manifest_path=args.manifest,
        data_root=args.data_root,
        sample_idx=args.sample_idx,
        device_str=args.device,
        query_override=args.query,
    )


if __name__ == "__main__":
    main()
