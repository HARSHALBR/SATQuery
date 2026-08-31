# RS-InternVL Tiny-Subset Overfitting & Training Pipeline

## Purpose of Tiny-Subset Overfit

Before undertaking full-scale parameter-efficient fine-tuning (LoRA) or downstream benchmark evaluation on thousands of multi-sensor satellite patches, this experiment serves as a **critical architecture validation gate (Step 3)**.

The primary objective is to prove that the complete multimodal RS-InternVL architecture is capable of gradient-based learning by deliberately overfitting a deterministic, tiny subset of the BigEarthNet.txt training split (8, 16, or 32 samples).

This verifies:
1. **End-to-End Multimodal Gradient Flow**: Gradients successfully propagate from the causal language modeling head back through the Qwen2/InternVL backbone, the modality projection layers (`S1Projection` and `S2Projection`), and the spatial modality encoders (`S1Encoder` and `S2Encoder`).
2. **Causal Attention & Sequence Alignment**: The unified sequence `[S1 visual tokens (225)] + [S2 visual tokens (225)] + [text query tokens] + [target answer tokens]` processes attention masks and position IDs without index out-of-bounds or shape collapses.
3. **Loss Masking Integrity**: Visual token positions (and instruction query prompts) are assigned label ID `-100` (PyTorch CrossEntropyLoss `ignore_index`). Only the target text tokens generate loss gradients.
4. **Optimization Stability**: Loss decreases steadily across training epochs without encountering numerical instability (zero NaNs, zero Infs, bounded gradient norms).

---

## Configuration (`training/config.yaml`)

```yaml
dataset:
  train_manifest: "data/manifests/manifest_train.jsonl"
  data_root: "data/bigearthnet_txt"
  num_samples: 16       # Options: 8, 16, 32
  seed: 42             # Deterministic subset seed
  img_size: 120
  s1_bands: ["VV", "VH"]
  s2_bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

model:
  checkpoint: null
  freeze_s1_encoder: false   # Trainable for feasibility
  freeze_s2_encoder: false   # Trainable for feasibility
  freeze_llm: false          # Trainable for feasibility
  num_hidden_layers: 4       # Scalable depth for rapid test
  num_attention_heads: 14
  num_key_value_heads: 2
  llm_hidden_dim: 896
  vocab_size: 151674

training:
  batch_size: 2
  epochs: 20
  learning_rate: 0.0001
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  max_seq_length: 512
  mixed_precision: "no"      # Options: "fp16", "bf16", "no"
  seed: 42
  log_every_n_steps: 1
  clip_grad_norm: 1.0
  save_best: true

output:
  checkpoint_dir: "checkpoints/tiny_overfit"
  log_dir: "outputs/tiny_overfit"
```

---

## Commands & Usage

### 1. Default Run with Config File
```bash
python training/train_tiny.py --config training/config.yaml
```

### 2. Override Subset Size and Epochs
```bash
python training/train_tiny.py \
    --config training/config.yaml \
    --num-samples 16 \
    --epochs 20
```

### 3. Quick Smoke Test Script
```bash
python scripts/test_tiny_training.py
```

### 4. Running the Test Suite
```bash
pytest tests/ -v
```

---

## Expected Behavior

- **Device**: Automatic CPU/CUDA device detection. CUDA mixed precision (`fp16`/`bf16`) is automatically utilized when available.
- **Subset Sampling**: Deterministic selection of exactly `num_samples` from `split="train"` using fixed `seed=42`. Validation/test splits are strictly excluded.
- **Loss Trajectory**: A distinct and consistent downward trend in training loss from epoch 1 to final epoch.
- **Gradients**: Gradient norms remain finite ($> 0.0$ and $< \infty$) across all trainable layers.
- **Checkpoints**: `best_checkpoint.pt` and `final_checkpoint.pt` are saved to the designated checkpoint directory alongside the serialized configuration snapshot and `metrics.json`.

---

## Interpretation of Convergence vs. Failure

| Status / Indicator | Diagnosis | Action |
| :--- | :--- | :--- |
| **Loss strictly decreases** (e.g. 10.0 $\to$ 0.5) | **Passed**: Architecture is learning properly | Proceed to Step 4 (LoRA Adaptation). |
| **Loss remains flat** ($\Delta \text{loss} \approx 0$) | **Optimization issue**: Learning rate too low, weights frozen unintentionally, or all labels masked to `-100`. | Check `param.requires_grad`, learning rate schedule, and text target tokenization. |
| **Loss is NaN or Inf** | **Numerical instability**: Learning rate too high, gradient explosion, or unnormalized input tensors. | Check `MultiBandNormalize`, enable gradient clipping (`clip_grad_norm: 1.0`), reduce learning rate. |
| **Shape mismatch on fusion** | **Dimension error**: S1/S2 projection output dimensions do not equal `llm_hidden_dim`. | Verify projection layer `out_dim` in `models/rs_internvl/projection.py`. |
| **Index out of range in embedding** | **Vocabulary mismatch**: Text token IDs exceed `vocab_size`. | Verify tokenizer max ID against `config.vocab_size` (InternVL3-1B uses 151674). |
