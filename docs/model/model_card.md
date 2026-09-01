# RS-InternVL Model Card & Technical Specification

**System:** SATQuery AI — Problem Statement 167 (SIH 2026)  
**Member:** Member 1 (Core VLM & Data Engineering)  
**Model Name:** `RS-InternVL`  
**Base Backbone:** `OpenGVLab/InternVL3-1B`  
**Best Checkpoint:** [`checkpoints/pretrained_lora/best`](../../checkpoints/pretrained_lora/best)  

---

## 1. Model Overview & Purpose

`RS-InternVL` is a multi-sensor remote sensing vision-language model (VLM) engineered for natural language grounding and question answering over paired Earth observation data. It directly ingests and fuses co-registered **Sentinel-1 SAR** (Synthetic Aperture Radar) and **Sentinel-2 Multispectral** imagery to perform land cover classification, environmental claim verification, and terrain understanding.

---

## 2. Dataset Engineering (`data/bigearthnet_txt/`)

- **Dataset Source:** BigEarthNet.txt multimodal remote sensing benchmark.
- **Manifests:**
  - Train: [`data/manifests/manifest_train.jsonl`](../../data/manifests/manifest_train.jsonl) — **32 real samples**
  - Validation: [`data/manifests/manifest_validation.jsonl`](../../data/manifests/manifest_validation.jsonl) — **8 real samples**
- **Data Integrity:**
  - Patch leakage between Train and Validation: **0 (Strictly Verified)**
  - No synthetic data generation or duplicate sample inflation.

---

## 3. Sensor Modality Specifications

### Sentinel-1 SAR Branch
- **Channels:** 2 polarizations (`VV`, `VH`)
- **Spatial Resolution:** 10 meters / pixel
- **Input Dimensions:** `[B, 2, 120, 120]`
- **Encoder Architecture:** 4-stage residual convolutional network $\rightarrow$ `[B, 225, 512]`
- **Projection Layer:** Two-layer MLP with LayerNorm + GELU $\rightarrow$ `[B, 225, 896]`

### Sentinel-2 Multispectral Branch
- **Channels:** 10 spectral bands (`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11`, `B12`)
- **Spatial Resolution:** 10m and 20m resampled to 10m grid ($120 \times 120$ pixels)
- **Excluded Bands:** `B01` (Coastal Aerosol, 60m) and `B09` (Water Vapour, 60m)
- **Input Dimensions:** `[B, 10, 120, 120]`
- **Encoder Architecture:** 10-channel patch embedding ($8 \times 8$ stride) + Transformer blocks $\rightarrow$ `[B, 225, 768]`
- **Projection Layer:** Two-layer MLP with LayerNorm + GELU $\rightarrow$ `[B, 225, 896]`

---

## 4. Language Backbone & Parameter Allocation

- **Language Backbone:** `OpenGVLab/InternVL3-1B` (Qwen2 architecture: 24 layers, 896 hidden dim, 151674 vocab size).
- **Adaptation Strategy:** Parameter-Efficient Fine-Tuning (PEFT) via **LoRA**:
  - Rank $r = 8$, Alpha $\alpha = 32$, Dropout $= 0.1$, Bias = `none`
  - Target Modules: `q_proj`, `v_proj` in causal self-attention layers
- **Parameter Breakdown:**

| Component | Parameter Count | Trainable? |
|---|---|---|
| Base Qwen2 Language Backbone | 629,697,920 | **FROZEN** |
| LoRA Attention Adapters | 540,672 | **TRAINABLE** |
| S1 SAR Encoder | 1,283,072 | **TRAINABLE** |
| S2 Multispectral Encoder | 14,842,368 | **TRAINABLE** |
| Modality Projections + Fusion | 3,153,664 | **TRAINABLE** |
| **Total Model Parameters** | **649,517,696** | — |
| **Total Trainable Parameters** | **19,819,776 (3.05%)** | — |

---

## 5. Training & Optimization Policy

- **Loss Function:** Masked Cross-Entropy (Visual tokens assigned `-100` ignore index).
- **Optimizer:** `AdamW` ($\text{weight\_decay} = 0.01, \text{lr} = 1 \times 10^{-4}$).
- **Scheduler:** Cosine annealing with 8 linear warmup steps.
- **Gradient Accumulation:** 4 steps (Effective Batch Size = 4).
- **Precision:** Float32.
- **Gradient Clipping:** Max norm 1.0.

---

## 6. Verified Empirical Results

### Step 9: Semantic Overfit Validation (8 Train / 8 Val)
- **Train Binary Accuracy:** `85.71%`
- **Val Binary Accuracy:** `71.43%`
- **Garbage / Noise Generation Rate:** `0.0%`
- **Generation Validity:** `100.0%`

### Step 10: Full Available Manifest Training (32 Train / 8 Val)
- **Total Training Duration:** 112.4 minutes (25 epochs on CPU)
- **Loss Progression:**
  - Baseline Epoch 0 Val Loss: `7.2178`
  - Epoch 1 Train Loss: `5.8518` $\rightarrow$ Val Loss: `3.8760`
  - **Epoch 10 Best Val Loss:** **`1.1962`**
  - **Epoch 25 Final Train Loss:** **`0.0012`** (**-99.98% drop**)
- **Binary Presence/Absence Metrics:**
  - **Train:** **`100.0%` Accuracy**, **`100.0%` Precision**, **`100.0%` Recall**, **`100.0%` F1**
  - **Validation:** **`71.43%` Accuracy**, **`80.0%` Precision**, **`80.0%` Recall**, **`80.0%` F1**
- **Multiple Choice Land Cover Classification:**
  - **Train:** 1/1 correct (**`100.0%`**)
- **Generation Quality at Best Checkpoint (Epoch 10):**
  - Train Garbage Rate: **`0.0%`**
  - Validation Garbage Rate: **`0.0%`**
  - Valid English Sentence Output: **`100.0%`**
- **Test Suite Status:** **97/97 Regression Tests Passed**.

---

## 7. Structured Inference Interface

The model outputs the standard JSON schema:
```json
{
  "answer": "Yes, broad-leaved forest is present.",
  "claim": "Multi-modal SAR (VV/VH) and Optical (10 bands) query: Is broad-leaved forest present in this area?",
  "claim_type": "presence_verification",
  "model_score": 0.9654,
  "model_version": "RS-InternVL3-1B-LoRA (backbone: OpenGVLab/InternVL3-1B)",
  "grounding": null
}
```

---

## 8. Known Limitations & Constraints

1. **Dataset Scope:** The current verified dataset consists of the available 32 train and 8 validation real BigEarthNet patches. While loss convergence and semantic overfit are strictly proven, full BigEarthNet generalization across European ecoregions will require scaling the patch count during Phase 2.
2. **Execution Latency on CPU:** Autoregressive generation on CPU averages ~4 seconds per query. CUDA GPU acceleration is strongly recommended for large-batch evaluation.
3. **Task Imbalance:** The current manifest is predominantly binary presence queries (38 samples) with 2 MCQ queries. Balanced task sampling should be configured for downstream fine-tuning.

---

## 9. Reproducibility Instructions

### Load Best Checkpoint & Run Inference
```python
from models.rs_internvl.config import RSInternVLConfig
from training.lora import load_lora_checkpoint
from transformers import AutoTokenizer

cfg = RSInternVLConfig(
    model_id="OpenGVLab/InternVL3-1B",
    pretrained_backbone=True,
    img_size=120,
    s1_channels=2,
    s2_channels=10,
    freeze_llm=True,
)
model = load_lora_checkpoint(
    checkpoint_dir="checkpoints/pretrained_lora/best",
    config_override=cfg,
    device="cpu",
    is_trainable=False,
)
tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)
result = model.predict(image_s1=s1_tensor, image_s2=s2_tensor, query="Is forest present?", tokenizer=tokenizer)
print(result)
```

### Run Verification Suite
```bash
.venv/Scripts/pytest.exe tests/ -v
```
