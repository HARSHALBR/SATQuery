# SATQuery AI — Member 1: Core VLM & Data Engineering Module

**SIH 2026 — Problem Statement 167 (SIH26167)**  
**Component:** Multi-Modal Remote Sensing Vision-Language Model (`RS-InternVL`)  
**Status:** **Step 10 Complete & Verified (PASS)**  

---

## 1. Overview & Responsibilities

This module constitutes the core multi-sensor Earth observation data ingestion, VLM architecture, PEFT/LoRA training, and structured inference pipeline for **SATQuery AI**.

**Official Member 1 Ownership:**
- **Data Engineering:** Ingestion, band extraction, and spatial co-registration of **Sentinel-1 SAR** (2 polarizations: `VV`, `VH`) and **Sentinel-2 Multispectral** (10 optical bands: `B02`–`B12`).
- **Manifests & Splits:** Deterministic manifest generation ([`data/manifests/`](data/manifests/)) with verified **zero patch overlap**.
- **Model Architecture:** Multi-sensor [`RS-InternVL`](models/rs_internvl/) leveraging an authentic frozen `OpenGVLab/InternVL3-1B` language backbone.
- **Efficient Fine-Tuning:** PEFT / LoRA adaptation ($r=8, \alpha=32$) on attention projections.
- **Training Pipeline:** Modular training loop with gradient accumulation, cosine scheduler, and checkpoint management.
- **Evaluation & Benchmarks:** Structured JSON output schema, BigEarthNet benchmark evaluation, and benchmark interface stubs for VRSBench and RSVQA ([`evaluation/vlm/`](evaluation/vlm/)).

---

## 2. Repository Layout

```text
.
├── configs/
│   └── model/              # YAML configurations (Step 10: pretrained_full_manifest.yaml)
├── data/
│   ├── bigearthnet_txt/    # Sentinel-1 and Sentinel-2 GeoTIFF imagery & text targets
│   └── manifests/          # Deterministic JSONL manifests (manifest_train.jsonl, manifest_validation.jsonl)
├── models/
│   └── rs_internvl/        # RS-InternVL model (S1/S2 encoders, projections, fusion, wrapper)
├── training/               # LoRA adaptation, audit utils, and trainer modules
├── evaluation/
│   └── vlm/                # VRSBench and RSVQA evaluation interfaces
├── docs/
│   └── model/              # Architecture spec (architecture.md) & Model Card (model_card.md)
├── scripts/                # Training, evaluation, preflight, and reload verification scripts
├── tests/                  # Pytest regression suite (100+ tests)
├── checkpoints/
│   └── pretrained_lora/    # Best trained LoRA checkpoint (best/)
└── outputs/                # Training logs, JSON metrics, and markdown reports
```

---

## 3. Environment Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Preparation & Manifests

BigEarthNet.txt samples are paired and mapped deterministically into training and validation splits:
```bash
# Generate deterministic manifests with zero patch overlap
.venv/Scripts/python.exe scripts/build_manifests.py
```
- **Train split:** 32 real samples (`data/manifests/manifest_train.jsonl`)
- **Validation split:** 8 real samples (`data/manifests/manifest_validation.jsonl`)
- **Patch overlap:** 0 (strictly verified)

---

## 5. Model Architecture (`RS-InternVL`)

```text
Sentinel-1 SAR [B, 2, 120, 120]  ──► S1Encoder (CNN) ──► S1Projection (512->896) ──┐
                                                                                   ├──► MultimodalFusion ──► Frozen InternVL3-1B ──► Output Text
Sentinel-2 MS  [B, 10, 120, 120] ──► S2Encoder (ViT) ──► S2Projection (768->896) ──┘    (+ Trainable LoRA)
```

- **Base Language Backbone:** `OpenGVLab/InternVL3-1B` (629.7M params, **FROZEN**)
- **LoRA Adapters:** $r=8, \alpha=32$ on `q_proj` / `v_proj` (540K params, **TRAINABLE**)
- **S1 SAR Encoder:** 2-channel input $\rightarrow$ 512 dim (1.28M params, **TRAINABLE**)
- **S2 Optical Encoder:** 10-channel input $\rightarrow$ 768 dim (14.84M params, **TRAINABLE**)
- **Total Trainable:** 19,819,776 params (3.05% of 649.5M total)

---

## 6. Training & Evaluation Commands

### Run Full Manifest GPU/CPU Training (Step 10 Baseline)
```bash
.venv/Scripts/python.exe scripts/pretrained_full_manifest_training.py --config configs/model/pretrained_full_manifest.yaml
```

### Run Checkpoint Reload & Inference Test
```bash
.venv/Scripts/python.exe scripts/test_step10_reload.py
```

### Checkpoint Location
- Best trained checkpoint: [`checkpoints/pretrained_lora/best`](checkpoints/pretrained_lora/best) (~75.6 MB modular weights)

---

## 7. Python Inference & Structured JSON Output

```python
from data.bigearthnet_txt.dataset import BigEarthNetDataset
from models.rs_internvl.config import RSInternVLConfig
from training.lora import load_lora_checkpoint
from transformers import AutoTokenizer

# 1. Load trained checkpoint
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

# 2. Perform multimodal inference
# (s1: [2, 120, 120], s2: [10, 120, 120])
output = model.predict(
    image_s1=s1_tensor,
    image_s2=s2_tensor,
    query="Is broad-leaved forest present in this area?",
    tokenizer=tokenizer,
)
print(output)
```

**Structured JSON Output Format:**
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

## 8. Verified Experimental Results

| Metric | Pretrained Baseline (Epoch 0) | Best Checkpoint (Epoch 10) | Final Overfit (Epoch 25) |
|---|---|---|---|
| **Validation Loss** | `7.2178` | **`1.1962`** | `1.4514` |
| **Train Loss** | — | `0.2470` | **`0.0012`** (-99.98%) |
| **Validation Binary Accuracy** | `0.0%` | **`71.43%`** | **`71.43%`** |
| **Validation Binary F1** | `0.0%` | **`83.3%`** | **`80.0%`** |
| **Train Binary Accuracy** | `0.0%` | `83.9%` | **`100.0%`** |
| **Validation Garbage Rate** | `37.5%` | **`0.0%`** | `25.0%` |

---

## 9. Running Regression Tests

Execute the full Member 1 regression test suite:
```bash
.venv/Scripts/pytest.exe tests/ -v
```

---

## 10. Known Limitations

- **Dataset Size:** Trained on the available 32 train / 8 val BigEarthNet samples. Large-scale multi-region scaling is scheduled for subsequent project phases.
- **Task Distribution:** Current split contains predominantly presence verification queries.
