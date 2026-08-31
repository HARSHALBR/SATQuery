# RS-InternVL Model Architecture

**SATQuery AI — SIH 2026 Problem Statement 167 (Member 1: Core VLM & Data Engineering)**

---

## 1. Overview & Reference Architecture

`RS-InternVL` is a multi-sensor remote sensing vision-language architecture tailored for instruction-driven Earth observation reasoning over co-registered **Sentinel-1 SAR** (Synthetic Aperture Radar) and **Sentinel-2 Multispectral** imagery.

The architecture integrates two dedicated modality branches with an authentic **InternVL3-1B-class** causal language backbone ([`OpenGVLab/InternVL3-1B`](https://huggingface.co/OpenGVLab/InternVL3-1B)).

```text
Sentinel-1 SAR [B, 2, 120, 120] (VV, VH)
    ↓
S1 Encoder (4-Stage Conv-Residual Hierarchy, Task-specific init, Frozen for Step 2)
    ↓ [B, 225, 512]
S1 Projection (Linear 512 → LayerNorm → GELU → Linear 896, Trainable)
    ↓ [B, 225, 896]
                                 ┐
                                 ├──→ Multimodal Token Fusion → InternVL3-1B Language Model → Output Logits
                                 │    ([S1 Tokens, S2 Tokens, Text Tokens])
Sentinel-2 MS [B, 10, 120, 120]  ┘
    ↓
S2 Encoder (10-Channel Patch Embedder + Transformer Blocks, Task-specific init, Frozen for Step 2)
    ↓ [B, 225, 768]
S2 Projection (Linear 768 → LayerNorm → GELU → Linear 896, Trainable)
    ↓ [B, 225, 896]
```

---

## 2. Modality Encoder Branches

### A. Sentinel-1 SAR Branch (`models/rs_internvl/s1_encoder.py`)
- **Input Channels**: 2 polarizations (`VV`, `VH`).
- **Input Resolution**: $120 \times 120$ pixels (10m spatial resolution).
- **Architecture**: 4-stage residual convolutional feature extractor.
  - Stage 1: Conv $3\times3$, stride 2 ($120 \to 60$) $\to$ 64 channels
  - Stage 2: Residual Block, stride 2 ($60 \to 30$) $\to$ 128 channels
  - Stage 3: Residual Block, stride 2 ($30 \to 15$) $\to$ 256 channels
  - Stage 4: Pointwise Conv $1\times1$ + LayerNorm $\to$ 512 channels
- **Output Token Grid**: $15 \times 15 = 225$ spatial tokens of dimension $512$.
- **Initialization & Freezing**: Initialized with standard Kaiming normal weights and frozen by default (`freeze_s1_encoder=True`).

### B. Sentinel-2 Multispectral Branch (`models/rs_internvl/s2_encoder.py`)
- **Input Channels**: 10 optical bands strictly selected from 10m and 20m resolutions (60m bands `B01` and `B09` excluded):
  - 10m bands: `B02` (Blue), `B03` (Green), `B04` (Red), `B08` (NIR)
  - 20m bands: `B05` (RE1), `B06` (RE2), `B07` (RE3), `B8A` (Narrow NIR), `B11` (SWIR1), `B12` (SWIR2)
- **Input Resolution**: $120 \times 120$ pixels.
- **Architecture**: 10-channel patch embedding ($8\times 8$ patch stride) + learnable 2D positional embeddings + Pre-LayerNorm Transformer Encoder blocks with multi-head self-attention.
- **Output Token Grid**: $15 \times 15 = 225$ spatial tokens of dimension $768$.
- **Initialization & Freezing**: Initialized with standard truncated normal weights and frozen by default (`freeze_s2_encoder=True`).

---

## 3. Modality Projection Layers (`models/rs_internvl/projection.py`)

Modality projection layers map the modality-specific token representations into the common language model embedding space ($\text{dim} = 896$ dynamically queried from the InternVL3-1B configuration):

$$\text{Tokens}_{\text{S1}}^{\text{proj}} = W_2^{(1)} \cdot \text{GELU}\left(\text{LayerNorm}\left(W_1^{(1)} \cdot \text{Tokens}_{\text{S1}} + b_1^{(1)}\right)\right) + b_2^{(1)}$$

$$\text{Tokens}_{\text{S2}}^{\text{proj}} = W_2^{(2)} \cdot \text{GELU}\left(\text{LayerNorm}\left(W_1^{(2)} \cdot \text{Tokens}_{\text{S2}} + b_1^{(2)}\right)\right) + b_2^{(2)}$$

- `S1Projection`: $512 \to 1024 \to 896$ (Trainable)
- `S2Projection`: $768 \to 1024 \to 896$ (Trainable)

---

## 4. Multimodal Token Fusion & Causal Language Semantics (`models/rs_internvl/fusion.py`)

1. **Multimodal Sequence Construction**:
   $$\text{inputs\_embeds} = \left[ \text{S1 Tokens } (225), \text{S2 Tokens } (225), \text{Text Embeddings } (L_{\text{text}}) \right]$$
   $$\text{Shape: } [B, 450 + L_{\text{text}}, 896]$$

2. **Attention Mask**:
   Visual tokens are fully visible (mask $= 1$), concatenated with the text attention mask:
   $$\text{attention\_mask} = \left[ \mathbf{1}^{B \times 450}, \text{text\_attention\_mask}^{B \times L_{\text{text}}} \right]$$

3. **Position IDs**:
   Sequential positional indices from $0$ to $450 + L_{\text{text}} - 1$.

4. **Visual Token Loss Masking**:
   Visual token positions ($0 \dots 449$) are assigned label value **`-100`** (`CrossEntropyLoss` ignore index). The model calculates loss strictly on the autoregressive text tokens.

---

## 5. InternVL3-1B Language Model Integration (`models/rs_internvl/model.py`)

- **Language Model Backbone**: `Qwen2ForCausalLM` parameterized by the exact configuration of `OpenGVLab/InternVL3-1B`.
- **Dynamic Configuration Loading**: Reads `hidden_size` ($896$), `vocab_size` ($151674$), `num_hidden_layers` ($24$), `num_attention_heads` ($14$), `num_key_value_heads` ($2$), and `intermediate_size` ($4864$) directly from [`configs/model/internvl3_1b_config.json`](file:///e:/sih2026/configs/model/internvl3_1b_config.json).

### Input/Output Contract

#### `forward(...)` Method
```python
outputs = model(
    image_s1=s1_tensor,     # [B, 2, 120, 120], float32
    image_s2=s2_tensor,     # [B, 10, 120, 120], float32
    input_ids=input_ids,    # [B, L_text], int64
    labels=labels           # [B, L_text], int64 (optional)
)
```

**Returned Object**:
```python
{
    "logits": torch.Tensor,       # [B, 450 + L_text, 151674]
    "loss": Optional[torch.Tensor],
    "s1_features": torch.Tensor,   # [B, 225, 512]
    "s2_features": torch.Tensor,   # [B, 225, 768]
    "s1_projected": torch.Tensor,  # [B, 225, 896]
    "s2_projected": torch.Tensor,  # [B, 225, 896]
    "fused_features": torch.Tensor,# [B, 450 + L_text, 896]
    "n_visual_tokens": 450
}
```

#### `predict(...)` Interface Contract
```python
prediction = model.predict(
    image_s1=s1_tensor,
    image_s2=s2_tensor,
    query="Is coniferous forest present?"
)
```

**Returned Structure**:
```python
{
    "answer": "...",
    "claim": "...",
    "claim_type": "...",
    "model_score": 0.9845,
    "model_version": "RS-InternVL3-1B-v1",
    "grounding": None
}
```
