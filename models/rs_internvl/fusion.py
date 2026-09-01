"""
Multimodal Token Fusion Module for RS-InternVL.

Assembles projected Sentinel-1 (SAR), Sentinel-2 (Multispectral), and text token embeddings
into an InternVL/Qwen2-compatible sequence with explicit attention masks, position IDs,
and proper causal language model loss masking (-100 on visual tokens).
"""

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn


class FusedMultimodalOutput(NamedTuple):
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    labels: Optional[torch.Tensor]
    n_s1_tokens: int
    n_s2_tokens: int
    n_visual_tokens: int


class MultimodalTokenFusion(nn.Module):
    """
    Multimodal sequence constructor for RS-InternVL.
    
    Constructs unified inputs_embeds sequence:
        [S1_Tokens (N_s1), S2_Tokens (N_s2), Text_Tokens (L_text)]
    
    Key Features:
    - Explicitly computes causal language model position IDs.
    - Constructs combined attention mask with full visibility over visual tokens.
    - Masks visual token positions in labels with -100 (PyTorch CrossEntropyLoss ignore_index)
      so the language model computes loss strictly on text targets.
    """

    def __init__(self, hidden_dim: int = 896):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(
        self,
        s1_tokens: torch.Tensor,
        s2_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_labels: Optional[torch.Tensor] = None,
        text_position_ids: Optional[torch.Tensor] = None,
    ) -> FusedMultimodalOutput:
        """
        Args:
            s1_tokens: Projected Sentinel-1 visual tokens [B, N_s1, D].
            s2_tokens: Projected Sentinel-2 visual tokens [B, N_s2, D].
            text_embeds: Embedded text instruction tokens [B, L_text, D].
            text_attention_mask: Optional text attention mask [B, L_text].
            text_labels: Optional ground-truth target token IDs [B, L_text].
            text_position_ids: Optional text position IDs.
            
        Returns:
            FusedMultimodalOutput with combined embeddings, attention mask, position IDs, and masked labels.
        """
        B, N_s1, D1 = s1_tokens.shape
        _, N_s2, D2 = s2_tokens.shape
        _, L_text, D_text = text_embeds.shape

        if D1 != D2 or D1 != D_text:
            raise ValueError(
                f"Embedding dimension mismatch: S1 has {D1}, S2 has {D2}, Text has {D_text}."
            )

        device = text_embeds.device
        dtype = text_embeds.dtype

        # 1. Concatenate Multimodal Token Embeddings: [B, N_s1 + N_s2 + L_text, D]
        inputs_embeds = torch.cat([s1_tokens, s2_tokens, text_embeds], dim=1)
        total_seq_len = N_s1 + N_s2 + L_text
        n_visual_tokens = N_s1 + N_s2

        # 2. Construct Combined Attention Mask: [B, Total_Seq_Len]
        # Visual tokens are always valid/active (mask value 1)
        visual_attention_mask = torch.ones((B, n_visual_tokens), dtype=torch.long, device=device)
        if text_attention_mask is not None:
            if text_attention_mask.ndim == 2:
                attention_mask = torch.cat([visual_attention_mask, text_attention_mask.to(device)], dim=1)
            else:
                attention_mask = text_attention_mask
        else:
            attention_mask = torch.ones((B, total_seq_len), dtype=torch.long, device=device)

        # 3. Construct Position IDs: [B, Total_Seq_Len]
        # Sequential position indices from 0 to total_seq_len - 1
        position_ids = torch.arange(total_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

        # 4. Construct Aligned Labels with Visual Token Masking (-100)
        # Visual tokens (S1 and S2) must NOT be predicted during autoregressive training
        fused_labels: Optional[torch.Tensor] = None
        if text_labels is not None:
            visual_ignore_labels = torch.full(
                (B, n_visual_tokens), -100, dtype=torch.long, device=device
            )
            fused_labels = torch.cat([visual_ignore_labels, text_labels.to(device)], dim=1)

        return FusedMultimodalOutput(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=fused_labels,
            n_s1_tokens=N_s1,
            n_s2_tokens=N_s2,
            n_visual_tokens=n_visual_tokens,
        )
