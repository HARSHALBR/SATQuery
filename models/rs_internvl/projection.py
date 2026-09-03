"""
Modality projection layers for RS-InternVL.

Maps task-specific Sentinel-1 (SAR) and Sentinel-2 (Multispectral) visual token dimensions
into the authentic common InternVL3-1B language model embedding space.
"""

from typing import Optional

import torch
import torch.nn as nn


class ModalityProjection(nn.Module):
    """
    Non-linear 2-layer MLP projection module mapping visual encoder tokens
    into the language model hidden dimension.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_dim: Dimension of input modality features.
            out_dim: Target LLM embedding dimension (dynamically read from InternVL config).
            hidden_dim: Intermediate projection dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual feature tokens [B, N, in_dim].
            
        Returns:
            Projected tokens in LLM space [B, N, out_dim].
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, N, in_dim], got {x.shape}")
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected feature dimension {self.in_dim}, got {x.shape[-1]}")

        return self.net(x)


class S1Projection(ModalityProjection):
    """
    Sentinel-1 SAR Modality Projection Layer.
    Maps S1 features [B, N_s1, s1_hidden_dim] -> [B, N_s1, llm_hidden_dim].
    """

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 896,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )


class S2Projection(ModalityProjection):
    """
    Sentinel-2 Multispectral Modality Projection Layer.
    Maps S2 features [B, N_s2, s2_hidden_dim] -> [B, N_s2, llm_hidden_dim].
    """

    def __init__(
        self,
        in_dim: int = 768,
        out_dim: int = 896,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
